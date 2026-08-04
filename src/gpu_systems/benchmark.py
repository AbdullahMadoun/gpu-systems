import torch 
import timeit
import pandas as pd 
import gc 
from statistics import mean, stdev
from cs336_basics.model import Transformer
from tqdm import tqdm
from cs336_systems.model_configs import load_all_configs, generate_latex_table
from torch.profiler import record_function

def benchmark(config:dict, w:int, n:int, device = "mps"): 
    model = Transformer(**config).to(device)
    for _ in tqdm(range(w)): 
        x = torch.randint(low = 0, high = config["vocab_size"] , size=(4, config["context_length"] ), device = device)
        with torch.inference_mode():
            model(x)

    times = []
    for _ in tqdm(range(n)):
        x = torch.randint(low = 0, high = config["vocab_size"], size=(4, config["context_length"] ), device = device)
        torch.mps.synchronize()
        start_time = timeit.default_timer()
        with torch.inference_mode():
            model(x)
        end_time = timeit.default_timer()
        torch.mps.synchronize()
        
        times.append(end_time - start_time)
    time = mean(times)
    return time

import torch
import math
from typing import Optional
from collections.abc import Callable ,Iterable

from einops import reduce

def learning_rate_schedule(t, lr_min, lr_max, Tw, Tc):
    if t < Tw: 
        lr = t/ Tw * lr_max
    elif t < Tc: 
        lr = lr_min + 1/2* (1 + math.cos(((t- Tw)/(Tc - Tw))*math.pi))*(lr_max - lr_min)
    else: 
        lr = lr_min 
    return lr
def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: 
    """
    -log [softmax(oi)] = -log [e^oi / (sum of all e^oi)] = -(oi + log(sum of all e^oi) 

    """
    # Get oi
    input_max = torch.max(input, dim =-1, keepdim = True).values
    log_sm = input - input_max - torch.log(torch.sum(torch.exp(input - input_max) , dim = -1,  keepdim=True))
    target_logits = torch.gather(log_sm, dim = -1, index = target.unsqueeze(-1) )

    return reduce(-target_logits, "...  -> ", "mean")

class AdamW(torch.optim.Optimizer): 
    def __init__(self, params, lr = 0.001, betas =  (0.9, 0.95), eps = 1e-5 , weight_decay = 0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults )
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups: 
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            B1, B2 = group["betas"]
            for p in group["params"]:
                g = p.grad.data
                if g is None: 
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)
                
                at = lr * math.sqrt(1 - B2 ** t) / (1-B1**t)
              
                p.data -= lr* weight_decay* p.data
                m = B1* m + (1- B1) *g 
                v= B2* v + (1- B2) *g**2

                p.data -= at* m/ (torch.sqrt(v) + eps)
                state["m"] = m 
                state["v"] = v
                state["t"] = t+1
        return loss 

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads: 
        return
    total_norm_sq = torch.stack([(g ** 2).sum() for g in grads]).sum()
    l2 = torch.sqrt(total_norm_sq)
    if l2 < max_l2_norm: 
        return 
    scaling = max_l2_norm / (l2  + 1e-6)
    for g in grads: 
          g.mul_(scaling)
        
    
    
import timeit


def benchmark_model(configs: dict, w: int, n: int, mode: str, device="cuda"):
    """
    Benchmarks a Transformer model based on the specified mode.
    
    Args:
        config (dict): Hyperparameters for the model, optimizer, etc.
        w (int): Number of warm-up steps.
        n (int): Number of measured benchmark steps.
        mode (str): Execution mode. Options: 'forward', 'forward_backward', 'full'.
        device (str): Device to run benchmarking on.
    """
    assert mode in ["forward", "forward_backward", "full"], f"Invalid mode: {mode}"
    
    # 1. Initialize the model and optimizer
    config = configs["transformer"]
    model = Transformer(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 2. Warm-up Phase
    # We match the warmup execution path to the benchmarking path to accurately warm up the cache/allocator.
    if mode == "forward":
        model.eval()
        warmup_context = torch.no_grad()
    else:
        model.train()
        warmup_context = torch.enable_grad()

    with warmup_context:
        for _ in range(w):
            x = torch.randint(low=0, high=config["vocab_size"], size=(1, config["context_length"]), device=device)
            inputs, targets = x[:, :-1], x[:, 1:]
            
            pred = model(inputs)
            
            if mode in ["forward_backward", "full"]:
                loss = cross_entropy_loss(pred, targets)
                loss.backward()
                if mode == "full":
                    gradient_clipping(model.parameters(), **configs["clipping"])
                    opt.step()
                opt.zero_grad()

    # Clear lingering tracking states before the main loop
    opt.zero_grad()
    torch.cuda.synchronize()

    # 3. Benchmarking Phase
    step_times = []

    for step in range(n):
        # Generate a fresh random batch of data outside of timing
        x = torch.randint(low=0, high=config["vocab_size"], size=(1, config["context_length"]), device=device)
        inputs, targets = x[:, :-1], x[:, 1:]

        if mode == "forward":
            # --- FORWARD ONLY MODE ---
            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = timeit.default_timer()
                
                with record_function(mode):
                    pred = model(inputs)
                
                torch.cuda.synchronize()
                end_time = timeit.default_timer()

        elif mode == "forward_backward":
            # --- FORWARD AND BACKWARD MODE ---
            opt.zero_grad()
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            
            with record_function(mode):
                pred = model(inputs)
                loss = cross_entropy_loss(pred, targets)
                loss.backward()
            
            torch.cuda.synchronize()
            end_time = timeit.default_timer()

        elif mode == "full":
            # --- FULL TRAINING STEP MODE ---
            opt.zero_grad()
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            
            torch.cuda.nvtx.range_push(f"full_step_{step}")
            with record_function(mode):
                torch.cuda.nvtx.range_push("forward_pass")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(inputs)
                    loss = cross_entropy_loss(pred, targets)
                torch.cuda.nvtx.range_pop() # end forward_pass
                
                torch.cuda.nvtx.range_push("backward_pass")
                loss.backward()
                torch.cuda.nvtx.range_pop() # end backward_pass
                
                torch.cuda.nvtx.range_push("optimizer_step")
                gradient_clipping(model.parameters(), **configs["clipping"])
                opt.step()
                torch.cuda.nvtx.range_pop() # end optimizer_step
                
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.nvtx.range_pop() # end full_step
                
            torch.cuda.synchronize()
            end_time = timeit.default_timer()

        # Log total elapsed time for this step block
        step_times.append(end_time - start_time)

    return step_times


configs = load_all_configs()
df_list = []
for name,config in configs.items(): 
    # Skip huge models that will OOM an A10G and corrupt the nsys trace
    if name in ["10B", "xl"]:
        print(f"Skipping {name} to avoid OOM crash on A10G.")
        continue

    df = pd.DataFrame([config["transformer"]])
    for test_mode in ["forward", "forward_backward", "full"]:
        try:
            times = benchmark_model(configs=config, w=5, n=10, mode=test_mode, device="cuda")
            avg_time = sum(times) / len(times)
            std_time = stdev(times)
            print(f"Mode: {test_mode:<18} | {name:<6} | Avg Step Time: {avg_time:.6f} seconds")
            
            df[f"Time_{test_mode}"] = avg_time 
            df[f"STD_{test_mode}"] = std_time 
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM on {name} - skipping")
            else:
                raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            
    df_list.append(df)
torch.cuda.empty_cache()
gc.collect()

final_df = pd.concat(df_list, ignore_index=True)
print(final_df)
print(final_df.to_latex(index=False))
