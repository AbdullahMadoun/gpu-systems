import os
from pathlib import Path
from typing import Optional
import modal

app = modal.App("cs336-nsys-profiling")

image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch~=2.11.0",
    "numpy",
    "matplotlib",
    "humanfriendly",
    "pytest>=8",
    "regex>=2026.3.32",
    "tqdm>=4.67.1",
    "wandb>=0.19.7",
    "pandas>=2",
    "ty>=0.0.29",
    "ruff>=0.15.10"
).add_local_dir("../assignment1-basics", remote_path="/assignment1-basics", copy=True).run_commands(
    "uv pip install --system -e /assignment1-basics"
).add_local_dir(".", remote_path="/assignment2-systems", copy=True).run_commands(
    "uv pip install --system -e /assignment2-systems"
)

config = {"gpu": "a10g", "image": image}

@app.function(**config)
def run_nsys_profiler(mode: str):
    import subprocess
    import shutil
    
    # We create a temporary script to run the benchmark exactly as you want it
    script = f"""
import torch
import gc
from gpu_systems.benchmark import benchmark_model
from gpu_systems.model_configs import load_all_configs

configs = load_all_configs()
config = configs["medium"]

print("Warming up...")
benchmark_model(configs=config, w=3, n=0, mode="{mode}", device="cuda")
torch.cuda.empty_cache()
gc.collect()

print("Profiling...")
benchmark_model(configs=config, w=0, n=3, mode="{mode}", device="cuda")
    """
    
    with open("/tmp/run_bench.py", "w") as f:
        f.write(script)
        
    print(f"Running nsys profiler for {mode}...")
    
    # Execute nsys profile
    env = os.environ.copy()
    env["LD_PRELOAD"] = "/usr/local/cuda/lib64/libcudart.so"
    
    res = subprocess.run([
        "/usr/local/cuda/bin/nsys", "profile",
        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        "--output=/tmp/nsys_output",
        "--force-overwrite=true",
        "python3", "/tmp/run_bench.py"
    ], env=env, capture_output=True, text=True)
    
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
        
    with open("/tmp/nsys_output.nsys-rep", "rb") as f:
        trace_data = f.read()
        
    return trace_data


@app.local_entrypoint()
def main(mode: str = "forward"):
    print(f"Running Nsight Systems (nsys) for {mode}...")
    trace_data = run_nsys_profiler.remote(mode=mode)
    
    output_path = Path.home() / "Desktop" / f"nsys_medium_{mode}.nsys-rep"
    output_path.write_bytes(trace_data)
    print(f"SUCCESS! Trace perfectly saved to your Mac at: {output_path}")

