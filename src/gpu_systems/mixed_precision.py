import torch
import torch.nn as nn

class ToyMixedPrecisionModel(nn.Module):
    """
    Toy model demonstrating PyTorch Autocast behavior under Mixed Precision:
    - Linear layers automatically cast activations to Float16/BFloat16 for speed.
    - LayerNorm forces precision back to Float32 for numerical stability.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_fc1 = self.fc1(x)
        print(f"  [Autocast Forward] fc1 output type: {out_fc1.dtype}")
        
        x = self.relu(out_fc1)
        out_ln = self.ln(x)
        print(f"  [Autocast Forward] LayerNorm output type (forced Float32): {out_ln.dtype}")
        
        out_fc2 = self.fc2(out_ln)
        print(f"  [Autocast Forward] fc2 output type: {out_fc2.dtype}")
        return out_fc2

def inspect_mixed_precision_types(device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Inspects activation, weight, loss, and gradient precision types during autocasting."""
    print(f"\n=== Mixed Precision Precision Inspection (device={device}, dtype={dtype}) ===")
    with torch.autocast(device_type=device, dtype=dtype):
        x = torch.rand(size=(12, 512), device=device)
        targets = torch.rand(size=(12, 512), device=device)
        model = ToyMixedPrecisionModel(512, 512).to(device)
        
        print(f"Model fc1 Weight Type: {model.fc1.weight.dtype}")
        print(f"Model LayerNorm Weight Type: {model.ln.weight.dtype}")
        
        preds = model(x)
        loss = nn.functional.cross_entropy(preds, targets)
        print(f"Loss Output Type: {loss.dtype}")
        
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient Type for {name:<15}: {param.grad.dtype}")

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if dev != "cpu":
        inspect_mixed_precision_types(device=dev)
    else:
        print("Skipping autocast test on CPU.")
