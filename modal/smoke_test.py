"""
Smoke test to verify the Modal container environment works correctly.
Tests: CUDA availability, nsys binary, package imports, config loading.
Run with: python3 -m modal run modal_runner.py --script cs336_systems/smoke_test.py
"""
import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  ✅ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False

print("=" * 60)
print("CS336 Modal Container Smoke Test")
print("=" * 60)

all_ok = True

# 1. Check CUDA
print("\n[1/6] CUDA")
all_ok &= check("torch.cuda.is_available()", lambda: __import__("torch").cuda.is_available())
all_ok &= check("GPU name", lambda: __import__("torch").cuda.get_device_name(0))
all_ok &= check("GPU memory (GB)", lambda: f"{__import__('torch').cuda.get_device_properties(0).total_mem / 1e9:.1f}")

# 2. Check nsys binary
print("\n[2/6] Nsight Systems (nsys)")
import shutil, glob, os
def find_nsys():
    nsys_bin = shutil.which("nsys")
    if nsys_bin:
        return nsys_bin
    for p in ["/usr/local/cuda/bin/nsys", "/usr/local/cuda-12.1/bin/nsys", "/usr/bin/nsys"]:
        if os.path.exists(p):
            return p
    for p in glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys"):
        if os.path.exists(p):
            return p
    return "NOT FOUND"
all_ok &= check("nsys binary", find_nsys)

# 3. Check nsys version to discover valid flags
print("\n[3/6] nsys version & supported flags")
import subprocess
nsys_path = find_nsys()
if nsys_path != "NOT FOUND":
    result = subprocess.run([nsys_path, "--version"], capture_output=True, text=True)
    version = (result.stdout + result.stderr).strip()
    print(f"  ℹ️  {version}")
    # Check which flags are supported
    help_result = subprocess.run([nsys_path, "profile", "--help"], capture_output=True, text=True)
    help_text = help_result.stdout + help_result.stderr
    for flag in ["--trace", "--nvtx-capture", "--python-backtrace", "--pytorch", "--cudabacktrace", "--gpu-metrics-devices"]:
        supported = flag in help_text
        status = "✅ supported" if supported else "⚠️  NOT supported"
        print(f"  {status}: {flag}")

# 4. Check key imports
print("\n[4/6] Python Package Imports")
for pkg in ["torch", "triton", "numpy", "pandas", "matplotlib", "einops", "wandb", "pytest", "tqdm"]:
    all_ok &= check(f"import {pkg}", lambda p=pkg: __import__(p).__name__)

# 5. Check cs336 packages
print("\n[5/6] CS336 Packages")
all_ok &= check("import cs336_basics", lambda: __import__("cs336_basics").__name__)
all_ok &= check("import cs336_systems", lambda: __import__("cs336_systems").__name__)
all_ok &= check("load_all_configs()", lambda: list(__import__("cs336_systems.model_configs", fromlist=["load_all_configs"]).load_all_configs().keys()))

# 6. Quick CUDA tensor test
print("\n[6/6] CUDA Tensor Smoke Test")
import torch
def cuda_matmul():
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    return f"shape={c.shape}, device={c.device}"
all_ok &= check("CUDA matmul", cuda_matmul)

print("\n" + "=" * 60)
if all_ok:
    print("🎉 ALL CHECKS PASSED — Container is ready!")
else:
    print("⚠️  Some checks failed — review above output.")
print("=" * 60)
