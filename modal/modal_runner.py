import os
import subprocess
from pathlib import Path
import modal

# Define the Modal App
app = modal.App("cs336-assignment2-systems")

ASSIGNMENT_DIR = Path(__file__).parent.resolve()

# Container Image with CUDA 12.1, PyTorch 2.x, Triton, Ninja, nsys, etc.
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build", "cuda-nsight-systems-12-1")
    .pip_install(
        "torch",
        "triton",
        "ninja",
        "numpy",
        "pandas",
        "matplotlib",
        "humanfriendly",
        "tqdm",
        "wandb",
        "pytest",
        "regex",
        "einops",
        "uv",
    )
    .add_local_dir(str(ASSIGNMENT_DIR), remote_path="/root/assignment2-systems")
)


def _setup_environment():
    """Install local packages in editable mode without PyPI dependency lookup."""
    os.chdir("/root/assignment2-systems")
    if Path("/root/assignment2-systems/cs336_basics").exists():
        subprocess.run(["pip", "install", "--no-deps", "-e", "./cs336_basics"], check=True)
    subprocess.run(["pip", "install", "--no-deps", "-e", "."], check=True)


def _find_nsys() -> str:
    """Find the path to the nsys binary inside the CUDA container."""
    import shutil
    import glob

    nsys_bin = shutil.which("nsys")
    if nsys_bin:
        return nsys_bin

    candidates = [
        "/usr/local/cuda/bin/nsys",
        "/usr/local/cuda-12.1/bin/nsys",
        "/usr/bin/nsys",
    ]
    candidates.extend(glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys"))
    candidates.extend(glob.glob("/opt/nvidia/nsight-systems/bin/nsys"))

    for path in candidates:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "nsys binary was not found in PATH or standard CUDA directories. "
        "Make sure nsight-systems is installed in the container image."
    )


# Function: Multi-GPU Pytest (2x A10G GPUs)
@app.function(image=image, gpu="A10G:2", timeout=300)
def run_pytest(test_filter: str = ""):
    _setup_environment()
    cmd = ["pytest", "-v", "-s"] + (["-k", test_filter] if test_filter else [])
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode


# Function: Single-GPU Pytest (Cheap T4 GPU @ ~$0.59/hr)
@app.function(image=image, gpu="T4", timeout=180)
def run_pytest_t4(test_filter: str = ""):
    _setup_environment()
    cmd = ["pytest", "-v", "-s"] + (["-k", test_filter] if test_filter else [])
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode


# Function: Run Custom Script (1x A10G GPU)
@app.function(image=image, gpu="A10G", timeout=600)
def run_script(script_path: str):
    _setup_environment()
    cmd = ["python3", script_path]
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode


# Function: Run Nsight Systems Profiler & Download Trace File to Mac
@app.function(image=image, gpu="A10G", timeout=600)
def run_nsys_profile(script_path: str, nvtx_capture: str = "") -> bytes:
    _setup_environment()
    output_base = "/tmp/profile_output"
    rep_file = f"{output_base}.nsys-rep"

    if os.path.exists(rep_file):
        os.remove(rep_file)

    nsys_executable = _find_nsys()

    cmd = [
        nsys_executable,
        "profile",
        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        f"--output={output_base}",
        "--force-overwrite=true",
        "--env-var=LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so",
    ]
    if nvtx_capture:
        print(f"Note: Profiling entire run. Use the Nsight Systems GUI to filter for the '{nvtx_capture}' NVTX range.")
    
    cmd.extend(["python3", script_path])
    print(f"Running Nsight Profiler: {' '.join(cmd)}")
    
    subprocess.run(cmd, check=False)

    if os.path.exists(rep_file):
        print(f"✅ Generated trace report ({os.path.getsize(rep_file)} bytes)!")
        
        with open(rep_file, "rb") as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Expected nsys trace output file {rep_file} was not generated!")


# Function: Leaderboard (2x A100-80GB GPUs)
@app.function(image=image, gpu="A100:2", timeout=600)
def run_leaderboard(script_path: str = "cs336_systems/benchmark.py"):
    _setup_environment()
    cmd = ["python3", script_path]
    print(f"Executing Leaderboard: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode


# Local CLI Entrypoint
@app.local_entrypoint()
def main(
    test: str = "",
    script: str = "",
    nsys: str = "",
    capture: str = "",
    section2: bool = False,
    section3: bool = False,
    section4: bool = False,
    section5: bool = False,
    section6: bool = False,
    section7: bool = False,
    leaderboard: bool = False,
):
    """
    CS336 Assignment 2 Modal CLI Runner.

    Section Flags:
      --section4      : Run Section 4 Attention & Triton Unit Tests (1x T4 GPU)
      --section5      : Run Section 5 DDP Unit Tests (2x A10G GPUs)
      --section6      : Run Section 6 Sharded Optimizer Unit Tests (2x A10G GPUs)
      --section7      : Run Section 7 FSDP Unit Tests (2x A10G GPUs)
      --leaderboard   : Run Section 9 8B Leaderboard Benchmark (2x A100 GPUs)

    Profiling Flags:
      --nsys <script>           : Run Nsight Systems Profiler on GPU and download profile_output.nsys-rep to Mac
      --nsys <script> --capture <label> : Restrict Nsight profiling capture ONLY to NVTX range label (e.g. --capture benchmark)
      --script <path>           : Run arbitrary benchmarking script on A10G GPU
      --test <filter>           : Run specific pytest test by name
    """
    if leaderboard:
        print("⚠️  [BUDGET ALERT] Launching 8B Leaderboard Benchmark on 2x A100-80GB GPUs...")
        code = run_leaderboard.remote(script_path=script if script else "cs336_systems/benchmark.py")
        return

    if nsys:
        filter_msg = f" (filtering for NVTX range '{capture}')" if capture else ""
        print(f"📊 Running Nsight Systems Profiler on '{nsys}'{filter_msg}...")
        file_bytes = run_nsys_profile.remote(script_path=nsys, nvtx_capture=capture)
        local_output = ASSIGNMENT_DIR / "profile_output.nsys-rep"
        with open(local_output, "wb") as f:
            f.write(file_bytes)
        print(f"✅ Nsight Systems trace saved locally to: {local_output}")
        print("   Open this file in NVIDIA Nsight Systems GUI app on your Mac!")
        return

    if section4:
        print("🚀 Running Section 4 (Attention & Triton) Unit Tests on T4 GPU...")
        code = run_pytest_t4.remote(test_filter="test_attention or test_flash")
        return

    if section5:
        print("🚀 Running Section 5 (DDP) Unit Tests on 2x A10G GPUs...")
        code = run_pytest.remote(test_filter="test_ddp or test_distributed")
        return

    if section6:
        print("🚀 Running Section 6 (Sharded Optimizer) Unit Tests on 2x A10G GPUs...")
        code = run_pytest.remote(test_filter="test_sharded")
        return

    if section7:
        print("🚀 Running Section 7 (FSDP) Unit Tests on 2x A10G GPUs...")
        code = run_pytest.remote(test_filter="test_fsdp")
        return

    if section2 or section3:
        target_script = script if script else "cs336_systems/benchmark.py"
        print(f"🚀 Running Section 2/3 Script '{target_script}' on A10G GPU...")
        code = run_script.remote(script_path=target_script)
        return

    if script:
        print(f"🚀 Running script '{script}' on A10G GPU...")
        code = run_script.remote(script_path=script)
        return

    # Default pytest runner
    use_multi = any(kw in test for kw in ["ddp", "fsdp", "sharded", "dist", "parallel"])
    if use_multi:
        print(f"🚀 Running multi-GPU pytest ('{test}') on 2x A10G GPUs...")
        code = run_pytest.remote(test_filter=test)
    else:
        print(f"🚀 Running single-GPU pytest ('{test}') on T4 GPU...")
        code = run_pytest_t4.remote(test_filter=test)

    if code == 0:
        print("✅ Run completed successfully!")
    else:
        print(f"❌ Run failed with exit code {code}")
