import os
from pathlib import Path
from typing import Optional

import modal

app = modal.App("cs336-profiling")
traces = modal.Volume.from_name("cs336-traces", create_if_missing=True)
TRACE_DIR = Path("/traces")

# Define the image with requirements
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

@app.function(**config, volumes={TRACE_DIR: traces})
def run_benchmark_with_profiler(mode: str, label: str):
    import torch
    import gc
    from uuid import uuid4
    from cs336_systems.benchmark import benchmark_model
    from cs336_systems.model_configs import load_all_configs
    
    output_dir = TRACE_DIR / f"{label}_{mode}" / str(uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = load_all_configs()
    config_dict = configs["medium"]

    print(f"Warming up CUDA for {mode}...")
    benchmark_model(configs=config_dict, w=3, n=0, mode=mode, device="cuda")
    torch.cuda.empty_cache()
    gc.collect()

    schedule = torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1)

    print(f"Profiling {mode}...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        # We need to inject prof.step() inside the benchmark loop, or we can just step it manually
        # Wait, benchmark_model doesn't take `profiler` argument in the user's code. 
        # But wait, torch.profiler can trace everything in the context, but schedule requires .step() 
        # If we just use a default non-scheduled profiler, we don't need step().
        pass
        
    # Since we can't easily inject prof.step() into the user's original benchmark_model 
    # without modifying it, we'll just run a non-scheduled profiler for the benchmark phase!
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        benchmark_model(configs=config_dict, w=0, n=3, mode=mode, device="cuda")

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda pth: pth.stat().st_mtime,
        reverse=True,
    )[0]

    print(f"Trace saved to {trace_path.relative_to(TRACE_DIR)}")
    return trace_path.read_text(), trace_path.relative_to(TRACE_DIR)


@app.local_entrypoint()
def main(mode: str = "forward"):
    print(f"Running profiler for {mode}...")
    results, remote_path = run_benchmark_with_profiler.remote(mode=mode, label="benchmark")
    output_path = Path("/tmp") / remote_path.name
    output_path.write_text(results)
    print(f"Trace saved locally at {output_path}")
    print("You can view the trace locally using Chrome (chrome://tracing) or Perfetto UI.")


# --- TensorBoard Server ---
tb_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "tensorboard==2.18.0", "torch_tb_profiler==0.4.3", "setuptools"
)

class VolumeMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        if (route := environ.get("PATH_INFO")) in ["/", "/modal-volume-reload"]:
            try:
                traces.reload()
            except Exception as e:
                print("Exception while re-loading traces: ", e)
            if route == "/modal-volume-reload":
                environ["PATH_INFO"] = "/"  # redirect
        return self.app(environ, start_response)

@app.function(
    volumes={TRACE_DIR: traces},
    image=tb_image,
    max_containers=1,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def tensorboard():
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=str(TRACE_DIR))
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
        experimental_middlewares=[VolumeMiddleware],
    )

    return wsgi_app._create_wsgi_app()

