import json
import pandas as pd
import os
from glob import glob

# Specifications of different model sizes
# Context length is 512 unless otherwise specified
configs = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 512},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 512},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20, "context_length": 512},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32, "context_length": 512},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36, "context_length": 512},
}

# The assignment 1 style boilerplate for configs
def get_assignment1_style_config(model_config):
    return {
        "transformer": {
            "vocab_size": 50257,
            "context_length": model_config["context_length"],
            "num_layers": model_config["num_layers"],
            "d_model": model_config["d_model"],
            "num_heads": model_config["num_heads"],
            "d_ff": model_config["d_ff"],
            "rope_theta": 10000.0,
            "device": 0
        },
        "optimizer": {
            "lr": 0.0005,
            "weight_decay": 0.1
        },
        "lr_schedule": {
            "lr_min": 0.0,
            "lr_max": 0.0005,
            "Tw": 1000,
            "Tc": 100000
        },
        "clipping": {
            "max_l2_norm": 1.0
        },
        "loop": {
            "epochs": 10,
            "batch_size": 32,
            "checkpoints": 5,
            "model_path": "checkpoint.pt"
        }
    }

get_model_config = get_assignment1_style_config
get_assignment_config = get_assignment1_style_config

def generate_json_configs(output_dir="configs"):
    os.makedirs(output_dir, exist_ok=True)
    
    for size, config in configs.items():
        full_config = get_assignment1_style_config(config)
        
        # Optionally update the model path based on size
        full_config["loop"]["model_path"] = f"checkpoint_{size}.pt"
        
        filepath = os.path.join(output_dir, f"config_{size}.json")
        with open(filepath, 'w') as f:
            json.dump(full_config, f, indent=4)
        print(f"Generated {filepath}")

def generate_latex_table(df):
    latex_caption = "Specifications of different model sizes. These are mostly based on GPT-2 configs."
    latex_table = df.to_latex(
        index=False, 
        caption=latex_caption, 
        label="tab:model_specs"
    )
    print("\n=== LaTeX Table ===")
    print(latex_table)

def load_all_configs(configs_dir="configs"):
    """
    Loads all JSON configurations from the specified directory.
    Returns a dictionary mapping the model size (e.g., 'small') to its config dict.
    """
    loaded_configs = {}
    
    # Find all config JSON files
    config_paths = glob(os.path.join(configs_dir, "config_*.json"))
    if not config_paths:
        generate_json_configs(output_dir=configs_dir)
        config_paths = glob(os.path.join(configs_dir, "config_*.json"))
    
    for path in config_paths:
        # Extract the size from the filename (e.g., "config_small.json" -> "small")
        filename = os.path.basename(path)
        size_name = filename.replace("config_", "").replace(".json", "")
        
        with open(path, "r") as f:
            config_data = json.load(f)
            loaded_configs[size_name] = config_data
            
    return loaded_configs

if __name__ == "__main__":
    generate_json_configs()
    
    # Test loading them
    print("\n=== Loaded Configs ===")
    loaded = load_all_configs()
    for size, config_data in loaded.items():
        print(f"Loaded {size} config with d_model = {config_data['transformer']['d_model']}")
