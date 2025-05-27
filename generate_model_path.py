import os
import yaml

PRETRAINED_DIR = "pretrained"
OUTPUT_YAML_FILE = "model_path.yaml"
EXPECTED_CONFIG_FILE = "config.json"
# Common model weight file names
MODEL_WEIGHT_FILES = ["pytorch_model.bin", "model.safetensors"]


def is_valid_model_dir(dir_path):
    """
    Checks if a directory appears to be a valid Hugging Face model directory.
    A directory is considered valid if it contains:
    1. A 'config.json' file.
    2. At least one of the common model weight files (e.g., pytorch_model.bin, model.safetensors).
    """
    if not os.path.isdir(dir_path):
        return False

    has_config_json = os.path.exists(
        os.path.join(dir_path, EXPECTED_CONFIG_FILE))
    if not has_config_json:
        return False

    has_model_weights = any(os.path.exists(
        os.path.join(dir_path, wf)) for wf in MODEL_WEIGHT_FILES)
    if not has_model_weights:
        return False

    return True


def scan_pretrained_directory():
    """
    Scans the PRETRAINED_DIR for valid Hugging Face model directories.
    """
    valid_model_paths = []
    if not os.path.isdir(PRETRAINED_DIR):
        print(
            f"Directory '{PRETRAINED_DIR}' not found. Cannot scan for models.")
        return valid_model_paths

    for item_name in os.listdir(PRETRAINED_DIR):
        potential_model_path = os.path.join(PRETRAINED_DIR, item_name)
        if is_valid_model_dir(potential_model_path):
            # Store relative paths
            valid_model_paths.append(potential_model_path)
            print(f"Found valid model directory: {potential_model_path}")
        else:
            # Only log if it was a dir but not valid model
            if os.path.isdir(potential_model_path):
                print(
                    f"Skipping directory (not a valid model format): {potential_model_path}")

    return valid_model_paths


def write_paths_to_yaml(paths_list, output_file):
    """
    Writes the list of model paths to a YAML file.
    """
    data_to_write = {"available_model_paths": paths_list}
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data_to_write, f, allow_unicode=True, sort_keys=False)
        print(
            f"Successfully wrote {len(paths_list)} model paths to {output_file}")
    except IOError as e:
        print(f"Error writing to YAML file {output_file}: {e}")


if __name__ == "__main__":
    print(f"Scanning for model paths in ./{PRETRAINED_DIR}...")
    found_paths = scan_pretrained_directory()

    if found_paths:
        write_paths_to_yaml(found_paths, OUTPUT_YAML_FILE)
    else:
        print("No valid model paths found.")
        # Optionally, still write an empty list to the YAML file
        write_paths_to_yaml([], OUTPUT_YAML_FILE)

    print("Script finished.")
