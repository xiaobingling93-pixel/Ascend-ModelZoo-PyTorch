import argparse
import os
import shutil
import yaml


def get_local_weight_folder():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    upper_directory = current_dir
    for _ in range(2):
        upper_directory = os.path.dirname(upper_directory)
    return upper_directory


def check_if_weights_exist(local_weights_folder):
    for _, _, files in os.walk(local_weights_folder):
        for file in files:
            if file.endswith('.safetensors'):
                print(f"Detected .safetensors file in {local_weights_folder}.")
                return True
    return False


def read_weights_url(file_path) -> dict[str, str]:
    weight_dict = {}
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            print(f"Parsing from weights_url.yaml...")
            for hub, link in config.items():
                print(f"{hub}: {link}")
                weight_dict[hub] = link
        return weight_dict
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error in configuration file: {exc}")
        return {}


def change_files_permissions(directory: str, mode: int) -> None:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified path does not exist: {directory}")
    
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"The specified path is not a directory: {directory}")
    
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                file_path = entry.path
                if entry.is_file():
                    try:
                        os.chmod(file_path, mode)
                        print(f"Permissions for '{file_path}' have been successfully changed to {oct(mode)}.")
                    except PermissionError as e:
                        print(f"Permission denied while trying to change permissions for '{file_path}': {e}")
                    except Exception as e:
                        print(f"An error occurred while changing permissions for '{file_path}': {e}")
                elif entry.is_dir():
                    print(f"Skipping directory: {file_path}")

    except Exception as e:
        print(f"An error occurred while processing the directory: {e}")
        raise


def download_weights(weight_dict, local_weights_folder):
    if download_huggingface(weight_dict.get("HuggingFace"), local_weights_folder):
        print(f"Successfully downloaded {weight_dict.get('HuggingFace')} weights from HuggingFace.")
    elif download_modelscope(weight_dict.get("ModelScope"), local_weights_folder):
        print(f"Successfully downloaded {weight_dict.get('ModelScope')} weights from ModelScope.")
    elif download_modelers(weight_dict.get("Modelers"), local_weights_folder):
        print(f"Successfully downloaded {weight_dict.get('Modelers')} weights from Modelers.")
    else:
        print(f"Failed to download weights from all platforms. Please prepare locally.")


def download_huggingface(huggingface_url, local_weights_folder):
    from huggingface_hub import snapshot_download
    try:
        print("Downloading from HuggingFace...")
        snapshot_download(
            repo_id=huggingface_url,
            local_dir=local_weights_folder
        )
        shutil.rmtree(os.path.join(local_weights_folder, ".cache"))
        return True
    except Exception as e:
        print(f"An error occurred while downloading the model: {e}")
        return False


def download_modelscope(modelscope_url, local_weights_folder):
    from modelscope.hub.snapshot_download import snapshot_download
    try:
        print("Downloading from ModelScope...")
        snapshot_download(
            modelscope_url,
            local_dir=local_weights_folder
        )
        shutil.rmtree(os.path.join(local_weights_folder, "._____temp"))
        return True
    except Exception as e:
        print(f"An error occurred while downloading the model: {e}")
        return False


def download_modelers(modelers_url, local_weights_folder):
    from openmind_hub import snapshot_download
    try:
        print("Downloading from Modelers...")
        snapshot_download(
            repo_id=modelers_url,
            local_dir=local_weights_folder
        )
        return True
    except Exception as e:
        print(f"An error occurred while downloading the model: {e}")
        return False


def main():
    yaml_file_path = 'weights_url.yaml'
    local_weights_folder = get_local_weight_folder()
    weight_dict = read_weights_url(yaml_file_path)

    parser = argparse.ArgumentParser(description="Download models from different hubs.")
    parser.add_argument(
        '--hub', 
        type=str,
        required=False,
        choices=['huggingface', 'modelscope', 'modelers'],
        help='Specify the hub from which to download the model.'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        required=False,
        help='The repository ID or model name to download.'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default=local_weights_folder,
        help='The target directory where the model will be saved. Defaults to "./downloads".'
    )
    args = parser.parse_args()

    if check_if_weights_exist(args.target_dir):
        print("Using local weights, skip downloading.")

    if not args.hub:
        download_weights(weight_dict, args.target_dir)
    elif args.hub == 'huggingface':
        download_huggingface(args.repo_id, args.target_dir)
    elif args.hub == 'modelscope':
        download_modelscope(args.repo_id, args.target_dir)
    elif args.hub == 'modelers':
        download_modelers(args.repo_id, args.target_dir)
    else:
        print(f"Unsupported hub: {args.hub}")
    change_files_permissions(args.target_dir, 0o750)


if __name__ == "__main__":
    main()