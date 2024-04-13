import os
from huggingface_hub import snapshot_download

from tools.config import config, paths
from tools.config.mapper_selection import get_full_model_path


def model_download(model_branch=None):
    model_name = "BierHerr/InfernoSaber"
    model_folder = paths.model_path
    if model_branch is None:
        model_branch = config.use_mapper_selection

    # Create folder if it doesn't exist
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Check if model already exists
    if len(os.listdir(model_folder)) > 5:
        print("Model is already setup. Skipping download")
    else:
        print(f"Downloading model: {model_branch} from huggingface...")
        snapshot_download(repo_id=model_name, revision=model_branch, repo_type='model',
                          local_dir=model_folder, local_dir_use_symlinks=False,
                          ignore_patterns=['.gitignore', '.gitattributes'])

    # check that model exists on the example of event generator
    _ = get_full_model_path(config.event_gen_version)
