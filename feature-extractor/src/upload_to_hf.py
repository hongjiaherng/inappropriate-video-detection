import argparse
import os
import shutil
import glob

from huggingface_hub import HfApi

ROOT_PATH = os.path.abspath(os.path.join(__file__, "../.."))


def main(args):
    hf_dataset_name = args.hf_dataset
    feature_dir = os.path.abspath(args.feature_dir)
    path_in_repo = args.path_in_repo
    remove_after_uploading = bool(args.remove_after_uploading)
    hf_token = args.hf_token
    n_files = len(glob.glob(os.path.join(feature_dir, "**/*.npy"), recursive=True))

    api = HfApi(token=hf_token)

    uploaded_url = api.upload_folder(
        folder_path=feature_dir,
        path_in_repo=path_in_repo,
        repo_id=hf_dataset_name,
        repo_type="dataset",
        commit_message=f"Uploading {n_files} rgb features",
    )
    print(uploaded_url)

    if remove_after_uploading:
        shutil.rmtree(feature_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_dataset",
        type=str,
        help='HuggingFace dataset name that contains source videos to extract visual features from. Currently only supports ["jherng/xd-violence"].',
        choices=["jherng/xd-violence"],
        required=True,
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=os.path.join(ROOT_PATH, "data", "outputs", "i3d_rgb"),
        help="Directory to the extracted features locally.",
    )
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default="/".join(["data", "i3d_rgb"]),
        help="Relative path of the directory in the repo.",
    )
    parser.add_argument(
        "--remove_after_uploading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove files after uploading.",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HF Access Token (Write)"
    )

    args = parser.parse_args()
    main(args)
