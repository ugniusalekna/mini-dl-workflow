import os
import zipfile
import requests
import argparse

try:
    import inquirer
except ImportError:
    inquirer = None


DATASETS = {
    "cifar10": "1PKd-2Ou5IwXBerbnHEl9DRlB80Aby0W_",
    "cifar100": "1MWbKrbL1FNbUfY4R2xnbpXxs2xnRgkPb",
    "EMNIST": "1yUqBM-QCyjdpsrJQLCLtf1NxyipNStGo",
    "FER2013": "1pR7US6NuBRcQVFZme5FV6eBr5LrARz2Y",
    "QuickDraw": "13aA23PBAYZDDeGiJl6qFeryVkhQXN34A",
    "Download ALL": "ALL"
}


def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    with requests.get(URL, stream=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                f.write(chunk)


def extract_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(file_path)


def download_and_extract(dataset_name, file_id, data_dir):
    file_path = f"{data_dir}/{dataset_name}.zip"
    print(f"Downloading {dataset_name}...")
    download_file_from_google_drive(file_id, file_path)
    print(f"Extracting {dataset_name}...")
    extract_zip(file_path, data_dir)
    print(f"Dataset is ready in {data_dir}/{dataset_name}")


def colab_mode():
    print("\nSelect a dataset to download:")
    for i, dataset in enumerate(DATASETS.keys()):
        print(f"{i+1}. {dataset}")

    choice = input("\nEnter the number of the dataset to download: ").strip()

    try:
        choice = int(choice)
        dataset_names = list(DATASETS.keys())
        if 1 <= choice <= len(dataset_names):
            dataset_name = dataset_names[choice - 1]
            return dataset_name
        else:
            print("Invalid choice. Please enter a valid number.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None


def terminal_mode():
    if inquirer is None:
        print("`inquirer` is not installed.")
        exit(1)

    questions = [
        inquirer.List(
            "dataset",
            message="Select a dataset to download:",
            choices=list(DATASETS.keys()),
        )
    ]
    answer = inquirer.prompt(questions)
    return answer["dataset"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--colab", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    dataset_name = colab_mode() if args.colab else terminal_mode()

    if dataset_name is None:
        print("No valid selection. Exiting.")
        return

    if dataset_name == "Download ALL":
        for name, file_id in DATASETS.items():
            if file_id != "ALL":
                download_and_extract(name, file_id, data_dir)
        print("All datasets have been downloaded and extracted.")
    else:
        download_and_extract(dataset_name, DATASETS[dataset_name], data_dir)


if __name__ == "__main__":
    main()
