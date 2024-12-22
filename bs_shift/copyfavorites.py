"""
Thanks to @trendy_ideology for creating this script!
Second script (2/2)
"""

import csv
import os
import shutil


def read_ids_from_csv(file_path):
    ids = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['ID'].strip():  # Ensuring the ID is not empty
                ids.append(row['ID'].strip())
    return ids


def copy_matching_folders(ids, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    folders = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]

    for folder in folders:
        folder_id = folder.split(' ')[0]  # Get the part before the first space
        folder_id = folder_id.split('(')[0]  # Or before the first parenthesis if needed
        if folder_id in ids:
            source_folder_path = os.path.join(source_dir, folder)
            target_folder_path = os.path.join(target_dir, folder)

            # Check if the folder already exists in the target to avoid overwriting
            if not os.path.exists(target_folder_path):
                shutil.copytree(source_folder_path, target_folder_path)
                print(f"Copied {folder} to {target_dir}")
            else:
                print(f"Skipped {folder} as it already exists in {target_dir}")


def main():
    csv_file = 'output.csv'
    source_directory = "E:/SteamLibrary/steamapps/common/Beat Saber/Beat Saber_Data/CustomLevels/"
    target_directory = "C:/Users/frede/Desktop/BS_Automapper/Data/training/favorites_bs_input/"

    ids = read_ids_from_csv(csv_file)
    copy_matching_folders(ids, source_directory, target_directory)


if __name__ == "__main__":
    main()
