import re


def update_dir_path(file_path, input_dir):
    """
    Updates the line containing 'dir_path' in the specified file to the new value.

    Args:
        file_path (str): Path to the file to be updated.
        input_dir (str): The new directory path to set.
    """
    try:
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Update the specific line containing 'dir_path'
        with open(file_path, 'w') as file:
            for line in lines:
                # Check if the line contains 'dir_path'
                if re.match(r"^\s*dir_path\s*=", line):
                    # Replace with the new value
                    file.write(f'dir_path = "{input_dir}"')
                else:
                    file.write(line)
                    print(f"Updated 'dir_path' in {file_path} to: {input_dir}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    file_path = "tools/config/paths.py"
    input_dir = "/new/directory/path/"
    update_dir_path(file_path, input_dir)