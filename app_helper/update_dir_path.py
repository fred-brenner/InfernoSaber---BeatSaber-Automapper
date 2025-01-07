import re


def update_dir_path(file_path, keyword='dir_path', new_value=''):
    """
    Updates the line containing 'dir_path' in the specified file to the new new_value.

    Args:
        file_path (str): Path to the file to be updated.
        keyword (str): The new path/new_value to set.
        new_value (str): The new content to set.
    """
    if new_value.lower() == 'true':
        new_value = True
    elif new_value.lower() == 'false':
        new_value = False
    try:
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        found_it = False
        # Update the specific line containing 'dir_path'
        with open(file_path, 'w') as file:
            for line in lines:
                # Check if the line contains 'dir_path'
                if re.match(rf"^\s*{keyword}\s*=", line):
                    # Replace with the new value and ensure a newline is added
                    file.write(f'{keyword} = "{new_value}"\n')
                    print(f"Updated {keyword} in {file_path} to: {new_value}")
                    found_it = True
                else:
                    file.write(line)
        if not found_it:
            print(f"Error: Could not find keyword {keyword} in configuration file.")

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return


# Example usage
if __name__ == "__main__":
    file_path = "tools/config/paths.py"
    keyword = "dir_path"
    new_value = "/new/directory/path"
    update_dir_path(file_path, keyword, new_value)
