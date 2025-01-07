import re

def update_dir_path(file_path, keyword='dir_path', new_value=''):
    """
    Updates the line containing 'dir_path' in the specified file to the new new_value.

    Args:
        file_path (str): Path to the file to be updated.
        keyword (str): The keyword to search for in the file.
        new_value (str, int, float, bool): The new content to set.
    """
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
                    # Determine the format based on the type of new_value
                    if isinstance(new_value, str):
                        formatted_value = f'"{new_value}"'
                    else:
                        formatted_value = str(new_value)
                    # Replace with the new value and ensure a newline is added
                    file.write(f'{keyword} = {formatted_value}\n')
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