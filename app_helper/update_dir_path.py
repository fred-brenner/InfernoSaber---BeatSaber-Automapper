import re
from typing import Any


def _format_value(new_value: Any) -> str:
    """Return a string representation suitable for assignment statements."""
    if isinstance(new_value, str):
        escaped = new_value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(new_value, bool):
        return "True" if new_value else "False"
    return str(new_value)


def update_dir_path(file_path, keyword='dir_path', new_value=''):
    """
    Update the line containing the keyword in the specified file to ``new_value``.

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
                # Check if the line contains the keyword (optionally prefixed with "self.")
                match = re.match(rf"^(?P<indent>\s*)(?P<prefix>self\.)?{re.escape(keyword)}\s*=", line)
                if match and not found_it:
                    indent = match.group('indent') or ''
                    prefix = match.group('prefix') or ''
                    formatted_value = _format_value(new_value)
                    # Replace with the new value and ensure a newline is added
                    file.write(f"{indent}{prefix}{keyword} = {formatted_value}\n")
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

