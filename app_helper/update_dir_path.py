import os
import re

import numpy as np

# Settings persisted at runtime go to untracked override files so app use
# never dirties the tracked configuration and `git pull --ff-only` works.
_OVERRIDES = {
    'tools/config/config.py': 'tools/config/user_settings.py',
    'tools/config/paths.py': 'tools/config/user_paths.py',
}
_HEADER = ("# Written by InfernoSaber at runtime: overrides for the tracked\n"
           "# defaults; delete a line to revert it. Derived settings keep\n"
           "# following the tracked values unless set explicitly here.\n"
           "import numpy as np\n\n")


def _target(file_path):
    """Map tracked config files to their untracked override file."""
    normalized = file_path.replace('\\', '/')
    for tracked, override in _OVERRIDES.items():
        if normalized.endswith(tracked):
            return normalized[:-len(tracked)] + override, True
    return file_path, False


def _formatted(new_value):
    if isinstance(new_value, np.ndarray):
        return f"np.array({new_value.tolist()!r})"
    if isinstance(new_value, str):
        return f'"{new_value}"'
    return str(new_value)


def update_dir_path(file_path, keyword='dir_path', new_value=''):
    """
    Persist ``keyword = new_value`` in the given file. Writes targeting the
    tracked ``tools/config/config.py``/``paths.py`` redirect to the untracked
    override files, which are created or appended to as needed.

    Args:
        file_path (str): Path to the file to be updated.
        keyword (str): The keyword to search for in the file.
        new_value (str, int, float, bool): The new content to set.
    """
    file_path, is_override = _target(file_path)
    try:
        lines = []
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
        elif not is_override:
            raise FileNotFoundError(file_path)

        pattern = re.compile(rf"^\s*{keyword}\s*=")
        out, found_it = list(lines), False
        for index, line in enumerate(lines):
            if not found_it and pattern.match(line):
                out[index] = f'{keyword} = {_formatted(new_value)}\n'
                found_it = True
        if not found_it:
            if not is_override:
                print(f"Error: Could not find keyword {keyword} in configuration file.")
                return
            if not lines:
                out = [_HEADER]
            out.append(f'{keyword} = {_formatted(new_value)}\n')

        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'w') as file:
            file.writelines(out)
        print(f"Updated {keyword} in {file_path} to: {new_value}")

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
