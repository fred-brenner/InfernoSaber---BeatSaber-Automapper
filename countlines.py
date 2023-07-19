import os
import re
import numpy as np


def check_def(line):
    # match def functions at start of line and with optional tab
    m = re.match("^[\s]*\t?def (\w+)", line)
    if m:
        return m.group(1)
    else:
        return None


def countlines(start, lines=0, header=True, begin_start=None):
    if header:
        print('{:>10} |{:>10} |{:>10} |{:>10} | {:<20}'.format('ADDED', 'TOTAL', 'CUR_FCT', 'FCT', 'FILE'))
        print('{:->11}|{:->11}|{:->11}|{:->11}|{:->20}'.format('', '', '', '', ''))

    global fct_count
    global script_count
    self_defined_functions = set()
    for thing in os.listdir(start):
        thing = os.path.join(start, thing)
        if os.path.isfile(thing):
            if thing.endswith('.py'):
                with open(thing, 'r') as f:
                    cur_newlines = f.readlines()
                    newlines = len(cur_newlines)
                    lines += newlines

                    if begin_start is not None:
                        reldir_of_thing = '.' + thing.replace(begin_start, '')
                    else:
                        reldir_of_thing = '.' + thing.replace(start, '')

                    cur_fct_count = 0
                    for cur_line in cur_newlines:
                        function_name = check_def(cur_line)
                        if function_name:
                            self_defined_functions.add(function_name)
                            cur_fct_count += 1
                    fct_count += cur_fct_count

                    script_count += 1

                    print('{:>10} |{:>10} |{:>10} |{:>10} | {:<20}'.format(
                        newlines, lines, cur_fct_count, fct_count, reldir_of_thing))

    for thing in os.listdir(start):
        thing = os.path.join(start, thing)
        if os.path.isdir(thing):
            lines = countlines(thing, lines, header=False, begin_start=start)

    return lines


def count_function_calls(file_path, self_defined_functions):
    function_counts = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            matches = re.findall(r'(\w+)\(', line)
            for match in matches:
                if match in self_defined_functions:
                    if match in function_counts:
                        function_counts[match] += 1
                    else:
                        function_counts[match] = 1
    return function_counts


def display_function_usage(project_path):
    print("\nFunction call counts for self-defined functions:")
    self_defined_functions = set()
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        function_name = check_def(line)
                        if function_name:
                            self_defined_functions.add(function_name)

    function_counts = {}
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                counts = count_function_calls(file_path, self_defined_functions)
                for function, count in counts.items():
                    if function in function_counts:
                        function_counts[function] += count
                    else:
                        function_counts[function] = count

    sorted_counts = sorted(function_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"Function calls mean: {np.mean(list(function_counts.values()))}")
    print("Top ten:")
    i = 0
    for function, count in sorted_counts:
        i += 1
        if i > 10:
            break
        print(f"Function {function:<30} is called {count:>3} times in the project.")


if __name__ == '__main__':
    py_path = os.getcwd()
    global fct_count
    fct_count = 0
    global script_count
    script_count = 0
    countlines(py_path)
    print(f"Total of {script_count} python scripts.")

    display_function_usage(py_path)
