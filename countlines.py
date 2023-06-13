import os
import re


def check_def(line):
    # match def functions at start of line and with optional tab
    m = re.match("^[\s]*\t?def .*$", line)
    if m:
        return True
    else:
        return False


def countlines(start, lines=0, header=True, begin_start=None):
    if header:
        print('{:>10} |{:>10} |{:>10} |{:>10} | {:<20}'.format('ADDED', 'TOTAL', 'CUR_FCT', 'FCT', 'FILE'))
        print('{:->11}|{:->11}|{:->11}|{:->11}|{:->20}'.format('', '', '', '', ''))

    global fct_count
    global script_count
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
                        if check_def(cur_line):
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


if __name__ == '__main__':
    py_path = os.getcwd()
    global fct_count
    fct_count = 0
    global script_count
    script_count = 0
    countlines(py_path)
    print(f"Total of {script_count} python scripts.")
