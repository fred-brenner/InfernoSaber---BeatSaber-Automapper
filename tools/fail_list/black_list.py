from tools.config import paths
import os


# Append name of failed title without ending (e.g. no .dat)
def append_fail(name):
    # check if already on black list
    on_black_list = False
    try:
        with open(paths.black_list_file, 'r') as f:
            # black_list exists
            black_list = f.readlines()
            if name + '\n' in black_list:
                on_black_list = True
    except:
        pass

    if not on_black_list:
        # open black_list
        with open(paths.black_list_file, 'a') as f:
            # save title name
            f.writelines(name + '\n')


def delete_fails():
    with open(paths.black_list_file, 'r') as f:
        black_list = f.readlines()

    # print("Attention: Removing is irreversible. Check file names. ")
    # remove \n from black list elements
    for idx, el in enumerate(black_list):
        if "\n" in el:
            black_list[idx] = black_list[idx][:-1]

    check = [paths.copy_path_map, paths.dict_all_path, paths.copy_path_song]
    endings = ['.egg', '.dat', '_info.dat', '_events.dat', '_obstacles.dat', '_notes.dat']

    # stack list for endings
    black_list_end = []
    for el in black_list:
        for ending in endings:
            black_list_end.append(el + ending)

    for check_path in check:
        for file_name in os.listdir(check_path):
            if file_name in black_list_end:
                print(f"Delete {file_name}")
                os.remove(check_path + file_name)


if __name__ == '__main__':
    q = input("Reset black list? [y or n]")
    if q == 'y':
        os.remove(paths.black_list_file)
        print("Black list deleted.")
