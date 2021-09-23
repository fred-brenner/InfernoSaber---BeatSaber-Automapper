#######################################
# compare a string to a list of strings
# may return the matched string
#######################################

def str_compare(str, str_list, return_str, silent):
    str = str.lower()
    for s in str_list:
        if s in str:
            if not silent:
                print("Exclude: " + str + " | Found " + s)

            if return_str:
                return s
            else:
                return True
    # no similarity found
    return False
