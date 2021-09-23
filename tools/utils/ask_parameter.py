####################################################
# This script creates a prompt and checks input
# Parameter type can be 'int', 'float', 'string' or
# custom list of variables e.g. [5.1, 'yes']
####################################################
# Author: Frederic Brenner
# Email: frederic.brenner@tum.de
####################################################
# Date: 04.2020
####################################################

def ask_parameter(parameter, param_type=None):
    inp = None
    tries = 2  # try max <x> times
    input_flag = True
    while input_flag:
        if tries == 0:
            print("Too many false inputs. \nExit")
            exit()
        inp = input('Enter value for {}: '.format(parameter))
        if inp == '':
            print("Empty input not allowed.")
            tries -= 1
            continue

        if param_type is not None:

            # positive integer > 0
            if param_type == 'uint':
                if inp.isdigit():
                    inp = int(inp)
                    if inp > 0:
                        input_flag = False  # finished
                    else:
                        print("Wrong input, must be greater zero.")
                else:
                    print("Wrong input, only unsigned integer allowed.")

            elif param_type == 'int':
                if inp.isdigit():
                    inp = int(inp)
                    input_flag = False  # finished
                elif inp.startswith('-') and inp[1:].isdigit():
                    inp = int(inp)
                    input_flag = False  # finished
                else:
                    print("Wrong input, only integer allowed.")

            elif param_type == 'float':
                if inp.replace('.', '', 1).isdigit():
                    inp = float(inp)
                    input_flag = False  # finished
                else:
                    print("Wrong input, only float allowed.")

            elif param_type == 'string':
                if not inp.isdigit() and not inp.replace('.', '', 1).isdigit():
                    input_flag = False  # finished
                else:
                    print("Wrong input, only string allowed.")
            else:
                if inp.isdigit():
                    inp = int(inp)
                elif inp.replace('.', '', 1).isdigit():
                    inp = float(inp)
                # list of possible values
                if inp in param_type:
                    input_flag = False  # finished
                else:
                    print("Wrong input, only {} allowed.".format(param_type))
        else:
            # No parameter check defined -> finished
            input_flag = False

        tries -= 1

    print('')
    return inp


if __name__ == '__main__':
    # Test
    parameter = 'test_parameter'
    # param_type = 'int'
    param_type = [0, 1.5, 'test']
    test = ask_parameter(parameter, param_type)
    print('{} = {}'.format(parameter, test))
