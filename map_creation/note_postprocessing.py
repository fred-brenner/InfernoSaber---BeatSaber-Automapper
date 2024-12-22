from tools.config import config
import random


def remove_double_notes(notes_all):
    for idx, notes in enumerate(notes_all):
        # each "notes" is one
        type_covered = []
        remove_index = []
        for i in range(int(len(notes) / 4)):
            n_type = notes[i * 4 + 2]
            if n_type == 3:
                # ignore bombs?
                continue
            if n_type not in type_covered:
                type_covered.append(n_type)
            else:
                remove_index.append(i)
        if len(remove_index) > 0:
            remove_index.reverse()
            for r_index in remove_index:
                for _ in range(4):
                    notes.pop(r_index * 4)
            notes_all[idx] = notes

    if config.single_notes_only_strict_flag:
        for idx, notes in enumerate(notes_all):
            if len(notes) > 4:
                target_note = 0 if random.random() > config.single_notes_remove_lr else 1
                for i in range(int(len(notes) / 4)):
                    if notes[i*4+2] == target_note:
                        notes = notes[i*4:i*4+4]
                        break
                # else
                if len(notes) > 4:
                    notes = notes[0:4]
                notes_all[idx] = notes

    return notes_all
