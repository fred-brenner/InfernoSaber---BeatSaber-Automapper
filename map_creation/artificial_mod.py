# helper functions for changing the map apart from the original input
from random import random

from tools.config import config


def mirror_notes(n):
    def mirror_note_type(nt):
        if nt == 0:
            # Left (Red) Note to blue
            nt = 1
        elif nt == 1:
            # Right (Blue) Note to red
            nt = 0
        else:
            # stays the same for bombs (nt=3)
            pass
        return nt

    def mirror_note_index(ni):
        if ni == 0:
            ni = 3
        elif ni == 1:
            ni = 2
        elif ni == 2:
            ni = 1
        elif ni == 3:
            ni = 0
        return ni

    note_type = n[2::4]
    # check for a single note side
    if len(set(note_type)) == 1:
        note_type = note_type[0]
        new_nt = mirror_note_type(note_type)
        # change the note type
        for i in range(int(len(n) / 4)):
            n[2 + i * 4] = new_nt
        # inverse the lineIndex (column)
        for i in range(int(len(n) / 4)):
            new_index = mirror_note_index(n[0 + i * 4])
            n[0 + i * 4] = new_index
    return n


def gimme_more_notes(notes: list):
    more_note_probability = config.gimme_more_notes_prob
    more_note_prob_increase_diff = 5    # increase the probability linearly after this threshold
    if more_note_probability > 0:
        extra_note_prob = config.max_speed/4 - more_note_prob_increase_diff
        if extra_note_prob > 0:
            more_note_probability += 0.1 * 0.1*extra_note_prob
        # take list of notes and search for single ones
        for i, section in enumerate(notes):
            if more_note_probability >= random():
                # skip in case of empty notes
                if len(section) < 4:
                    continue
                section_mirror = mirror_notes(section.copy())
                notes[i].extend(section_mirror)

        return notes
