###########################################
# This script finds the value to a keyword
# in a BeatSaber string dictionary.
##########################################
# Normalize is used for cleaning non-chars
##########################################


def index_find_str(idx, line, search_string):
    idx = line.find(search_string, idx)
    idx = line.find(':', idx) + 1
    idx_end = line.find(',', idx)
    idx_end_help = line.find('}', idx)
    if idx_end > idx_end_help or idx_end == -1:
        # End of paragraph
        idx_end = idx_end_help

    return idx, idx_end


def normalize_song_name(song_name: str, check_out: bool) -> str:
    if check_out:
        # uncheck points for float values
        song_name = song_name.replace(".", "")
        song_name = song_name.replace("-", "")

    song_name = song_name.replace(",", "")
    song_name = song_name.replace('"', "")
    song_name = song_name.replace("'", "")
    # delete folder placeholder
    song_name = song_name.replace("/", "")
    song_name = song_name.replace("\\", "")
    song_name = song_name.replace("?", "")
    song_name = song_name.replace("<", "")
    song_name = song_name.replace(">", "")
    song_name = song_name.replace("!", "")
    song_name = song_name.replace("&", "")
    song_name = song_name.replace("%", "")
    song_name = song_name.replace(":", "")
    song_name = song_name.replace("*", "")

    if song_name[0] == ' ':
        song_name = song_name[1:]

    return song_name


def return_find_str(idx, line, search_string, check_out=True) -> (str, int):
    idx = line.find(search_string, idx)
    idx = line.find(':', idx) + 1
    idx_end = line.find(',', idx)
    idx_end_help = line.find('}', idx)
    if idx_end > idx_end_help or idx_end == -1:
        # End of paragraph
        idx_end = idx_end_help

    value = line[idx:idx_end]

    # remove unknown chars
    value = normalize_song_name(value, check_out)

    return value, idx_end
