"""Generation entry point shared by the existing main scripts."""


def main(name_ar: list, debug_beats=False) -> bool:
    from map_creation.v2_generation import generate_song
    if len(name_ar) != 1:
        raise ValueError("Generate one song per call; main_multi schedules parallel songs")
    result = generate_song(name_ar[0], debug_beats=debug_beats)
    return result if debug_beats else False
