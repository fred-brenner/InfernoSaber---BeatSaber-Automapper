import os

music_folder_path = r"C:\Users\frede\Desktop\BS_Automapper\Data\prediction\songs_predict\y_old"
list_of_artists = []

for i, song in enumerate(os.listdir(music_folder_path)):
    src = os.path.join(music_folder_path, song)
    ending = song[-4:]
    song = song[:-4]
    song_split = song.split(" - ")
    if len(song_split) == 2:
        artist = song_split[0]
        song = song_split[1]
        print(f"{i} | Artist: {artist}, Song: {song}")

        if artist in list_of_artists:
            print(f"Found artist {artist}. Continue.")
            continue
        if song in list_of_artists:
            print(f"Found wrong order. Correcting.")
            new_song = artist
            new_artist = song
        else:

            inp1 = input("Correct? (empty for yes, else specify artist)")
            if inp1 == "":
                list_of_artists.append(artist)
                continue
            else:
                new_artist = inp1
                inp1 = input("Song? (empty for old, else specify song name)")
                if inp1 != "":
                    new_song = inp1
                else:
                    new_song = song_split[1]
    else:
        print(f"{i} | Song: {song_split[0]}")
        new_artist = input("Update Artist name (optional)")
        new_song = input("Update song name (optional)")
        if new_song == "":
            if new_artist == "":
                continue
            else:
                new_song = song_split[0]

    if new_artist != "":
        if new_artist not in list_of_artists:
            list_of_artists.append(new_artist)
        new_song_name = f"{new_artist} - {new_song}{ending}"
    else:
        new_song_name = f"{new_song}{ending}"
    if new_song == "":
        print("Warning. Skipping unspecified song name")
        continue
    dst = os.path.join(music_folder_path, new_song_name)
    print(dst)
    os.rename(src, dst)
