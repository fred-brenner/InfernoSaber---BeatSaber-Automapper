"""
Thanks to @trendy_ideology for creating this script!
First script (1/2)
"""

import requests
import csv
import json


def query_beatsaver_api(hash_value):
    url = f"https://api.beatsaver.com/maps/hash/{hash_value}"
    response = requests.get(url, headers={"accept": "application/json"})
    if response.status_code == 200:
        data = response.json()
        return {
            "ID": data.get("id", ""),
            "SongName": data["metadata"]["songName"],
            "LevelAuthorName": data["metadata"]["levelAuthorName"]
        }
    else:
        return None


def main(player_dat_file):
    # Open the output CSV file
    with open('output.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['QueriedHash', 'ID', 'SongName', 'LevelAuthorName'])

        # Read the hash values from hashes.txt
        player_data_all = []
        with open(player_dat_file, 'r') as f:
            player_data_all = json.load(f)

        hashes_file = player_data_all['localPlayers'][0]['favoritesLevelIds']
        hashes_file = [f.strip("custom_level_") for f in hashes_file]
        for line in hashes_file:
            hash_value = line.strip()
            result = query_beatsaver_api(hash_value)
            if result:
                # Write the extracted information to the CSV file
                writer.writerow([hash_value, result["ID"], result["SongName"], result["LevelAuthorName"]])
            else:
                # If no result, write the hash with empty fields for the other columns
                writer.writerow([hash_value, "", "", ""])


if __name__ == "__main__":
    player_data_file = r"C:\Users\frede\AppData\LocalLow\Hyperbolic Magnetism\Beat Saber\PlayerData.dat"
    main(player_data_file)
