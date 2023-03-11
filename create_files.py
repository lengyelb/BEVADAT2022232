import os
from pathlib import Path

week = input("What weeks files do you want to create?")

try:
    week = int(week)
except ValueError:
    print("This not a number")
    exit()

gyak_name = f"GYAK0{week}" if week < 10 else f"GYAK{week}"
hazi_name = f"HAZI0{week}" if week < 10 else f"HAZI{week}"

gyak_folder = Path(os.getcwd()).joinpath("GYAK", gyak_name)
hazi_folder = Path(os.getcwd()).joinpath("HAZI", hazi_name)

gyak_folder.mkdir(exist_ok=True)
hazi_folder.mkdir(exist_ok=True)

with open(gyak_folder.joinpath(gyak_name+".py"), 'w'):
    pass

with open(hazi_folder.joinpath(hazi_name+".py"), 'w'):
    pass
