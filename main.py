import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import json
import os
from taipan_measurement import Measurement, MeasType
from pathlib import Path
from image import Image, ShownQuantity

# Settings
reset_prev_data_dir = False
pp_config = {"sub_offset": False,
             "en_windowing": False,
             "normalize": False}
thickness = 1000  # in um
selected_frequency = 0.850  # in THz
image_options = {"quantity": ShownQuantity.P2p}


def main():
    # main
    root = tk.Tk()
    root.withdraw()

    config_file = "config.json"

    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
    else:
        config = {}

    if reset_prev_data_dir or "prev_data_dir" not in config:
        data_dir = Path(filedialog.askdirectory())
        config["prev_data_dir"] = str(data_dir)
        with open(config_file, 'w') as file:
            json.dump(config, file)
    else:
        data_dir = Path(config["prev_data_dir"])

    print(f"Using data from {data_dir}")
    print("Change the path in config.json to use a different location\n")

    all_measurements = []
    for file in data_dir.glob("**/*.txt"):
        try:
            all_measurements.append(Measurement(file, pp_config))
        except ValueError:
            continue

    refs = [meas for meas in all_measurements if meas.meas_type == MeasType.Ref]
    sams = [meas for meas in all_measurements if meas.meas_type == MeasType.Sam]

    print(f"{len(all_measurements)} taipan measurement files found.", end=" ")
    print(f"{len(refs)} reference measurements, {len(sams)} sample measurements")

    image = Image(all_measurements, thickness, selected_frequency, image_options)
    image.plot_image()

    plt.show()


if __name__ == '__main__':
    main()
