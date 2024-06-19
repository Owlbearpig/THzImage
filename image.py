import itertools
import random
import re
import timeit
from functools import partial
from itertools import product
import matplotlib.pyplot as plt
import matplotlib as mpl
from consts import *
import numpy as np
import matplotlib.ticker as ticker
from functions import do_fft, do_ifft, phase_correction, unwrap, window, polyfit, f_axis_idx_map, to_db
from functions import remove_spikes, save_fig, export_array
from Measurements.measurements import get_all_measurements, MeasurementType
from tmm_slim import coh_tmm
from tmm import coh_tmm as coh_tmm_full
from mpl_settings import mpl_style_params, fmt
from scipy.optimize import shgo
from Evaluation.sub_eval_tmm_numerical import tmm_eval
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter


# shgo = partial(shgo, workers=1)


class Image:
    plotted_ref = False
    noise_floor = None
    time_axis = None
    cache_path = None
    sample_idx = None
    all_points = None
    options = {}
    name = ""

    def __init__(self, data_path, sub_image=None, sample_idx=None, options=None):
        self.data_path = data_path
        self.sub_image = sub_image

        self.refs, self.sams, self.other = self._set_measurements()
        if sample_idx is not None:
            self.sample_idx = sample_idx

        self.image_info = self._set_info()
        self._set_options(options)
        self.image_data = self._image_cache()
        self._evaluated_points = {}

    def _set_options(self, options_):
        if options_ is None:
            options_ = {}

        # set defaults if missing # TODO use default_dict ?
        if "excluded_areas" not in options_.keys():
            options_["excluded_areas"] = None
        if "one2onesub" not in options_.keys():
            options_["one2onesub"] = False

        if "cbar_min" not in options_.keys():
            options_["cbar_min"] = 0
        if "cbar_max" not in options_.keys():
            options_["cbar_max"] = np.inf

        if "log_scale" not in options_.keys():
            options_["log_scale"] = False

        if "color_map" not in options_.keys():
            options_["color_map"] = "autumn"

        if "invert_x" not in options_.keys():
            options_["invert_x"] = False
        if "invert_y" not in options_.keys():
            options_["invert_y"] = False

        if "load_mpl_style" not in options_.keys():
            options_["load_mpl_style"] = True
        else:
            options_["load_mpl_style"] = False

        self.options.update(options_)
        self._apply_options()

    def _apply_options(self):
        if self.options["load_mpl_style"]:
            mpl.rcParams = mpl_style_params()

    def _set_measurements(self):
        all_measurements = get_all_measurements(data_dir_=self.data_path)
        refs, sams, other = self._filter_measurements(all_measurements)

        refs = tuple(sorted(refs, key=lambda meas: meas.meas_time))
        sams = tuple(sorted(sams, key=lambda meas: meas.meas_time))

        first_measurement = min(refs[0], sams[0], key=lambda meas: meas.meas_time)
        last_measurement = max(refs[-1], sams[-1], key=lambda meas: meas.meas_time)
        print(f"First measurement at: {first_measurement.meas_time}, last measurement: {last_measurement.meas_time}")
        dt = last_measurement.meas_time - first_measurement.meas_time
        print(f"Total measurement time: {round(dt.total_seconds() / 3600, 2)} hours\n")

        return refs, sams, other

    @staticmethod
    def _filter_measurements(measurements):
        refs, sams, other = [], [], []
        for measurement in measurements:
            if measurement.meas_type.value == MeasurementType.REF.value:
                refs.append(measurement)
            elif measurement.meas_type.value == MeasurementType.SAM.value:
                sams.append(measurement)
            else:
                other.append(measurement)

        return refs, sams, other

    def _set_info(self):
        parts = self.sams[0].filepath.parts
        if self.sample_idx is None:
            self.sample_idx = sample_names.index(parts[-2])
        # self.name = f"Sample {self.sample_idx + 1} {parts[-3]}"
        self.name = f"Sample {self.sample_idx + 1}"

        sample_data_td = self.sams[0].get_data_td()
        samples = int(sample_data_td.shape[0])
        self.time_axis = sample_data_td[:, 0].real

        sample_data_fd = self.sams[0].get_data_fd()
        self.freq_axis = sample_data_fd[:, 0].real

        dt = np.mean(np.diff(self.time_axis))

        x_coords, y_coords = [], []
        for sam_measurement in self.sams:
            x_coords.append(sam_measurement.position[0])
            y_coords.append(sam_measurement.position[1])

        x_coords, y_coords = array(sorted(set(x_coords))), array(sorted(set(y_coords)))

        self.all_points = list(itertools.product(x_coords, y_coords))

        w, h = len(x_coords), len(y_coords)
        x_diff, y_diff = np.abs(np.diff(x_coords)), np.abs(np.diff(y_coords))
        dx = np.min(x_diff[np.nonzero(x_diff)])
        dy = np.min(y_diff[np.nonzero(y_diff)])

        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        self._empty_grid = np.zeros((w, h), dtype=complex)

        return {"w": w, "h": h, "dx": dx, "dy": dy, "dt": dt, "samples": samples, "extent": extent}

    def _image_cache(self):
        """
        read all measurements into array and save as npy at location of first measurement
        """
        self.cache_path = Path(self.sams[0].filepath.parent / "cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)

        try:
            img_data = np.load(str(self.cache_path / "_raw_img_cache.npy"))
        except FileNotFoundError:
            w, h, samples = self.image_info["w"], self.image_info["h"], self.image_info["samples"]
            dx, dy = self.image_info["dx"], self.image_info["dy"]
            img_data = np.zeros((w, h, samples))
            min_x, max_x, min_y, max_y = self.image_info["extent"]

            for sam_measurement in self.sams:
                x_pos, y_pos = sam_measurement.position
                x_idx, y_idx = int((x_pos - min_x) / dx), int((y_pos - min_y) / dy)
                img_data[x_idx, y_idx] = sam_measurement.get_data_td(get_raw=True)[:, 1]

            np.save(str(self.cache_path / "_raw_img_cache.npy"), img_data)

        return img_data

    def _coords_to_idx(self, x, y):
        x_idx = int((x - self.image_info["extent"][0]) / self.image_info["dx"])
        y_idx = int((y - self.image_info["extent"][2]) / self.image_info["dy"])

        return x_idx, y_idx

    def _idx_to_coords(self, x_idx, y_idx):
        dx, dy = self.image_info["dx"], self.image_info["dy"]
        y = self.image_info["extent"][2] + y_idx * dy
        x = self.image_info["extent"][0] + x_idx * dx

        return x, y

    def get_transmittance(self, x, y, **kwargs):
        measurement = self.get_measurement(x, y)

        sam_td = measurement.get_data_td()
        ref_td = self.get_ref(both=False, coords=(x, y))

        # sam_td = window(sam_td, win_len=12, shift=0, en_plot=False, slope=0.99)
        # ref_td = window(ref_td, win_len=12, shift=0, en_plot=False, slope=0.99)

        sam_td[:, 0] -= sam_td[0, 0]
        ref_td[:, 0] -= ref_td[0, 0]

        ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

        t_abs_meas = np.abs(sam_fd[:, 1] / ref_fd[:, 1])

        if "freq_range" in kwargs.keys():
            freq_idx = f_axis_idx_map(ref_fd[:, 0].real, kwargs["freq_range"])
        else:
            freq_idx = f_axis_idx_map(ref_fd[:, 0].real, None)

        return array([ref_fd[freq_idx, 0].real, t_abs_meas[freq_idx]], dtype=float).T

    def _calc_power_grid(self, freq_range):
        def power(measurement_):
            freq_slice = (freq_range[0] < self.freq_axis) * (self.freq_axis < freq_range[1])

            ref_td, ref_fd = self.get_ref(coords=measurement_.position, both=True)

            sam_fd = measurement_.get_data_fd()
            power_val_sam = np.sum(np.abs(sam_fd[freq_slice, 1])) / np.sum(freq_slice)
            power_val_ref = np.sum(np.abs(ref_fd[freq_slice, 1])) / np.sum(freq_slice)

            return (power_val_sam / power_val_ref) ** 2

        grid_vals = self._empty_grid.copy()

        for i, sam_measurement in enumerate(self.sams):
            print(f"{round(100 * i / len(self.sams), 2)} % done. "
                  f"(Measurement: {i}/{len(self.sams)}, {sam_measurement.position} mm)")
            x_idx, y_idx = self._coords_to_idx(*sam_measurement.position)
            val = power(sam_measurement)
            grid_vals[x_idx, y_idx] = val

        return grid_vals

    def _is_excluded(self, idx_tuple):
        if self.options["excluded_areas"] is None:
            return False

        if np.array(self.options["excluded_areas"]).ndim == 1:
            areas = [self.options["excluded_areas"]]
        else:
            areas = self.options["excluded_areas"]

        for area in areas:
            x, y = self._idx_to_coords(*idx_tuple)
            if (area[0] <= x <= area[1]) * (area[2] <= y <= area[3]):
                return True

        return False

    def _calc_grid_vals(self, quantity="p2p", selected_freq=1.200, ret_compl=False):
        info = self.image_info

        if self.options["one2onesub"]:
            grid_vals_cache_name = self.cache_path / f"{quantity}_{selected_freq}_s{self.sample_idx + 1}_121sub_layer1.npy"
        else:
            grid_vals_cache_name = self.cache_path / f"{quantity}_{selected_freq}_s{self.sample_idx + 1}_layer1.npy"

        # grid_vals_cache_name = self.cache_path / f"{quantity}_{selected_freq}_s{self.sample_idx + 1}_10_10.npy"

        if isinstance(selected_freq, tuple) and (quantity in ["MeanConductivity", "ConductivityRange"]):
            try:
                grid_vals = np.load(str(grid_vals_cache_name))
            except FileNotFoundError:
                freq_slice = (selected_freq[0] < self.freq_axis) * (self.freq_axis < selected_freq[1])
                freq_cnt = np.sum(freq_slice)

                grid_vals = np.zeros((info["w"], info["h"], freq_cnt), dtype=complex)
                for f_idx, freq in enumerate(self.freq_axis[freq_slice]):
                    for i, measurement in enumerate(self.sams):
                        pos = measurement.position
                        print(f"{round(100 * i / len(self.sams), 2)} % done, Frequency: {f_idx} / {freq_cnt}. "
                              f"(Measurement: {i}/{len(self.sams)}, {pos} mm)")
                        x_idx, y_idx = self._coords_to_idx(*pos)
                        res = self.tmm_film_fit(measurement, freq)

                        grid_vals[x_idx, y_idx, f_idx] = res["sigma"]

                np.save(str(grid_vals_cache_name), grid_vals)

            if quantity == "MeanConductivity":
                return np.mean(grid_vals.real, axis=2)
            elif quantity == "ConductivityRange":
                return grid_vals.real
        if quantity.lower() == "power":
            if isinstance(selected_freq, tuple):
                grid_vals = self._calc_power_grid(freq_range=selected_freq)
            else:
                print("Selected frequency must be range given as tuple")
                grid_vals = self._empty_grid
        elif quantity == "p2p":
            grid_vals = np.max(self.image_data, axis=2) - np.min(self.image_data, axis=2)
        elif quantity.lower() == "conductivity":
            try:
                grid_vals = np.load(str(grid_vals_cache_name))
            except FileNotFoundError:
                grid_vals = self._empty_grid.copy()

                for i, measurement in enumerate(self.sams):
                    pos = measurement.position
                    print(f"{round(100 * i / len(self.sams), 2)} % done. "
                          f"(Measurement: {i}/{len(self.sams)}, {pos} mm)")
                    res = self.tmm_film_fit(measurement, selected_freq)
                    x_idx, y_idx = self._coords_to_idx(*pos)
                    grid_vals[x_idx, y_idx] = res["sigma"]

                np.save(str(grid_vals_cache_name), grid_vals)
        elif quantity.lower() == "ref_amp":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                amp_, _ = self._ref_interpolation(measurement, selected_freq_=selected_freq,
                                                  ret_cart=False)
                grid_vals[x_idx, y_idx] = amp_
        elif quantity.lower() == "loss":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                res = self.tmm_film_fit(measurement, selected_freq)
                grid_vals[x_idx, y_idx] = np.log10(res["loss"])

        elif quantity == "Reference phase":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                _, phi_ = self._ref_interpolation(measurement, selected_freq_=selected_freq,
                                                  ret_cart=False)
                grid_vals[x_idx, y_idx] = phi_
        elif quantity == "ri":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                ri_ = tmm_eval(self, measurement.position, freq_range=selected_freq, disable_saving=True)
                grid_vals[x_idx, y_idx] = ri_
        else:
            # grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)
            grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)

        if ret_compl:
            return grid_vals

        return grid_vals.real

    def get_absorbance(self, point, en_plot=False):
        sam_fd, sam_fd = self.get_point(*point, both=True)
        ref_td, ref_fd = self.get_ref(both=True)

        freqs = ref_fd[:, 0].real

        absorb_ = 20 * np.log10(np.abs(ref_fd[:, 1] / sam_fd[:, 1]))

        if en_plot:
            plt.figure("Absorbance")
            plt.plot(freqs, absorb_, label=point)

        return array([freqs, absorb_]).T

    def plot_cond_vs_d(self, point=None, freq=None):
        if point is None:
            point = (33.0, 11.0)

        measurement = self.get_measurement(*point)
        if freq is None:
            freq = 1.200

        thicknesses = np.arange(0.000001, 0.001000, 0.000001)
        conductivities = np.zeros_like(thicknesses, dtype=complex)
        for idx, d in enumerate(thicknesses):
            res = self.tmm_film_fit(measurement, freq, d_film=d)
            conductivities[idx] = res["sigma"]

        plt.figure("Conductivity vs thickness")
        plt.title(f"Conductivity as a function of thickness at {freq} THz\n{self.name} {point} mm")
        plt.plot(thicknesses * 1e6, conductivities.real, label="Conductivity real part")
        plt.plot(thicknesses * 1e6, conductivities.imag, label="Conductivity imaginary part")
        plt.xlabel("Thickness (nm)")
        plt.ylabel("Conductivity (S/m)")
        plt.legend()
        plt.show()

    def _exclude_pixels(self, grid_vals):
        filtered_grid = grid_vals.copy()
        dims = filtered_grid.shape
        for x_idx in range(dims[0]):
            for y_idx in range(dims[1]):
                if self._is_excluded((x_idx, y_idx)):
                    filtered_grid[x_idx, y_idx] = 0

        return filtered_grid

    def _mirror_image(self, grid_vals):
        # should flip horizontal axis around center
        grid_vals_mirrored = grid_vals.copy()

        w = grid_vals.shape[0] - 1
        for x_idx in range(grid_vals.shape[0]):
            grid_vals_mirrored[w - x_idx, :] = grid_vals[x_idx, :]

        return grid_vals_mirrored

    def _smoothen_(self, grid_vals):
        return grid_vals
        # TODO
        grid_vals_smooth = grid_vals.copy()
        w, h = grid_vals.shape

        for x_idx in range(w):
            for y_idx in range(h):
                try:
                    if np.sum(grid_vals[x_idx - 1:x_idx + 1, y_idx - 1:y_idx + 1]) > 4 * 1e6:
                        grid_vals_smooth[x_idx:y_idx] = 1e6
                except IndexError as e:
                    print(e)
                    continue

        return grid_vals_smooth

    def publication_image(self, selected_freq_):
        info = self.image_info
        grid_vals = self._calc_grid_vals(quantity="Conductivity", selected_freq=selected_freq_)

        full_extent = info["extent"]

        # self.options["excluded_areas"] = [[-8, -7, -10, 5], [-10, 38, 3, 7], [37.5, 39, -14, 6]]
        grid_vals = self._exclude_pixels(grid_vals)

        if sample_idx == 3:
            img_extent = [-8, 39.5, -14, 4.5]
            v_min_, v_max_ = 1, 4
        elif sample_idx == 0:
            img_extent = [-10, 40, -13, 16]
            v_min_, v_max_ = 5, 150
        else:
            exit("1")

        dx, dy = info["dx"], info["dy"]
        w0, w1 = int((img_extent[0] - full_extent[0]) / dx), int((img_extent[1] - full_extent[0]) / dx)
        h0, h1 = int((img_extent[2] - full_extent[2]) / dy), int((img_extent[3] - full_extent[2]) / dy)

        grid_vals *= 1e-5  # S/m -> kS/cm

        grid_vals = grid_vals[w0:w1, h0:h1]

        if sample_idx == 0:
            grid_vals = gaussian_filter(grid_vals, 0.55)
        else:
            grid_vals = gaussian_filter(grid_vals, 0.55)

        grid_vals[grid_vals < v_min_] = 0
        grid_vals[grid_vals > v_max_] = 0

        fig_label_ = f"sample{sample_idx + 1}"
        fig = plt.figure(fig_label_)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2)

        axes_extent = [img_extent[0] - self.image_info["dx"] / 2, img_extent[1] + self.image_info["dx"] / 2,
                       img_extent[2] - self.image_info["dy"] / 2, img_extent[3] + self.image_info["dy"] / 2]
        print(np.mean(grid_vals[grid_vals != 0]), np.std(grid_vals[grid_vals != 0]))
        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=v_min_, vmax=v_max_,
                        origin="lower",
                        cmap=plt.get_cmap("hot"),
                        extent=axes_extent,
                        )
        ax.grid(visible=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.invert_xaxis()
        ax.invert_yaxis()

        # add scale bar to image (2 mm)
        if sample_idx == 3:
            pass
            ax.text(*(35, 0), s="2 mm", color="white", horizontalalignment="center", fontsize=22)
            ax.plot([34, 36], [0.5, 0.5], c="white", lw=4, zorder=1)
        elif sample_idx == 0:
            pass
            ax.text(*(-3, 14), s="2 mm", color="white", horizontalalignment="center", fontsize=22)
            ax.plot([-4, -2], [14.5, 14.5], c="white", lw=4, zorder=1)
        else:
            exit("1")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(img, cax=cax)
        cbar.set_label("Film conductivity (kS/cm)", rotation=270, labelpad=30)

    def plot_image(self, selected_freq=None, quantity="p2p", img_extent=None, flip_x=False):
        if quantity.lower() == "p2p":
            label = ""
        elif quantity.lower() == "conductivity":
            label = " (S/m)  at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity.lower() == "ref_amp":
            label = " Interpolated ref. amp. at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity == "Reference phase":
            label = " interpolated at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity.lower() == "power":
            label = f" ({selected_freq[0]}-{selected_freq[1]}) THz"
        elif quantity.lower() == "loss":
            label = " function value (log10)"
        elif quantity.lower() == "ri":
            label = " Refractive index"
        else:
            label = " (S/m)"

        info = self.image_info
        if img_extent is None:
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        if quantity.lower() == "ri":
            grid_vals = self._calc_grid_vals(quantity=quantity, selected_freq=selected_freq, ret_compl=True)
            print("Average: ", np.mean(grid_vals[grid_vals.real < 2.0]))
            print(np.std(grid_vals.real[grid_vals.real < 2.0]), np.std(grid_vals.imag[grid_vals.real < 2.0]))
            grid_vals = grid_vals.real
        else:
            grid_vals = self._calc_grid_vals(quantity=quantity, selected_freq=selected_freq)

        grid_vals = grid_vals[w0:w1, h0:h1]

        grid_vals = self._exclude_pixels(grid_vals)

        if self.options["log_scale"]:
            grid_vals = np.log10(grid_vals)

        if flip_x:
            grid_vals = self._mirror_image(grid_vals)

        grid_vals = self._smoothen_(grid_vals)

        fig = plt.figure(f"{self.name} {sample_labels[self.sample_idx]}")
        ax = fig.add_subplot(111)
        ax.set_title(f"{self.name} {sample_labels[self.sample_idx]}")
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.image_info["extent"]

        if self.options["log_scale"]:
            self.options["cbar_min"] = np.log10(self.options["cbar_min"])
            self.options["cbar_max"] = np.log10(self.options["cbar_max"])

        try:
            cbar_min = np.min(grid_vals[grid_vals > self.options["cbar_min"]])
            cbar_max = np.max(grid_vals[grid_vals < self.options["cbar_max"]])
        except ValueError:
            print("Check cbar bounds")
            cbar_min = np.min(grid_vals[grid_vals > 0])
            cbar_max = np.max(grid_vals[grid_vals < np.inf])

        # grid_vals[grid_vals < self.options["cbar_min"]] = 0
        # grid_vals[grid_vals > self.options["cbar_max"]] = 0 # [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        axes_extent = [img_extent[0] - self.image_info["dx"] / 2, img_extent[1] + self.image_info["dx"] / 2,
                       img_extent[2] - self.image_info["dy"] / 2, img_extent[3] + self.image_info["dy"] / 2]
        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=cbar_min, vmax=cbar_max,
                        origin="lower",
                        cmap=plt.get_cmap(self.options["color_map"]),
                        extent=axes_extent)
        if self.options["invert_x"]:
            ax.invert_xaxis()
        if self.options["invert_y"]:
            ax.invert_yaxis()

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        if np.max(grid_vals) > 1000:
            cbar = fig.colorbar(img, format=ticker.FuncFormatter(fmt))
        else:
            cbar = fig.colorbar(img)

        if quantity.lower() == "conductivity":
            label = " at " + str(np.round(selected_freq, 3)) + " THz"
            cbar.set_label("$\log_{10}$($\sigma$) " + label, rotation=270, labelpad=30)
            cbar.set_label("$\sigma$ (S/m) " + label, rotation=270, labelpad=30)
        else:
            cbar.set_label(f"{quantity}".title() + label, rotation=270, labelpad=30)

    def get_measurement(self, x, y, meas_type=MeasurementType.SAM.value):
        if meas_type == MeasurementType.REF.value:
            meas_list = self.refs
        elif meas_type == MeasurementType.SAM.value:
            meas_list = self.sams
        else:
            meas_list = self.other

        closest_meas, best_fit_val = None, np.inf
        for meas in meas_list:
            val = abs(meas.position[0] - x) + \
                  abs(meas.position[1] - y)
            if val < best_fit_val:
                best_fit_val = val
                closest_meas = meas

        return closest_meas

    def get_point(self, x, y, normalize=False, sub_offset=False, both=False, add_plot=False):
        dt = self.image_info["dt"]

        x_idx, y_idx = self._coords_to_idx(x, y)
        y_ = self.image_data[x_idx, y_idx]

        if sub_offset:
            y_ -= (np.mean(y_[:10]) + np.mean(y_[-10:])) * 0.5

        if normalize:
            y_ *= 1 / np.max(y_)

        t = np.arange(0, len(y_)) * dt
        y_td = np.array([t, y_]).T

        if add_plot:
            self.plot_point(x, y, y_td)

        if not both:
            return y_td
        else:
            return y_td, do_fft(y_td)

    def get_ref(self, both=False, normalize=False, sub_offset=False, coords=None, ret_meas=False):
        if coords is not None:
            closest_sam = self.get_measurement(*coords, meas_type=MeasurementType.SAM.value)

            closest_ref, best_fit_val = None, np.inf
            for ref_meas in self.refs:
                val = np.abs((closest_sam.meas_time - ref_meas.meas_time).total_seconds())
                if val < best_fit_val:
                    best_fit_val = val
                    closest_ref = ref_meas
            dt = (closest_sam.meas_time - closest_ref.meas_time).total_seconds()
            print(f"Time between ref and sample: {dt} seconds")
            chosen_ref = closest_ref
        else:
            chosen_ref = self.refs[-1]

        ref_td = chosen_ref.get_data_td()

        if sub_offset:
            ref_td[:, 1] -= (np.mean(ref_td[:10, 1]) + np.mean(ref_td[-10:, 1])) * 0.5

        if normalize:
            ref_td[:, 1] *= 1 / np.max(ref_td[:, 1])

        ref_td[:, 0] -= ref_td[0, 0]

        if ret_meas:
            return chosen_ref

        if both:
            ref_fd = do_fft(ref_td)
            return ref_td, ref_fd
        else:
            return ref_td

    def get_phase(self, point_):
        film_measurement = self.get_measurement(*point_)

        film_td = film_measurement.get_data_td()
        # film_td[:, 0] -= film_td[0, 0]

        t = film_td[:, 0]

        film_ref_td = self.get_ref(both=False, coords=point_)

        film_ref_fd, film_fd = do_fft(film_ref_td), do_fft(film_td)

        f = film_ref_fd[:, 0].real

        argmax_ref, argmax_sam = np.argmax(film_ref_td[:, 1]), np.argmax(film_td[:, 1])

        t0_ref, t0_sam = t[argmax_ref], t[argmax_sam]

        phi0_ref, phi0_sam = 2 * pi * t0_ref * f, 2 * pi * t0_sam * f
        phi0_offset = 2 * pi * (t0_sam - t0_ref) * f

        phi_ref = np.angle(film_ref_fd[:, 1] * np.exp(-1j * phi0_ref))
        phi_sam = np.angle(film_fd[:, 1] * np.exp(-1j * phi0_sam))

        phi0 = np.unwrap(phi_sam - phi_ref)

        phi = phi0 - phi0_ref + phi0_sam + phi0_offset

        fit_slice = (f >= 1.0) * (f <= 1.5)
        p = np.polyfit(f[fit_slice], phi[fit_slice], 1)
        phi = phi - p[1].real  # 2*pi*int(p[1].real / (2*pi))
        """
        plt.figure("aeee")
        plt.plot(phi, label=str(self.data_path))
        plt.legend()
        """
        phi = np.angle(np.exp(1j * phi))

        return phi

    def plot_point(self, x, y, sam_td=None, ref_td=None, sub_noise_floor=False, label="", td_scale=1):
        if (sam_td is None) and (ref_td is None):
            sam_td = self.get_point(x, y, sub_offset=True)
            ref_td = self.get_ref(sub_offset=True, coords=(x, y))

            # sam_td = window(sam_td, win_len=12, shift=0, en_plot=False, slope=0.99)
            # ref_td = window(ref_td, win_len=12, shift=0, en_plot=False, slope=0.99)

            ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

            # sam_td, sam_fd = phase_correction(sam_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)
            # ref_td, ref_fd = phase_correction(ref_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)

        else:
            ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

        phi_ref, phi_sam = unwrap(ref_fd), unwrap(sam_fd)

        noise_floor = np.mean(20 * np.log10(np.abs(ref_fd[ref_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        if not self.plotted_ref:
            plt.figure("Spectrum")
            plt.plot(ref_fd[plot_range1, 0], 20 * np.log10(np.abs(ref_fd[plot_range1, 1])) - noise_floor,
                     label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

            plt.figure("Phase")
            plt.plot(ref_fd[plot_range1, 0], phi_ref[plot_range1, 1], label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase (rad)")

            plt.figure("Time domain")
            plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (Arb. u.)")

            self.plotted_ref = True

        label += f" (x={x} (mm), y={y} (mm))"
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        plt.figure("Spectrum")
        plt.plot(sam_fd[plot_range1, 0], 20 * np.log10(np.abs(sam_fd[plot_range1, 1])) - noise_floor, label=label)

        plt.figure("Phase")
        plt.plot(sam_fd[plot_range1, 0], phi_sam[plot_range1, 1], label=label)

        plt.figure("Time domain")
        plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=label + f"\n(Amplitude x {td_scale})")

    def evaluate_point(self, x, y, **kwargs):
        key_ = f"{x} {y}"
        if key_ in self._evaluated_points.keys():
            res_ = self._evaluated_points[key_]
        else:
            measurement = self.get_measurement(x, y)
            res_ = self.tmm_film_fit(measurement, **kwargs)
            self._evaluated_points[key_] = res_

        return res_

    def plot_refractive_index(self, x, y, **kwargs):
        res = self.evaluate_point(x, y, **kwargs)

        n_ = res["n"]

        if not plt.fignum_exists("n_spectrum"):
            fig = plt.figure("n_spectrum")
            ax_ = fig.add_subplot(111)
        else:
            plt.figure("n_spectrum")
            ax_ = plt.gca()

        freqs = n_[:, 0].real

        ax_.set_ylabel("Refractive index")
        ax_.set_xlabel("Frequency (THz)")
        ax_.plot(freqs, n_[:, 1].real, label="Re(n)")
        ax_.plot(freqs, n_[:, 1].imag, label="Im(n)")

        return n_

    def plot_transmittance(self, x, y, **kwargs):
        res = self.evaluate_point(x, y, **kwargs)

        t_abs = res["t_abs"]
        t_abs_meas = res["t_abs_meas"]

        if not plt.fignum_exists("Amp. transmission"):
            fig = plt.figure("Amp. transmission")
            ax_ = fig.add_subplot(111)
        else:
            plt.figure("Amp. transmission")
            ax_ = plt.gca()

        freqs = t_abs[:, 0].real

        ax_.set_ylabel("Amplitude transmission")
        ax_.set_xlabel("Frequency (THz)")
        ax_.plot(freqs, t_abs[:, 1], label="TMM")
        ax_.plot(freqs, t_abs_meas[:, 1], label="Measured")

        return t_abs, t_abs_meas

    def plot_reflectance(self, x, y, **kwargs):
        res = self.evaluate_point(x, y, **kwargs)

        R = res["R"]

        if not plt.fignum_exists("Reflectance"):
            fig = plt.figure("Reflectance")
            ax_ = fig.add_subplot(111)
        else:
            plt.figure("Reflectance")
            ax_ = plt.gca()

        freqs = R[:, 0].real

        ax_.set_ylabel("%")
        ax_.set_xlabel("Frequency (THz)")
        ax_.plot(freqs, R[:, 1], label="Reflectance")

    def plot_conductivity_spectrum(self, x, y, **kwargs):
        res = self.evaluate_point(x, y, **kwargs)

        sigma = res["sigma"]

        if not plt.fignum_exists("cond_spectrum"):
            fig = plt.figure("cond_spectrum")
            ax_ = fig.add_subplot(111)
        else:
            plt.figure("cond_spectrum")
            ax_ = plt.gca()

        freqs = sigma[:, 0].real

        ax_.set_ylabel("Conductivity (S/m)")
        ax_.set_xlabel("Frequency (THz)")
        ax_.plot(freqs, sigma[:, 1].real, label="Re($\sigma$)")
        ax_.plot(freqs, sigma[:, 1].imag, label="Im($\sigma$)")

        return sigma

    def histogram(self):
        grid_vals = self._calc_grid_vals(quantity="Conductivity", selected_freq=1.200)

        plt.figure("Histogram")
        plt.hist(grid_vals, density=False)  # density=False would make counts
        plt.ylabel("Conductivity (S/m)")
        plt.xlabel("Count")

    def system_stability(self, selected_freq_=0.800):
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))

        ref_ampl_arr, ref_angle_arr = [], []

        t0 = self.refs[0].meas_time
        meas_times = [(ref.meas_time - t0).total_seconds() / 3600 for ref in self.refs]
        for i, ref in enumerate(self.refs):
            ref_td = ref.get_data_td()
            # ref_td = window(ref_td, win_len=12, shift=0, en_plot=False, slope=0.05)
            ref_fd = do_fft(ref_td)
            # ref_fd = phase_correction(ref_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)

            ref_ampl_arr.append(np.sum(np.abs(ref_fd[f_idx, 1])) / 1)
            phi = np.angle(ref_fd[f_idx, 1])
            """ ???
            if i and (abs(ref_angle_arr[-1] - phi) > pi):
                phi -= 2 * pi
            """
            ref_angle_arr.append(phi)
        ref_angle_arr = np.unwrap(ref_angle_arr)
        ref_angle_arr -= np.mean(ref_angle_arr)
        ref_ampl_arr -= np.mean(ref_ampl_arr)

        random.seed(10)
        rnd_sam = random.choice(self.sams)
        position1 = (19, 4)
        position2 = (20, 4)
        sam1 = self.get_measurement(*position1)  # rnd_sam
        sam2 = self.get_measurement(*position2)  # rnd_sam

        sam_t1 = (sam1.meas_time - t0).total_seconds() / 3600
        amp_interpol1, phi_interpol1 = self._ref_interpolation(sam1, ret_cart=False, selected_freq_=selected_freq_)

        sam_t2 = (sam2.meas_time - t0).total_seconds() / 3600
        amp_interpol2, phi_interpol2 = self._ref_interpolation(sam2, ret_cart=False, selected_freq_=selected_freq_)

        plt.figure("System stability amplitude")
        plt.title(f"Reference amplitude at {selected_freq_} THz")
        plt.plot(meas_times, ref_ampl_arr, label=t0)
        # plt.plot(sam_t1, amp_interpol1, marker="o", markersize=5, label=f"Interpol (x={position1[0]}, y={position1[1]}) mm")
        # plt.plot(sam_t2, amp_interpol2, marker="o", markersize=5, label=f"Interpol (x={position2[0]}, y={position2[1]}) mm")
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Amplitude (Arb. u.)")

        plt.figure("System stability angle")
        plt.title(f"Reference phase at {selected_freq_} THz")
        plt.plot(meas_times, ref_angle_arr, label=t0)
        # plt.plot(sam_t1, phi_interpol1, marker="o", markersize=5, label=f"Interpol (x={position1[0]}, y={position1[1]}) mm")
        # plt.plot(sam_t2, phi_interpol2, marker="o", markersize=5, label=f"Interpol (x={position2[0]}, y={position2[1]}) mm")
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Phase (rad)")

    def _ref_interpolation(self, sam_meas, selected_freq_=0.800, ret_cart=False):
        sam_meas_time = sam_meas.meas_time

        nearest_ref_idx, smallest_time_diff, time_diff = None, np.inf, 0
        for ref_idx in range(len(self.refs)):
            time_diff = (self.refs[ref_idx].meas_time - sam_meas_time).total_seconds()
            if abs(time_diff) < abs(smallest_time_diff):
                nearest_ref_idx = ref_idx
                smallest_time_diff = time_diff

        t0 = self.refs[0].meas_time
        if smallest_time_diff <= 0:
            # sample was measured after reference
            ref_before = self.refs[nearest_ref_idx]
            ref_after = self.refs[nearest_ref_idx + 1]
        else:
            ref_before = self.refs[nearest_ref_idx - 1]
            ref_after = self.refs[nearest_ref_idx]

        t = [(ref_before.meas_time - t0).total_seconds(), (ref_after.meas_time - t0).total_seconds()]
        ref_before_td, ref_after_td = ref_before.get_data_td(), ref_after.get_data_td()

        # ref_before_td = window(ref_before_td, win_len=12, shift=0, en_plot=False, slope=0.05)
        # ref_after_td = window(ref_after_td, win_len=12, shift=0, en_plot=False, slope=0.05)

        ref_before_fd, ref_after_fd = do_fft(ref_before_td), do_fft(ref_after_td)

        # ref_before_fd = phase_correction(ref_before_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)
        # ref_after_fd = phase_correction(ref_after_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)

        # if isinstance(selected_freq_, tuple):

        # else:
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))
        y_amp = [np.sum(np.abs(ref_before_fd[f_idx, 1])) / 1,
                 np.sum(np.abs(ref_after_fd[f_idx, 1])) / 1]
        y_phi = [np.angle(ref_before_fd[f_idx, 1]), np.angle(ref_after_fd[f_idx, 1])]

        amp_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_amp)
        phi_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_phi)

        if ret_cart:
            return amp_interpol * np.exp(1j * phi_interpol)
        else:
            return amp_interpol, phi_interpol

    def _4pp_measurement(self, s_idx=0, row_id=None):
        if row_id is None:
            s1 = array([6.49, 6.40, 6.40, 6.64, 6.38, 6.29, 6.38, 6.47, 6.29, 6.22]) * 1e5
            s4 = array([1.09, 1.46, 1.26, 0.81, 1.28, 1.30]) * 1e4
            map_ = {"0": s1, "3": s4}

            return map_[str(s_idx)]

        s1_r1 = array([2.76E+06, 2.76E+06, 3.15E+06, 3.68E+06, 4.20E+06, 4.15E+06,
                       3.92E+06, 4.31E+06, 4.31E+06, 4.41E+06, 4.53E+06, 4.31E+06])
        s1_r2 = array([3.46E+06, 3.46E+06, 3.15E+06, 4.90E+06, 4.41E+06, 4.20E+06,
                       4.41E+06, 4.53E+06, 4.90E+06, 4.90E+06, 4.84E+06, 5.19E+06])
        s1_r3 = array([4.44E+06, 4.20E+06, 3.97E+06, 4.77E+06, 4.77E+06, 3.80E+06,
                       4.64E+06, 4.64E+06, 4.90E+06, 4.77E+06, 4.77E+06, 4.90E+06])
        s1_r4 = array([4.44E+06, 4.77E+06, 4.64E+06, 4.58E+06, 4.41E+06, 4.90E+06,
                       4.90E+06, 4.90E+06, 4.90E+06, 4.90E+06, 4.77E+06, 4.90E+06])

        s2_r2 = array([1.13, 3.09, 7.35, 4.56, 22.6, 2.31, 7.82, 10.9, 13.3, 10.4, 9.13,
                       9.57, 8.54, 10.1, 4.69, 0.682, 12.9, 10.3, 10.7, 10.7, 9.06, 0.937, 0.421]) * 1e3
        s4_r3_old = array([2.08, 5.64, 5.44, 3.43, 6.51, 16.6, 10.8, 11.5, 12.7, 9.5, 6.29]) * 1e4
        s4_r3 = array([1.22E+03, 6.21E+03, 6.32E+03, 8.08E+03, 1.04E+04, 5.53E+03,
                       1.52E+04, 1.88E+04, 1.72E+04, 1.54E+04, 1.11E+04, 4.47E+03, ])
        s4_r4 = array([1.07E+03, 3.87E+03, 5.43E+03, 7.96E+03, 9.16E+03, 9.21E+03,
                       1.45E+04, 1.40E+04, 1.36E+04, 1.09E+04, 8.79E+03, 4.36E+03, ])
        s4_r5 = array([3.67E+03, 6.79E+03, 6.39E+03, 6.52E+03, 6.10E+03, 9.96E+03,
                       8.88E+03, 8.72E+03, 8.61E+03, 8.40E+03, 7.09E+03, 4.61E+03, ])
        if not plt.fignum_exists("abe"):
            plt.figure("abe")
            plt.semilogy(s4_r3_old, label="s4_r3_old")
            plt.semilogy(s4_r3, label="s4_r3")
            plt.semilogy(s4_r4, label="s4_r4")
            plt.semilogy(s4_r5, label="s4_r5")
            plt.legend()

        map_ = {"s1_r1": s1_r1, "s1_r2": s1_r2, "s1_r3": s1_r3, "s1_r4": s1_r4,
                "s2_r2": s2_r2,
                "s4_r3": s4_r3, "s4_r4": s4_r4, "s4_r5": s4_r5,
                "s4_r3_old": s4_r3_old,
                }

        # slice_ = slice(0, 12)
        # vals = map_[f"s{s_idx + 1}_r{row_id}"][slice_]
        vals = map_[f"s{s_idx + 1}_r{row_id}"]

        if s_idx == 3:
            return vals[1:]
        else:
            return vals

    def _average_area(self, line_segment):
        p0, p1 = line_segment

        # convert to idx. Direction of the line doesn't matter
        p1_idx, p0_idx = self._coords_to_idx(*p1), self._coords_to_idx(*p0)
        min_x, max_x = min(p1_idx[0], p0_idx[0]), max(p1_idx[0], p0_idx[0])
        min_y, max_y = min(p1_idx[1], p0_idx[1]), max(p1_idx[1], p0_idx[1])

        x_range = range(min_x, max_x + 1)
        y_range = range(min_y, max_y + 1)
        area_indices = list(product(x_range, y_range))

        grid_vals = self._calc_grid_vals(quantity="Conductivity")
        conductivities = array([grid_vals[p] for p in area_indices], dtype=float)
        print(f"segment: {line_segment}, conductivities: {conductivities}")
        avg_val = np.sum(conductivities) / len(conductivities)
        print(f"mean: {avg_val}", "\n")

        return avg_val

    def _p0_map(self, row_idx_):
        """
        p0x_s1:
        measured based on photo and p2p image / laser pointer
        (39: border at substrate / coated substrate, 3.1: distance border to start of row 1) (mm)
        """
        p0x_s1 = 37 - 3  # p0x_s1 = 39 - 3.1

        p0_map_ = {"s1": (33, 15), "s4_13": (2, 13), "s4_46": (30, 12), "s2_r2": (44, 15.5),
                   # "s1_r4": (p0x_s1, -1.5), "s1_r3": (p0x_s1, 3), "s1_r2": (p0x_s1, 8.5), "s1_r1": (p0x_s1, 13.5),
                   "s1_r4": (p0x_s1, -2.5), "s1_r3": (p0x_s1, 3.5), "s1_r2": (p0x_s1, 8.5), "s1_r1": (p0x_s1, 13.5),
                   "s4_r3_old": (36.5, -3.50),  # "s4_r3_old": (37, 6), original

                   # "s4_r3": (35, -2.00), "s4_r4": (35, -5.00), "s4_r5": (35, 1.50), # measured
                   # "s4_r3": (37, -3.50),  "s4_r5": (37, 0.00), "s4_r4": (37.5, -5.00), # original r5, r4 "works"
                   # "s4_r3": (37.5, -2.50), "s4_r4": (37.5, -5.00), "s4_r5": (37.5, -0.50),  # best if not considering first point
                   "s4_r3": (37, -2.50), "s4_r4": (37, -5.00), "s4_r5": (37, -0.50),
                   }
        p0 = p0_map_[f"s{self.sample_idx + 1}_r{row_idx_}"]

        return p0

    def thz_vs_4pp(self, row_idx, segment_width=0, p0=None, en_plot=True, corr_measure="r2"):
        def scale(val, dst=None):
            """
            Scale the given value to the scale of dst.
            """
            val = array(val)
            return val
            # return np.log10(val)

            if dst is None:
                dst = (0, 1)
            src = np.min(val), np.max(val)
            # return val / np.max(val)
            return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]

        def line_segments(segment_cnt_, p0_, line_len_=None, arrow_height=None):
            if line_len_ is None:
                s = 3.5  # mm
                # s = 3.00  # mm
                # line_len = 3 * s
                line_len_ = 1 * s

            if arrow_height is None:
                arrow_height = 0  # self.image_info["dy"] assume single pixel, mm
                # arrow_height = 2

            segments_ = []
            for segment_idx in range(segment_cnt_):
                dy_l = arrow_height * (segment_idx / segment_cnt_)
                dy_r = arrow_height * ((segment_idx + 1) / segment_cnt_)

                pl = p0_[0] - segment_idx * line_len_, p0_[1] + dy_l  # left point of diagonal
                pr = p0_[0] - (segment_idx + 1) * line_len_, p0_[1] + dy_r  # right point
                print(f"segment: {pl}, {pr}")
                segments_.append((pl, pr))

            return segments_

        _4pp_vals = self._4pp_measurement(s_idx=self.sample_idx, row_id=row_idx)
        _4pp_val_scaled = scale(_4pp_vals)

        segment_cnt = len(_4pp_vals)
        positions = range(1, segment_cnt + 1)

        line_len = 3.5  # 3.5  # 4.0 # s2

        if p0 is None:
            p0 = self._p0_map(row_idx)

        segments, thz_cond_vals = {}, {}
        for dy in range(1):
            segments[str(dy)] = line_segments(segment_cnt, p0, line_len, dy)

            thz_cond = []
            for segment in segments[str(dy)]:
                avg_val = self._average_area(segment)
                thz_cond.append(avg_val)

            thz_cond = np.array(thz_cond)
            thz_cond_vals[str(dy)] = scale(thz_cond)

        if "flipped" in str(self.data_path):
            # positions = np.arange(22, 13, -1)
            positions = np.arange(14, 23, 1)

        res = polyfit(_4pp_vals, thz_cond_vals[str(segment_width)], 1, remove_worst_outlier=False)
        x = np.linspace(np.min([_4pp_vals]), np.max([_4pp_vals]), 1000)
        z = res["polynomial"]
        fit = z[0] * x + z[1]
        r2 = "$R^2=$" + str(round(res["determination"], 3))

        if en_plot:
            if not plt.fignum_exists("THz vs 4pp"):
                fig = plt.figure("THz vs 4pp")
                _4pp_vs_thz_ax = fig.add_subplot(111)
            else:
                plt.figure("THz vs 4pp")
                _4pp_vs_thz_ax = plt.gca()

            _4pp_vs_thz_ax.set_title("$\sigma_{THz}$(1.2 THz) vs $\sigma_{4pp}$(DC)")
            _4pp_vs_thz_ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
            _4pp_vs_thz_ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
            _4pp_vs_thz_ax.scatter(_4pp_vals, thz_cond_vals[str(segment_width)],
                                   **scatter_kwargs[f"row{row_idx}"])
            _4pp_vs_thz_ax.set_xlabel("Conductivity 4pp measurement (S/m)")
            _4pp_vs_thz_ax.set_ylabel("Conductivity THz measurement (S/m)")

            sign = (z[1] > 0) * "+"
            plot_kwargs[f"row{row_idx}"]["label"] += r2
            _4pp_vs_thz_ax.plot(x, fit, **plot_kwargs[f"row{row_idx}"])

            if not plt.fignum_exists("4pp"):
                fig = plt.figure("4pp")
                _ax = fig.add_subplot(111)
                # ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
                tick_spacing = 1
                _ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                _ax.set_ylabel("Conductivity (normalized)")
                _ax.set_xlabel("Position idx")
            else:
                plt.figure("4pp")
                _ax = plt.gca()

            _ax.plot(positions, _4pp_val_scaled, label="$\sigma_{4pp}$(DC)")
            # plt.plot(thz_slope * _4pp_slope)

            for dy in thz_cond_vals.keys():
                if dy == str(segment_width):
                    plt.figure("4pp")
                    _ax.plot(positions, thz_cond_vals[str(dy)], label="$\sigma_{THz}$(1.2 THz)" + f" dy={dy} mm")

            def _plot_line_segment(segment_, segment_idx_=None):
                thz_img_num = f"{self.name} {sample_labels[self.sample_idx]}"
                thz_figs = [thz_img_num, f"sample{self.sample_idx + 1}"]
                for fig_num in thz_figs:
                    if plt.fignum_exists(fig_num):
                        plt.figure(fig_num)
                        break

                img_ax = plt.gca()
                x1 = (segment_[0][0], segment_[1][0])
                y1 = (segment_[0][1], segment_[1][1])
                if segment_idx_ == 0:
                    img_ax.vlines(x=x1[0] + 0.1, ymin=y1[0] - 1, ymax=y1[0] + 1, colors="white", lw=3, zorder=1)
                if segment_idx_ == 11:
                    img_ax.vlines(x=x1[1], ymin=y1[1] - 1, ymax=y1[1] + 1, colors="white", lw=3, zorder=1)
                img_ax.plot(x1, y1, color="white", zorder=1)
                text_pos = ((x1[1] + x1[0]) / 2, y1[0] - 1)
                if segment_idx_ is not None:
                    # img_ax.text(*text_pos, s=str(segment_idx_ + 1), color="white", horizontalalignment="center", size=40)
                    if segment_idx_ == 11:
                        label_pos = (x1[0] - 4, y1[0])
                        img_ax.text(*label_pos, color="white", s=f"{row_idx - 2}",
                                    horizontalalignment="left", verticalalignment="center", size=42)

            for segment_idx, segment in enumerate(segments[str(segment_width)]):
                _plot_line_segment(segment, segment_idx)

        cols = [x, fit, _4pp_vals, thz_cond_vals[str(segment_width)]]
        cols = [arr * 1e-5 for arr in cols]
        export_array(*cols, file_name=f"cond_thz_vs_4pp_row{row_idx}_s{self.sample_idx + 1}")

        if corr_measure == "r2":
            return res["determination"] * np.sign(res["polynomial"][0])
        else:
            thz_slope = np.sign(np.diff(thz_cond_vals[str(segment_width)]))
            _4pp_slope = np.sign(np.diff(_4pp_val_scaled))

            return np.sum(thz_slope * _4pp_slope < 0)

    def correlation_image(self, row_idx, center=None, segment_width=0):
        fig = plt.figure("correlation grid")
        ax = fig.add_subplot(111)
        ax.set_title("R2 for different row positions")

        if center is None:
            center = self._p0_map(row_idx)

        w, h, ds = 7, 7, 0.5
        x_range, y_range = [int(center[0] - w), int(center[0] + w)], [int(center[1] - h), int(center[1] + h)]
        x_range = np.clip(x_range, *self.image_info["extent"][0:2])
        y_range = np.clip(y_range, *self.image_info["extent"][2:4])

        x, y = np.arange(*x_range, ds), np.arange(*y_range, ds)
        corr_grid = np.zeros((len(x), len(y)))
        for i, p0_x in enumerate(x):
            for j, p0_y in enumerate(y):
                corr = film_image.thz_vs_4pp(row_idx=row_idx, p0=(p0_x, p0_y), en_plot=False,
                                             corr_measure="r2", segment_width=segment_width)
                corr_grid[i, j] = corr
                print(f"=> {p0_x}, {p0_y}, R2={corr}")
        img = ax.imshow(corr_grid.transpose((1, 0)), extent=[*x_range, *y_range], origin="lower", vmin=-0.01, vmax=1.01)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        cbar = fig.colorbar(img)
        cbar.set_label(f"R2", rotation=270, labelpad=30)

        if self.options["invert_x"]:
            ax.invert_xaxis()
        if self.options["invert_y"]:
            ax.invert_yaxis()


if __name__ == '__main__':
    sample_idx = 0

    meas_dir_sub = data_dir / "Uncoated" / f"s{sample_idx + 1}"
    sub_image = Image(data_path=meas_dir_sub)
    # sub_image.plot_image(selected_freq=1, quantity="ri")
    # meas_dir = data_dir / "s1_new_area_20_07_2023" / "Image0"

    # meas_dir = data_dir / "s1_new_area" / "Image0"
    # meas_dir = data_dir / "s1_new_area" / "Image1_25_07_2023_"  # win update crash
    # meas_dir = data_dir / "s1_new_area" / "Image1_26_07_2023"  # same as Image0, but sample was "repositioned"
    # meas_dir = data_dir / "s1_new_area" / "Image2_27_07_2023"  # 20avg
    # meas_dir = data_dir / "s1_new_area" / "Image3_28_07_2023"  # 0.5 mm
    # meas_dir = data_dir / "s3_new_area" / "Image0"

    meas_dir = data_dir / "s4_new_area" / "Image0"
    # meas_dir = data_dir / "Edge_4pp2" / "s4"  # old image
    # meas_dir = data_dir / "Edge_4pp2_s2_redo" / "s2"  # s2
    # meas_dir = data_dir / "s1_new_area" / "Image3_28_07_2023"  # s1
    # meas_dir = data_dir / "Edge" / "s4"
    meas_dir = data_dir / "s1_new_area" / "Image3_28_07_2023"  # s1

    # options = {"excluded_areas": [[3, 13, -10, 30], [33, 35, -10, 30]], "cbar_min": 1.0e6, "cbar_max": 6.5e6}
    options = {"excluded_areas": [[-10, 55, 12, 30],
                                  [-10, -6, -10, 30],
                                  [37, 55, -10, 30], ]}
    options = {"excluded_areas": [[33, 60, -4, 30],  # s1, s2, s3
                                  [3, 11, -4, 30],
                                  # [36, 55, -4, 30],
                                  ]}
    """
    options = {"excluded_areas": [  # s4
        [-13, 60, 13, 30],
        [-10, -5, -5, 30],
        [37, 60, -5, 30]
    ]}
    """  # 1.60*1e4, 1.74*1e4
    """
    options = {"cbar_min": 1.47e5, "cbar_max": 2.9e5, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": True}  # s4 (idx 3)
    """
    options = {"cbar_min": 1.5e4, "cbar_max": 1.85e4, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": False}  # s2 (idx 1) unflipped
    options = {"cbar_min": 0, "cbar_max": np.inf, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": False}  # s2 (idx 1) unflipped

    options = {"cbar_min": 1.4e4, "cbar_max": 1.70e4, "log_scale": False, "color_map": "viridis",
               "invert_x": False, "invert_y": False}  # s2 (idx 1) flipped
    options = {"cbar_min": 1e5, "cbar_max": 3.5e5, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": True}  # s4
    options = {"cbar_min": 1.5e5, "cbar_max": 4.5e5, "log_scale": False, "color_map": "hot",
               "invert_x": True, "invert_y": True}  # s4 new phase correction
    # """
    options = {"cbar_min": 5e5, "cbar_max": 1.5e7, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": True}  # s1 new phase correction
    options = {"cbar_min": 0, "cbar_max": 0.6, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": True}  # s1 p2p
    options = {"cbar_min": 1, "cbar_max": 3, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": True}  # s4 p2p
    # """

    film_image = Image(meas_dir, sub_image, sample_idx, options)
    # s1, s2, s3 = [-10, 50, -3, 27]
    # film_image.plot_cond_vs_d()
    # film_image.plot_image(img_extent=[-10, 50, -3, 27], quantity="loss", selected_freq=1.200)
    # sub_image.plot_image(img_extent=[-10, 50, -3, 27], quantity="p2p")
    # film_image.plot_image(quantity="p2p")
    # film_image.plot_image(quantity="power", selected_freq=(1.200, 1.300))
    # film_image.histogram()
    # film_image.plot_point(10.5, -10.5)
    # film_image.plot_conductivity_spectrum(10, -5, en_all_plots=True)
    # film_image.plot_refractive_index(10, -5, en_all_plots=False)
    # film_image.plot_transmittance(10, -5)
    # film_image.plot_reflectance(10, -5)
    # film_image.plot_image(quantity="Conductivity", selected_freq=1.200)
    film_image.publication_image(selected_freq_=1.200)  ##
    # film_image.publication_image(selected_freq_=0.800)

    # s1 r3 and 4 are off due to sensitivity limit
    # film_image.thz_vs_4pp(row_idx=1, segment_width=0)
    # film_image.thz_vs_4pp(row_idx=2, segment_width=0)
    # film_image.thz_vs_4pp(row_idx=3, segment_width=0)
    #film_image.thz_vs_4pp(row_idx=3, segment_width=0)
    #film_image.thz_vs_4pp(row_idx=4, segment_width=0)

    # film_image.thz_vs_4pp(row_idx=4, segment_width=0)
    # film_image.thz_vs_4pp(row_idx=3, segment_width=0)
    # film_image.thz_vs_4pp(row_idx=5, segment_width=0)
    # film_image.correlation_image(row_idx=4, segment_width=0)

    # film_image.correlation_image(row_idx=4, segment_width=0)
    # film_image.correlation_image(row_idx=3, p0=(37, 6))  # s4
    # film_image.plot_image(img_extent=[-10, 50, -3, 27], quantity="Reference phase", selected_freq=1.200)

    # sub_image.system_stability(selected_freq_=1.200)
    # film_image.system_stability(selected_freq_=1.200)

    # s4 = [18, 51, 0, 20]
    # film_image.plot_image(img_extent=[18, 51, 0, 20], quantity="p2p", selected_freq=1.200)
    # film_image.plot_image(img_extent=[18, 51, 0, 20], quantity="Conductivity", selected_freq=0.600)
    # film_image.plot_image(img_extent=[-5, 36, 0, 12], quantity="Conductivity", selected_freq=1.200)
    # film_image.plot_image(img_extent=[18, 51, 0, 20], quantity="power", selected_freq=(1.150, 1.250))

    # stability_dir = data_dir / "Stability" / "2023-03-20"

    # stability_image = Image(stability_dir)
    # stability_image.system_stability(selected_freq_=1.200)

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        save_fig(fig_label)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            plt.legend()

    plt.show()
