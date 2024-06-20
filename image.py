import enum
import itertools
import matplotlib.pyplot as plt
import numpy as np
from functions import do_fft, save_fig, window, to_db, phase_correction
from pathlib import Path
from taipan_measurement import MeasType
from scipy.constants import c as c0

c_thz = c0 / 1e6


class ShownQuantity(enum.Enum):
    P2p = 0
    AbsorptionCoefficient = 1


class Image:
    time_axis = None
    cache_path = None
    all_points = None
    shown_quantity = None
    options = {}

    def __init__(self, measurements, sample_thickness, selected_frequency, options=None):
        self.measurements = measurements

        self.refs, self.sams, self.other = self._set_measurements()

        self.image_info = self._set_info()
        self._set_options(options)
        self.image_data_td, self.image_data_fd = self._image_cache()
        self._evaluated_points = {}
        self.sample_thickness = sample_thickness
        self.selected_frequency = selected_frequency

    def _set_options(self, options_):
        if options_ is None:
            options_ = {}

        # set defaults if missing # TODO use default_dict ?
        if "excluded_areas" not in options_.keys():
            options_["excluded_areas"] = None

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

        if "quantity" not in options_.keys():
            self.shown_quantity = ShownQuantity.AbsorptionCoefficient
        else:
            self.shown_quantity = options_["quantity"]

        self.options.update(options_)


    def _set_measurements(self):
        refs, sams, other = self._filter_measurements(self.measurements)

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
            if measurement.meas_type.value == MeasType.Ref.value:
                refs.append(measurement)
            elif measurement.meas_type.value == MeasType.Sam.value:
                sams.append(measurement)
            else:
                other.append(measurement)

        return refs, sams, other

    def _set_info(self):
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

        x_coords, y_coords = np.array(sorted(set(x_coords))), np.array(sorted(set(y_coords)))

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
            img_data_td = np.load(str(self.cache_path / "_raw_img_td_cache.npy"))
            img_data_fd = np.load(str(self.cache_path / "_raw_img_fd_cache.npy"))
        except FileNotFoundError:
            w, h, samples = self.image_info["w"], self.image_info["h"], self.image_info["samples"]
            dx, dy = self.image_info["dx"], self.image_info["dy"]
            img_data_td = np.zeros((w, h, samples))
            fd_samples = (samples + 1) // 2 if (samples % 2) else 1 + samples // 2
            img_data_fd = np.zeros((w, h, fd_samples))
            min_x, max_x, min_y, max_y = self.image_info["extent"]

            for i, sam_measurement in enumerate(self.sams):
                if i % 100 == 0:
                    print(f"{round(100 * i / len(self.sams), 2)} % processed")

                x_pos, y_pos = sam_measurement.position
                x_idx, y_idx = int((x_pos - min_x) / dx), int((y_pos - min_y) / dy)
                img_data_td[x_idx, y_idx] = sam_measurement.get_data_td(get_raw=True)[:, 1]
                img_data_fd[x_idx, y_idx] = sam_measurement.get_data_fd()[:, 1]

            np.save(str(self.cache_path / "_raw_img_td_cache.npy"), img_data_td)
            np.save(str(self.cache_path / "_raw_img_fd_cache.npy"), img_data_fd)

        return img_data_td, img_data_fd

    def _coords_to_idx(self, x, y):
        x_idx = int((x - self.image_info["extent"][0]) / self.image_info["dx"])
        y_idx = int((y - self.image_info["extent"][2]) / self.image_info["dy"])

        return x_idx, y_idx

    def _idx_to_coords(self, x_idx, y_idx):
        dx, dy = self.image_info["dx"], self.image_info["dy"]
        y = self.image_info["extent"][2] + y_idx * dy
        x = self.image_info["extent"][0] + x_idx * dx

        return x, y

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

    def _calculate_absorption(self, measurement):

        pos = measurement.position

        d = self.sample_thickness
        f_slice = self.freq_axis > 0
        freqs = self.freq_axis[f_slice]
        omega = 2 * np.pi * freqs
        ref_td, ref_fd = self.get_ref(coords=measurement.position, both=True)

        sam_td = self.image_data_td[*self._coords_to_idx(*pos)]
        sam_td = np.array([self.time_axis, sam_td]).T

        sam_td[:, 1] -= np.mean(sam_td[:10, 1])

        sam_td = window(sam_td)
        sam_fd = do_fft(sam_td)

        phi_sam = phase_correction(sam_fd)[f_slice, 1]
        phi_ref = phase_correction(ref_fd)[f_slice, 1]

        n = 1 + c_thz * (phi_sam - phi_ref) / (omega * d)

        # um -> cm
        alpha = -(1e4 * 2 / d) * np.log(np.abs(sam_fd[f_slice, 1] / ref_fd[f_slice, 1]) * (1 + n) ** 2 / (4 * n))
        f_idx = np.argmin(np.abs(self.selected_frequency - freqs))

        return alpha[f_idx]

    def _calc_grid_vals(self):
        data_td = self.image_data_td

        grid_vals = self._empty_grid.copy()
        if self.shown_quantity == ShownQuantity.P2p:
            grid_vals = np.max(data_td, axis=2) - np.min(data_td, axis=2)

        elif self.shown_quantity == ShownQuantity.AbsorptionCoefficient:
            for i, measurement in enumerate(self.sams):
                pos = measurement.position
                if i % 100 == 0:
                    print(f"{round(100 * i / len(self.sams), 2)} % done. ", end=" ")
                    print(f"(Measurement: {i}/{len(self.sams)}, {pos} mm)")

                res = self._calculate_absorption(measurement)
                x_idx, y_idx = self._coords_to_idx(*pos)
                grid_vals[x_idx, y_idx] = res

        return grid_vals.real

    def _exclude_pixels(self, grid_vals):
        filtered_grid = grid_vals.copy()
        dims = filtered_grid.shape
        for x_idx in range(dims[0]):
            for y_idx in range(dims[1]):
                if self._is_excluded((x_idx, y_idx)):
                    filtered_grid[x_idx, y_idx] = 0

        return filtered_grid

    def plot_image(self, img_extent=None):
        if self.shown_quantity == ShownQuantity.P2p:
            label = "Peak to peak"
        else:
            label = "Absorption coefficient (1/cm)"

        info = self.image_info
        if img_extent is None:
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        grid_vals = self._calc_grid_vals()

        grid_vals = grid_vals[w0:w1, h0:h1]

        grid_vals = self._exclude_pixels(grid_vals)

        if self.options["log_scale"]:
            grid_vals = np.log10(grid_vals)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.image_info["extent"]

        if self.options["log_scale"]:
            self.options["cbar_min"] = np.log10(self.options["cbar_min"])
            self.options["cbar_max"] = np.log10(self.options["cbar_max"])

        try:
            cbar_min = np.min(grid_vals[grid_vals > self.options["cbar_min"]])
            if self.options["cbar_max"] == np.inf:
                cbar_max = np.mean(grid_vals)
            else:
                cbar_max = np.max(grid_vals[grid_vals < self.options["cbar_max"]])
        except ValueError:
            print("Check cbar bounds")
            cbar_min = np.min(grid_vals[grid_vals > 0])
            cbar_max = np.max(grid_vals[grid_vals < np.inf])

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

        ax.set_title(f"{self.shown_quantity.name} at {self.selected_frequency} THz")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        cbar = fig.colorbar(img)

        cbar.set_label(label, rotation=270, labelpad=30)

    def get_measurement(self, x, y, meas_type=MeasType.Sam):
        if meas_type == meas_type.Ref:
            meas_list = self.refs
        elif meas_type == meas_type.Sam:
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

    def get_ref(self, both=False, normalize=False, sub_offset=False, coords=None, ret_meas=False):
        if coords is not None:
            closest_sam = self.get_measurement(*coords, meas_type=MeasType.Sam)

            closest_ref, best_fit_val = None, np.inf
            for ref_meas in self.refs:
                val = np.abs((closest_sam.meas_time - ref_meas.meas_time).total_seconds())
                if val < best_fit_val:
                    best_fit_val = val
                    closest_ref = ref_meas

            chosen_ref = closest_ref
        else:
            chosen_ref = self.refs[-1]

        ref_td = chosen_ref.get_data_td(get_raw=True)

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

    def system_stability(self, selected_freq_=0.800):
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))

        ref_ampl_arr, ref_angle_arr = [], []

        t0 = self.refs[0].meas_time
        meas_times = [(ref.meas_time - t0).total_seconds() / 3600 for ref in self.refs]
        for i, ref in enumerate(self.refs):
            ref_td = ref.get_data_td()
            ref_fd = do_fft(ref_td)

            ref_ampl_arr.append(np.sum(np.abs(ref_fd[f_idx, 1])) / 1)
            phi = np.angle(ref_fd[f_idx, 1])
            ref_angle_arr.append(phi)
        ref_angle_arr = np.unwrap(ref_angle_arr)
        ref_angle_arr -= np.mean(ref_angle_arr)
        ref_ampl_arr -= np.mean(ref_ampl_arr)

        plt.figure("System stability amplitude")
        plt.title(f"Reference amplitude at {selected_freq_} THz")
        plt.plot(meas_times, ref_ampl_arr, label=t0)
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Amplitude (Arb. u.)")

        plt.figure("System stability angle")
        plt.title(f"Reference phase at {selected_freq_} THz")
        plt.plot(meas_times, ref_angle_arr, label=t0)
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Phase (rad)")


if __name__ == '__main__':

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        save_fig(fig_label)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            plt.legend()

    plt.show()
