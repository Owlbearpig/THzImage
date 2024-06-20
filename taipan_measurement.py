from numpy.fft import fft, fftfreq
from datetime import datetime
import numpy as np
from functions import window
import enum


class MeasType(enum.Enum):
    Ref = 0
    Sam = 1


class Measurement:

    def __init__(self, filepath=None, post_process_config=None):
        self.filepath = filepath
        self.meas_time = None
        self.meas_type = None
        self.sample_name = None
        self.position = [None, None]

        self.post_process_config = post_process_config
        self._data_fd, self._data_td = None, None
        self.pre_process_done = False

        self._set_metadata()

    def __repr__(self):
        return str(self.filepath)

    def _set_metadata(self):

        # set time
        date_formats = [("%Y-%m-%dT%H-%M-%S.%f", 26), ("%Y-%m-%d_%H-%M-%S", 19)]
        for date_format in date_formats:
            try:
                length = date_format[1]
                self.meas_time = datetime.strptime(str(self.filepath.stem)[:length], date_format[0])
                break
            except ValueError:
                continue
        if self.meas_time is None:
            raise ValueError

        # set sample name
        try:
            dir_1above, dir_2above = self.filepath.parents[0], self.filepath.parents[1]
            if ("sam" in dir_1above.stem.lower()) or ("ref" in dir_1above.stem.lower()):
                self.sample_name = dir_2above.stem
            else:
                self.sample_name = dir_1above.stem
        except ValueError:
            self.sample_name = "N/A"

        # set measurement type
        if "ref" in str(self.filepath.stem).lower():
            self.meas_type = MeasType.Ref
        else:
            self.meas_type = MeasType.Sam

        # set position
        try:
            str_splits = str(self.filepath).split("_")
            x = float(str_splits[-2].split(" mm")[0])
            y = float(str_splits[-1].split(" mm")[0])
            self.position = [x, y]
        except ValueError:
            self.position = [0, 0]

    def do_preprocess(self, force=False):
        if self.pre_process_done and not force:
            return

        if self.post_process_config["sub_offset"]:
            self._data_td[:, 1] -= np.mean(self._data_td[:10, 1])
        if self.post_process_config["en_windowing"]:
            self._data_td = window(self._data_td)
        if self.post_process_config["normalize"]:
            self._data_td[:, 1] /= np.max(self._data_td[:, 1])

        self.pre_process_done = True

    def get_data_td(self, get_raw=False):
        def read_file(file_path):
            try:
                return np.loadtxt(file_path)
            except ValueError:
                return np.loadtxt(file_path, delimiter=",")

        if get_raw:
            return read_file(self.filepath)

        if self._data_td is None:
            self._data_td = read_file(self.filepath)

        self.do_preprocess()

        return self._data_td

    def get_data_fd(self, pos_freqs_only=True, reversed_time=True):
        if self._data_fd is not None:
            return self._data_fd

        data_td = self.get_data_td()
        t, y = data_td[:, 0], data_td[:, 1]

        if reversed_time:
            y = np.flip(y)

        dt = float(np.mean(np.diff(t)))
        freqs, data_fd = fftfreq(n=len(t), d=dt), fft(y)
        if pos_freqs_only:
            pos_slice = freqs >= 0
            self._data_fd = np.array([freqs[pos_slice], data_fd[pos_slice]]).T
        else:
            self._data_fd = np.array([freqs, data_fd]).T

        return self._data_fd
