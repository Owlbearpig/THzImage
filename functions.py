import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


def do_fft(data_td):
    data_td = np.nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0])))
    freqs, data_fd = np.fft.rfftfreq(n=len(data_td[:, 0]), d=dt), np.conj(np.fft.rfft(data_td[:, 1]))

    return np.array([freqs, data_fd]).T


def to_db(data_fd):
    if data_fd.ndim == 2:
        return 20 * np.log10(np.abs(data_fd[:, 1]))
    else:
        return 20 * np.log10(np.abs(data_fd))


def window(data_td, win_width=None, win_start=None, slope=0.15, en_plot=False):
    t, y = data_td[:, 0].real, data_td[:, 1].real

    pulse_width = 10  # ps
    dt = np.mean(np.diff(t))

    if win_width is None:
        win_width = int(pulse_width / dt)
    else:
        win_width = int(win_width / dt)

    if win_width > len(y):
        win_width = len(y)

    if win_start is None:
        win_center = np.argmax(np.abs(y))
        win_start = win_center - int(win_width / 2)
    else:
        win_start = int(win_start / dt)

    if win_start < 0:
        win_start = 0

    pre_pad = np.zeros(win_start)
    window_arr = windows.tukey(win_width, slope)

    post_pad = np.zeros(len(y) - win_width - win_start)

    window_arr = np.concatenate((pre_pad, window_arr, post_pad))

    y_win = y * window_arr

    if en_plot:
        plt.figure("Windowing")
        plt.plot(t, y, label="Sam. before windowing")
        plt.plot(t, np.max(np.abs(y)) * window_arr, label="Window")
        plt.plot(t, y_win, label="Sam. after windowing")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

    return np.array([t, y_win], dtype=float).T


def save_fig(fig_label, save_dir=None, filename=None, **kwargs):
    if filename is None:
        filename = fig_label

    if save_dir is None:
        save_dir = Path(mpl.rcParams["savefig.directory"])

    fig = plt.figure(fig_label)
    fig.set_size_inches((16, 9), forward=False)
    plt.savefig(save_dir / (filename.replace(" ", "_") + ".pdf"),
                bbox_inches='tight', dpi=300, pad_inches=0, **kwargs)


def unwrap(data_fd):
    if data_fd.ndim == 2:
        y = np.nan_to_num(data_fd[:, 1])
    else:
        y = np.nan_to_num(data_fd)
        return np.unwrap(np.angle(y))

    return np.array([data_fd[:, 0].real, np.unwrap(np.angle(y))]).T


def do_ifft(data_fd, hermitian=True, shift=0, flip=False):
    freqs, y_fd = data_fd[:, 0].real, data_fd[:, 1]

    y_fd = np.nan_to_num(y_fd)

    if hermitian:
        y_fd = np.concatenate((np.conj(y_fd), np.flip(y_fd[1:])))
        # y_fd = np.concatenate((y_fd, np.flip(np.conj(y_fd[1:]))))
        """
        * ``a[0]`` should contain the zero frequency term,
        * ``a[1:n//2]`` should contain the positive-frequency terms,
        * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
          increasing order starting from the most negative frequency.
        """

    y_td = np.fft.ifft(y_fd)
    df = np.mean(np.diff(freqs))
    n = len(y_td)
    t = np.arange(0, n) / (n * df)

    # t = np.linspace(0, len(y_td)*df, len(y_td))
    # t += 885

    # y_td = np.flip(y_td)
    dt = np.mean(np.diff(t))
    shift = int(shift / dt)

    y_td = np.roll(y_td, shift)

    if flip:
        y_td = np.flip(y_td)

    return np.array([t, y_td]).T


def phase_correction(data_fd, disable=False, fit_range=None, en_plot=False, extrapolate=False, rewrap=False,
                     ret_fd=False, both=False):
    freqs = data_fd[:, 0].real

    if disable:
        return np.array([freqs, np.unwrap(np.angle(data_fd[:, 1]))]).T

    phase_unwrapped = unwrap(data_fd)

    if fit_range is None:
        fit_range = [0.25, 0.85]

    fit_slice = (freqs >= fit_range[0]) * (freqs <= fit_range[1])

    p = np.polyfit(freqs[fit_slice], phase_unwrapped[fit_slice, 1], 1)

    phase_corrected = phase_unwrapped[:, 1] - p[1].real

    if en_plot:
        plt.figure("phase_correction")
        plt.plot(freqs, phase_unwrapped[:, 1], label="Unwrapped phase")
        plt.plot(freqs, phase_corrected, label="Shifted phase")
        # plt.plot(freqs, freqs * p[0].real, label="Lin. fit (slope*freq)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")
        plt.legend()

    if extrapolate:
        phase_corrected = p[0].real * freqs

    if rewrap:
        phase_corrected = np.angle(np.exp(1j * phase_corrected))

    y = np.abs(data_fd[:, 1]) * np.exp(1j * phase_corrected)
    if both:
        return do_ifft(np.array([freqs, y]).T), np.array([freqs, y]).T

    if ret_fd:
        return np.array([freqs, y]).T
    else:
        return np.array([freqs, phase_corrected]).T

