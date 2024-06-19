import numpy as np
from scipy.signal import windows


def window(data_td, win_width=None, win_start=None, slope=0.15):
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

    return np.array([t, y_win], dtype=float).T
