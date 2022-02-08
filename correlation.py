# -*- coding: utf-8 -*-
import math
from time import time
import numpy as np
from scipy import signal
from scipy import stats
from astropy.io import fits
from matplotlib.pyplot import *

ion()

# SPRAT
data = fits.open("v_e_20180704_32_1_0_2.fits")[0].data[40:, 110:]
Saxis = 1
Waxis = 0

# FRODO
data = fits.open("r_w_20191029_3_1_1_2.fits")[0].data
Saxis = 0
Waxis = 1

# Get the shape of the 2D spectrum and define upsampling ratio
scaling = 10
if Saxis == 1:
    N_wave = len(data[0])
    N_spatial = len(data)
else:
    N_wave = len(data)
    N_spatial = len(data[0])

N_resample = N_spatial * scaling

N_window = 25
# window size
w_size = N_wave // N_window
# w_shift_frac = 2
# w_shift = math.ceil(w_size / w_shift_frac)
w_iter = math.floor((N_wave / w_size - 1))
data_split = np.array_split(data, w_iter, axis=Saxis)

lines_ref_init = np.nanmedian(data_split[0], axis=Saxis)
lines_ref_init_resampled = signal.resample(lines_ref_init, N_resample)

# linear scaling limits
transform_min = 1.0
transform_max = 1.001
transform_step = 0.005
transform_scale = np.arange(transform_min, transform_max, transform_step)

# estimate the 5-th percentile as the sky background level
lines_ref = lines_ref_init_resampled - np.percentile(
    lines_ref_init_resampled, 5
)

shift_solution = np.zeros(w_iter)
scale_solution = np.ones(w_iter)

# maximum shift (SEMI-AMPLITUDE) from the neighbour (pixel)
tol = 3
tol_len = int(tol * scaling)

# trace each individual spetrum one by one
# figure(10)
# clf()

# Scipy correlate method

time1 = time.time()
for i in range(w_iter):
    # smooth by taking the median
    lines = np.nanmedian(data_split[i], axis=Saxis)
    lines = signal.resample(lines, N_resample)
    lines = lines - np.percentile(lines, 5)
    corr_val = np.zeros(len(transform_scale))
    corr_idx = np.zeros(len(transform_scale))
    # upsample by the same amount as the reference
    for j, scale in enumerate(transform_scale):
        # Upsampling the reference lines
        lines_ref_j = signal.resample(lines_ref, int(N_resample * scale))
        # find the linear shift
        corr = signal.correlate(lines_ref_j, lines)
        # only consider the defined range of shift tolerance
        corr = corr[N_resample - 1 - tol_len : N_resample + tol_len]
        # Maximum corr position is the shift
        corr_val[j] = np.nanmax(corr)
        corr_idx[j] = np.nanargmax(corr) - tol_len
    # Maximum corr_val position is the scaling
    shift_solution[i] = corr_idx[np.nanargmax(corr_val)]
    scale_solution[i] = transform_scale[np.nanargmax(corr_val)]
    lines_ref = lines

time2 = time.time()
print(time2 - time1)

# Get the shift and scale for the trace
# idx = np.nanargmax(corr_val, axis=1)
# shift_solution = np.array([corr_idx[i,j] for i, j in enumerate(idx)])
# scale_solution = transform_scale[idx]

N_spec = 144
# Find the spectral position in the middle of the gram in the upsampled
# pixel location location
peaks = signal.find_peaks(
    signal.resample(
        np.nanmedian(data_split[w_iter // 2], axis=Saxis), N_resample
    ),
    distance=5,
    prominence=1,
)
# Sort the positions by the prominences, and return to the original
# scale (i.e. with subpixel position)
spec_init = (
    np.sort(peaks[0][np.argsort(-peaks[1]["prominences"])][:N_spec]) / scaling
)
# Create array to populate the spectral locations
spec = np.zeros((len(spec_init), len(data_split)))
spec_val = np.zeros((len(spec_init), len(data_split)))
# Populate the initial values
spec[:, w_iter // 2] = spec_init

spec_wave = np.arange(len(data_split)) * w_size + w_size / 2.0

# Looping through pixels larger than middle pixel
for i in range(w_iter // 2 + 1, len(data_split)):
    spec[:, i] = (
        spec[:, i - 1]
        * scaling
        * int(N_resample * scale_solution[i])
        / N_resample
        - shift_solution[i]
    ) / scaling
    spec_val[:, i] = signal.resample(
        np.nanmedian(data_split[i], axis=Saxis),
        int(N_resample * scale_solution[i]),
    )[(spec[:, i] * scale_solution[i]).astype("int")]

# Looping through pixels smaller than middle pixel
for i in range(w_iter // 2, -1, -1):
    spec[:, i] = (
        (spec[:, i + 1] * scaling + shift_solution[i + 1])
        / (int(N_resample * scale_solution[i + 1]) / N_resample)
        / scaling
    )
    spec_val[:, i] = signal.resample(
        np.nanmedian(data_split[i], axis=Saxis),
        int(N_resample * scale_solution[i]),
    )[(spec[:, i] * scale_solution[i]).astype("int")]

# Plot
vmin = np.nanpercentile(np.log10(data), 5)
vmax = np.nanpercentile(np.log10(data), 95)

figure(1, figsize=(18, 9))
clf()
if Saxis == 1:
    imshow(np.log10(data), vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
else:
    imshow(
        np.log10(np.transpose(data)),
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower",
    )

for i in range(len(spec)):
    ap_p = np.polyfit(spec_wave, spec[i], max(1, N_window // 10))
    ap = np.polyval(ap_p, np.arange(N_wave))
    plot(np.arange(N_wave), ap, lw=1, color="grey")
    scatter(spec_wave, spec[i], s=5, color="black", zorder=15)

scatter(
    np.ones(len(spec)) * spec_wave[w_iter // 2],
    spec[:, w_iter // 2],
    zorder=20,
    s=10,
    color="red",
)

ylabel("Spatial Direction")
xlabel("Wavelength Direction")

tight_layout()
