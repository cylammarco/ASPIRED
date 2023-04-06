import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage, signal

# 1D
a = np.ones(100)
b = np.ones(100)
one_tenth = len(a) // 10

a[37] = 100
b[44] = 100


a_x2 = ndimage.zoom(a, zoom=2, order=3)
a_x3 = ndimage.zoom(a, zoom=3, order=3)
a_x4 = ndimage.zoom(a, zoom=4, order=3)
a_x5 = ndimage.zoom(a, zoom=5, order=3)

b_x2 = ndimage.zoom(b, zoom=2, order=3)
b_x3 = ndimage.zoom(b, zoom=3, order=3)
b_x4 = ndimage.zoom(b, zoom=4, order=3)
b_x5 = ndimage.zoom(b, zoom=5, order=3)

one_tenth_x2 = len(a_x2) // 10
one_tenth_x3 = len(a_x3) // 10
one_tenth_x4 = len(a_x4) // 10
one_tenth_x5 = len(a_x5) // 10

print(
    len(a)
    - 2 * one_tenth
    - np.argmax(
        signal.correlate(a[one_tenth:-one_tenth], b[one_tenth:-one_tenth])
    )
    - 1
)
print(
    len(a_x2)
    - 2 * one_tenth_x2
    - np.argmax(
        signal.correlate(
            a_x2[one_tenth_x2:-one_tenth_x2], b_x2[one_tenth_x2:-one_tenth_x2]
        )
    )
    - 1
)
print(
    len(a_x3)
    - 2 * one_tenth_x3
    - np.argmax(
        signal.correlate(
            a_x3[one_tenth_x3:-one_tenth_x3], b_x3[one_tenth_x3:-one_tenth_x3]
        )
    )
    - 1
)
print(
    len(a_x4)
    - 2 * one_tenth_x4
    - np.argmax(
        signal.correlate(
            a_x4[one_tenth_x4:-one_tenth_x4], b_x4[one_tenth_x4:-one_tenth_x4]
        )
    )
    - 1
)
print(
    len(a_x5)
    - 2 * one_tenth_x5
    - np.argmax(
        signal.correlate(
            a_x5[one_tenth_x5:-one_tenth_x5], b_x5[one_tenth_x5:-one_tenth_x5]
        )
    )
    - 1
)


# 2D
simulated_image = np.ones((100, 1000))

simulated_image[46] = 2.0
simulated_image[47] = 5.0
simulated_image[48] = 10.0
simulated_image[49] = 30.0
simulated_image[50] = 50.0
simulated_image[51] = 30.0
simulated_image[52] = 10.0
simulated_image[53] = 5.0
simulated_image[54] = 2.0

for i in range(100):
    simulated_image[i, 223 + i] += 100.0
    simulated_image[i, 671 + i] += 100.0


upsample_factor = 5

n_bin = 15
bin_size = 3
bin_half_size = bin_size / 2 * upsample_factor
n_down = int(n_bin // 2)
n_up = int(n_bin // 2)

img_tmp = ndimage.zoom(simulated_image, zoom=upsample_factor, order=3)

y_tmp = (
    ndimage.zoom(
        np.array(np.ones(len(simulated_image)) * 49.0),
        zoom=upsample_factor,
        order=3,
    )
    * upsample_factor
)

plt.figure(1)

ref = y_tmp[len(y_tmp) // 2]

# The x-coordinates of the trace (of length len_trace)
x = np.arange(len(simulated_image[0]) * upsample_factor).astype("int")

start = -bin_half_size
end = start + bin_size * upsample_factor + 1
print(int(np.round(ref + start)), int(np.round(ref + end)))
# s for "flattened signal of the slice"
s = [
    np.nansum(
        [
            img_tmp[
                int(ref + start) : int(ref + end),
                i,
            ]
            for i in x
        ],
        axis=1,
    )
]

plt.plot([n_bin // 2, n_bin // 2], [ref + start, ref + end])

# Get the length of 10% of the dispersion direction
# Do not use the first and last 10% for cross-correlation
one_tenth = len(s[0]) // 10

s[0] -= np.nanpercentile(s[0], 5.0)
s[0] -= min(s[0][one_tenth:-one_tenth])
s[0] /= max(s[0][one_tenth:-one_tenth])
s_down = []
s_up = []

# Loop through the spectra below the trace
for k in range(n_down):
    start = k * bin_half_size
    end = start + bin_size * upsample_factor + 1
    print(int(np.round(ref - end)), int(np.round(ref - start)))
    # Note the start and end are counting up, while the
    # indices are becoming smaller.
    s_down.append(
        np.nansum(
            [
                img_tmp[
                    int(np.round(ref - end)) : int(np.round(ref - start)),
                    i,
                ]
                for i in x
            ],
            axis=1,
        )
    )
    s_down[k] -= np.nanpercentile(s_down[k], 5.0)
    s_down[k] -= min(s_down[k][one_tenth:-one_tenth])
    s_down[k] /= max(s_down[k][one_tenth:-one_tenth])
    plt.plot(
        [n_bin // 2 - k - 1, n_bin // 2 - k - 1],
        [ref - end, ref - start],
        ls="dotted",
    )


# Loop through the spectra above the trace
for k in range(n_up):
    start = k * bin_half_size
    end = start + bin_size * upsample_factor + 1
    print(int(np.round(ref + start)), int(np.round(ref + end)))
    s_up.append(
        np.nansum(
            [
                img_tmp[
                    int(np.round(ref + start)) : int(np.round(ref + end)),
                    i,
                ]
                for i in x
            ],
            axis=1,
        )
    )
    s_up[k] -= np.nanpercentile(s_up[k], 5.0)
    s_up[k] -= min(s_up[k][one_tenth:-one_tenth])
    s_up[k] /= max(s_up[k][one_tenth:-one_tenth])
    plt.plot(
        [n_bin // 2 + k + 1, n_bin // 2 + k + 1],
        [ref + start, ref + end],
        ls="dashed",
    )


s_all = s_down[::-1] + s + s_up


y_trace_upsampled = np.arange(-n_down + 1, n_up + 1) * bin_half_size + ref

# correlate with the neighbouring slice to compute the shifts
shift_upsampled = np.zeros_like(y_trace_upsampled)

for i in range(1, len(s_all)):
    # Note: indice n_down is s
    corr = signal.correlate(
        s_all[i][one_tenth:-one_tenth], s_all[i - 1][one_tenth:-one_tenth]
    )
    # spec_size_tmp is the number of pixel in the upsampled axis
    shift_upsampled[i - 1 :] += (
        len(s[0])
        - 2 * one_tenth
        - np.argwhere(corr == corr[np.argmax(corr)])[0]
        - 1
    )

# Turn the shift to relative to the spectrum
shift_upsampled -= shift_upsampled[n_down]

coeff = np.polynomial.polynomial.polyfit(
    y_trace_upsampled,
    shift_upsampled,
    1,
)

img_tmp = simulated_image.copy()

for j, _ in enumerate(img_tmp):
    shift_j = np.polynomial.polynomial.polyval(j, coeff)
    img_tmp[j] = np.roll(img_tmp[j], int(np.round(shift_j)))


plt.figure(3)
plt.clf()
plt.imshow(img_tmp)
