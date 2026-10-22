#!/usr/bin/env python3
"""Compare automatic and fixed aperture-flux corrections."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aspired import spectral_reduction

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "test" / "test_data" / "v_e_20180810_12_1_0_0.fits.gz"
DEFAULT_OUTPUT = ROOT / "docs" / "source" / "_static"
APERTURE_HALFWIDTHS = (1, 2, 5)
REFERENCE_HALFWIDTH = 10
SPECTRUM_ID = 0

PLOT_STYLES = {
    "No correction": {"color": "#404040", "linestyle": "-"},
    "Automatic Gaussian": {"color": "#0072B2", "linestyle": "-"},
    "Ground truth factor": {"color": "#009E73", "linestyle": "--"},
    "Manual factor 10% low": {"color": "#D55E00", "linestyle": ":"},
}


def create_twodspec(input_path):
    """Create and trace the requested SPRAT frame."""
    twodspec = spectral_reduction.TwoDSpec(
        str(input_path),
        spatial_mask=np.arange(30, 200),
        spec_mask=np.arange(50, 1024),
        cosmicray=False,
        log_file_name=None,
        log_level="CRITICAL",
        saxis=1,
        flip=False,
    )
    twodspec.ap_trace(nspec=1, fit_deg=3, ap_faint=0, display=False)
    return twodspec


def extract(twodspec, apwidth, flux_correction):
    """Run one extraction and copy its count array before the next run."""
    twodspec.ap_extract(
        apwidth=apwidth,
        skysep=5,
        skywidth=7,
        skydeg=1,
        optimal=True,
        algorithm="horne86",
        model="gauss",
        flux_correction=flux_correction,
        display=False,
    )
    return np.asarray(
        twodspec.spectrum_list[SPECTRUM_ID].count, dtype=float
    ).copy()


def reference_mask(reference):
    """Mask low-reference-count pixels that produce unstable ratios."""
    finite_reference = reference[np.isfinite(reference)]
    threshold = max(1.0, 0.1 * np.nanmedian(np.abs(finite_reference)))
    return np.isfinite(reference) & (np.abs(reference) > threshold)


def manual_reference_correction(reference, uncorrected):
    """Derive the fixed factor that matches the wide-aperture reference."""
    valid = reference_mask(reference) & np.isfinite(uncorrected)
    valid &= np.abs(uncorrected) > 0
    return float(np.nanmedian(reference[valid] / uncorrected[valid]))


def plot_spectra(results, manual_factors, output_path):
    """Plot each raw extracted spectrum on a logarithmic count scale."""
    figure, axes = plt.subplots(
        len(APERTURE_HALFWIDTHS),
        1,
        figsize=(10, 10),
        sharex=True,
        constrained_layout=True,
    )

    for axis, apwidth in zip(axes, APERTURE_HALFWIDTHS):
        for label, style in PLOT_STYLES.items():
            counts = results[apwidth][label]
            pixels = np.arange(counts.size)
            valid = np.isfinite(counts) & (counts > 0)
            axis.plot(
                pixels[valid],
                counts[valid],
                label=label,
                linewidth=1.0,
                **style,
            )

        axis.set_yscale("log")
        axis.set_xlim(300, 900)
        axis.set_ylim(2e2, 2e3)
        axis.set_ylabel("Extracted count")
        axis.set_title(
            "Half-width = {} pixels; ground truth factor = {:.4f}".format(
                apwidth, manual_factors[apwidth]
            )
        )
        axis.grid(alpha=0.25)

    axes[0].legend(loc="lower right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Dispersion pixel in cropped frame")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_summary(reference, results, output_path):
    """Plot median relative flux as a function of extraction half-width."""
    mask = reference_mask(reference)
    summary = {label: [] for label in PLOT_STYLES}

    for apwidth in APERTURE_HALFWIDTHS:
        for label in PLOT_STYLES:
            counts = results[apwidth][label]
            valid = mask & np.isfinite(counts)
            summary[label].append(
                float(np.nanmedian(counts[valid] / reference[valid]))
            )

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, style in PLOT_STYLES.items():
        axis.plot(
            APERTURE_HALFWIDTHS,
            summary[label],
            marker="o",
            label=label,
            **style,
        )

    values = np.concatenate(list(summary.values()))
    axis.axhline(1.0, color="#202020", linewidth=0.8, alpha=0.6)
    axis.set_xlim(
        min(APERTURE_HALFWIDTHS) - 0.2, max(APERTURE_HALFWIDTHS) + 0.2
    )
    axis.set_ylim(np.nanmin(values) - 0.025, np.nanmax(values) + 0.025)
    axis.set_xticks(APERTURE_HALFWIDTHS)
    axis.set_xlabel("Extraction half-width (pixels)")
    axis.set_ylabel("Median count / wide-aperture reference")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run_demo(input_path, output_directory):
    """Extract the test frame and write the comparison plots."""
    twodspec = create_twodspec(input_path)
    reference = extract(twodspec, REFERENCE_HALFWIDTH, False)
    manual_factors = {}
    results = {}

    for apwidth in APERTURE_HALFWIDTHS:
        uncorrected = extract(twodspec, apwidth, False)
        manual_factor = manual_reference_correction(reference, uncorrected)
        manual_factors[apwidth] = manual_factor
        results[apwidth] = {
            "No correction": uncorrected,
            "Automatic Gaussian": extract(twodspec, apwidth, True),
            "Ground truth factor": extract(twodspec, apwidth, manual_factor),
            "Manual factor 10% low": extract(
                twodspec, apwidth, 0.9 * manual_factor
            ),
        }

    output_directory.mkdir(parents=True, exist_ok=True)
    spectra_plot = output_directory / "fig_07_flux_correction_spectra.png"
    summary_plot = output_directory / "fig_08_flux_correction_summary.png"
    plot_spectra(results, manual_factors, spectra_plot)
    plot_summary(reference, results, summary_plot)

    for apwidth in APERTURE_HALFWIDTHS:
        manual_factor = manual_factors[apwidth]
        print(
            "half-width={}: reference factor={:.4f}; deliberately low "
            "factor={:.4f}".format(apwidth, manual_factor, 0.9 * manual_factor)
        )
    print("Wrote {}".format(spectra_plot))
    print("Wrote {}".format(summary_plot))


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate ASPIRED aperture flux corrections."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="SPRAT FITS frame to extract.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the PNG plots.",
    )
    arguments = parser.parse_args()
    run_demo(arguments.input, arguments.output_dir)


if __name__ == "__main__":
    main()
