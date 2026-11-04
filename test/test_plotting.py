import os

import numpy as np
import pytest
from aspired import plotting
from aspired.plotting import FigureSpec, Panel, Trace

HERE = os.path.dirname(os.path.realpath(__file__))
OUTPUT = os.path.join(HERE, "test_output")


@pytest.fixture(autouse=True)
def restore_global_library():
    original = plotting.get_plot_library()
    yield
    plotting.set_plot_library(original)


def make_spec():
    x = np.linspace(1.0, 10.0, 50)
    top = Panel(
        traces=[
            Trace(x=x, y=x**2, name="line", color="royalblue"),
            Trace(x=x, y=x, name="right", color="black", secondary=True),
            Trace(kind="markers", x=x, y=x, name="points", color="red"),
        ],
        xlabel="Wavelength / A",
        ylabel="Flux",
        y2label="Sensitivity",
        y2scale="log",
        yscale="log",
        y_domain=(0.5, 1.0),
    )
    bottom = Panel(
        traces=[
            Trace(
                kind="heatmap",
                z=np.random.default_rng(0).random((5, 6)),
                colorbar_title="count",
            )
        ],
        y_domain=(0.0, 0.4),
    )
    return FigureSpec(
        panels=[top, bottom],
        title="spec",
        width=640,
        height=480,
        scale_toggle=True,
    )


def test_default_global_library_is_matplotlib():
    assert plotting.get_plot_library() == "matplotlib"


def test_global_and_local_selection():
    plotting.set_plot_library("plotly")
    assert plotting.resolve_plot_library() == "plotly"
    assert plotting.resolve_plot_library("matplotlib") == "matplotlib"
    plotting.set_plot_library("MATPLOTLIB")
    assert plotting.resolve_plot_library(None) == "matplotlib"


def test_unknown_library_raises():
    with pytest.raises(ValueError):
        plotting.set_plot_library("bokeh")
    with pytest.raises(ValueError):
        plotting.resolve_plot_library("bokeh")


def test_build_figure_types():
    spec = make_spec()
    mpl_fig = plotting.build_figure(spec, "matplotlib")
    plotly_fig = plotting.build_figure(spec, "plotly")
    assert type(mpl_fig).__module__.startswith("matplotlib")
    assert type(plotly_fig).__module__.startswith("plotly")
    plotting.finalise_figure(mpl_fig, display=False)


def test_global_choice_drives_build_figure():
    plotting.set_plot_library("plotly")
    fig = plotting.build_figure(make_spec())
    assert type(fig).__module__.startswith("plotly")


def test_matplotlib_save_and_return():
    fig = plotting.build_figure(make_spec(), "matplotlib")
    base = os.path.join(OUTPUT, "test_plotting_mpl")
    result = plotting.finalise_figure(
        fig,
        display=False,
        save_fig=True,
        fig_type="png+svg+iframe",
        filename=base,
        return_jsonstring=True,
        return_figure=True,
    )
    assert result is fig
    for ext in ["png", "svg"]:
        assert os.path.isfile(base + "." + ext)
        os.remove(base + "." + ext)
    assert not os.path.isfile(base + ".iframe")


def test_plotly_save_and_json():
    fig = plotting.build_figure(make_spec(), "plotly")
    base = os.path.join(OUTPUT, "test_plotting_plotly")
    result = plotting.finalise_figure(
        fig,
        display=False,
        save_fig=True,
        fig_type="iframe+png",
        filename=base,
        return_jsonstring=True,
    )
    assert isinstance(result, str)
    for ext in ["iframe", "png"]:
        assert os.path.isfile(base + "." + ext)
        os.remove(base + "." + ext)


def test_default_filename_used():
    fig = plotting.build_figure(make_spec(), "matplotlib")
    plotting.finalise_figure(
        fig,
        display=False,
        save_fig=True,
        fig_type="png",
        default_filename=os.path.join(OUTPUT, "test_plotting_default"),
    )
    path = os.path.join(OUTPUT, "test_plotting_default.png")
    assert os.path.isfile(path)
    os.remove(path)


@pytest.mark.parametrize("library", ["matplotlib", "plotly"])
def test_inspect_standard_both_libraries(library):
    from aspired import spectral_reduction

    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.load_standard(target="cd32d9927")
    result = onedspec.inspect_standard(
        display=False, plot_library=library, return_figure=True
    )
    assert type(result).__module__.startswith(library)


@pytest.mark.parametrize(
    "limits", [(np.nan, np.nan), (5.0, 1.0), (None, None)]
)
def test_heatmap_colour_limits_are_sanitised(limits):
    z = np.log10(np.random.default_rng(1).normal(size=(6, 6)))
    spec = FigureSpec(
        panels=[
            Panel(
                traces=[
                    Trace(kind="heatmap", z=z, zmin=limits[0], zmax=limits[1])
                ]
            )
        ]
    )
    fig = plotting.build_figure(spec, "matplotlib")
    plotting.finalise_figure(fig, display=False)


@pytest.mark.parametrize(
    "yrange", [(np.nan, 10.0), (-1.0, 5.0), (np.inf, 1.0), (10.0, 1.0)]
)
@pytest.mark.parametrize("library", ["matplotlib", "plotly"])
def test_invalid_axis_ranges_do_not_raise(yrange, library):
    x = np.linspace(1.0, 10.0, 20)
    spec = FigureSpec(
        panels=[
            Panel(
                traces=[Trace(x=x, y=x)],
                yscale="log",
                yrange=yrange,
                xrange=(np.nan, np.nan),
            )
        ]
    )
    fig = plotting.build_figure(spec, library)
    plotting.finalise_figure(fig, display=False)
