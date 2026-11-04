"""
Unified plotting layer for ASPIRED.

All diagnostic figures are described once with the light-weight ``Trace``,
``Panel`` and ``FigureSpec`` containers and rendered by either matplotlib or
plotly. The library is selected globally with :func:`set_plot_library` or for
a single call with the ``plot_library`` argument of the plotting methods.

"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("plotting")

SUPPORTED_LIBRARIES = ("matplotlib", "plotly")
STATIC_FIGURE_TYPES = ("jpg", "png", "svg", "pdf")

_PLOT_CONFIG = {"library": "matplotlib"}


def set_plot_library(library: str = "matplotlib"):
    """
    Set the plotting library used when no ``plot_library`` is given.

    Parameters
    ----------
    library: str (Default: 'matplotlib')
        Either 'matplotlib' or 'plotly'.

    """

    library = str(library).lower()
    if library not in SUPPORTED_LIBRARIES:
        raise ValueError(
            f"Unknown plotting library '{library}', choose from "
            f"{SUPPORTED_LIBRARIES}."
        )
    _PLOT_CONFIG["library"] = library
    logger.info("Global plotting library is set to %s.", library)


def get_plot_library():
    """
    Return the name of the globally selected plotting library.

    """

    library = _PLOT_CONFIG["library"]
    logger.debug("Global plotting library is %s.", library)
    return library


def resolve_plot_library(plot_library: Optional[str] = None):
    """
    Return the library to use: the local choice if given, else the global.

    Parameters
    ----------
    plot_library: str or None (Default: None)
        Either 'matplotlib', 'plotly' or None to use the global setting.

    """

    if plot_library is None:
        return get_plot_library()
    plot_library = str(plot_library).lower()
    if plot_library not in SUPPORTED_LIBRARIES:
        raise ValueError(
            f"Unknown plotting library '{plot_library}', choose from "
            f"{SUPPORTED_LIBRARIES}."
        )
    return plot_library


@dataclass
class Trace:
    """
    One drawable item.

    kind: 'line', 'markers', 'lines+markers' or 'heatmap'. Heatmaps use z
    (and optionally x, y) and ignore the styling of lines.
    secondary: draw against the right-hand y-axis of the panel.

    """

    kind: str = "line"
    x: Any = None
    y: Any = None
    z: Any = None
    name: Optional[str] = None
    color: Optional[str] = None
    width: float = 2.0
    marker_size: float = 6.0
    opacity: float = 1.0
    dash: Optional[str] = None
    secondary: bool = False
    show_legend: Optional[bool] = None
    colorscale: str = "Viridis"
    zmin: Optional[float] = None
    zmax: Optional[float] = None
    colorbar_title: Optional[str] = None


@dataclass
class Panel:
    """
    One set of axes. ``x_domain`` and ``y_domain`` are fractions of the
    figure (0 to 1) so that several panels can be stacked or tiled.

    """

    traces: List[Trace] = field(default_factory=list)
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    y2label: Optional[str] = None
    xscale: str = "linear"
    yscale: str = "linear"
    y2scale: str = "linear"
    xrange: Optional[Tuple[float, float]] = None
    yrange: Optional[Tuple[float, float]] = None
    y2range: Optional[Tuple[float, float]] = None
    x_domain: Tuple[float, float] = (0.0, 1.0)
    y_domain: Tuple[float, float] = (0.0, 1.0)
    show_grid: bool = True


@dataclass
class FigureSpec:
    """
    Library-independent description of a figure.

    scale_toggle: plotly only, adds a log/linear button for the y-axis. The
    matplotlib figure uses the scale given in the panel.

    """

    panels: List[Panel] = field(default_factory=list)
    title: Optional[str] = None
    width: int = 1280
    height: int = 720
    show_legend: bool = True
    scale_toggle: bool = False


def _to_array(values):
    """
    Convert list-like input to a float array and None stays None.

    """

    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    return array.reshape(-1) if array.ndim == 0 else array


def _plotly_scale_buttons():
    """
    Build the plotly buttons that switch the y-axis between log and linear.

    """

    log_button = dict(
        label="Log Scale",
        method="relayout",
        args=[{"yaxis.type": "log", "title.text": "Log scale"}],
    )
    linear_button = dict(
        label="Linear Scale",
        method="relayout",
        args=[{"yaxis.type": "linear", "title.text": "Linear scale"}],
    )
    return [dict(active=0, buttons=[log_button, linear_button])]


def _plotly_trace(trace: Trace, axis_suffix: str, secondary: bool):
    """
    Convert one Trace to a plotly graph object attached to its axes.

    """

    from plotly import graph_objects as go

    xaxis = "x" + axis_suffix
    yaxis = "y" + axis_suffix
    if secondary:
        yaxis = "y" + str(int(axis_suffix or 1) + 100)
    if trace.kind == "heatmap":
        return go.Heatmap(
            x=_to_array(trace.x),
            y=_to_array(trace.y),
            z=_to_array(trace.z),
            zmin=trace.zmin,
            zmax=trace.zmax,
            colorscale=trace.colorscale,
            colorbar=dict(title=trace.colorbar_title),
            xaxis=xaxis,
            yaxis=yaxis,
        )
    mode = {
        "line": "lines",
        "markers": "markers",
        "lines+markers": "lines+markers",
    }[trace.kind]
    show_legend = trace.name is not None
    if trace.show_legend is not None:
        show_legend = trace.show_legend
    return go.Scatter(
        x=_to_array(trace.x),
        y=_to_array(trace.y),
        mode=mode,
        name=trace.name,
        showlegend=show_legend,
        opacity=trace.opacity,
        line=dict(color=trace.color, width=trace.width, dash=trace.dash),
        marker=dict(color=trace.color, size=trace.marker_size),
        xaxis=xaxis,
        yaxis=yaxis,
    )


def _clean_range(values, scale: str):
    """
    Return an axis range that both libraries can use, or None to let the
    library choose. Ranges with non-finite limits, or with non-positive limits
    on a log axis, are discarded; reversed limits are put in order.

    """

    if values is None:
        return None
    low, high = (float(v) for v in values)
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    if scale == "log" and (low <= 0 or high <= 0):
        return None
    return (min(low, high), max(low, high)) if low != high else None


def _plotly_range(values, scale: str):
    """
    Plotly wants the range of a log axis as exponents, not data values.

    """

    values = _clean_range(values, scale)
    if values is None or scale != "log":
        return values
    return list(np.log10(np.asarray(values, dtype=float)))


def _plotly_axes_layout(panel: Panel, index: int):
    """
    Return the plotly layout entries (axes and domains) for one Panel.

    """

    suffix = "" if index == 0 else str(index + 1)
    layout = {
        "xaxis"
        + suffix: dict(
            title=panel.xlabel,
            type=panel.xscale,
            range=_plotly_range(panel.xrange, panel.xscale),
            domain=list(panel.x_domain),
            anchor="y" + suffix,
            showgrid=panel.show_grid,
        ),
        "yaxis"
        + suffix: dict(
            title=panel.ylabel,
            type=panel.yscale,
            range=_plotly_range(panel.yrange, panel.yscale),
            domain=list(panel.y_domain),
            anchor="x" + suffix,
            showgrid=panel.show_grid,
        ),
    }
    if any(trace.secondary for trace in panel.traces):
        layout["yaxis" + str(int(suffix or 1) + 100)] = dict(
            title=panel.y2label,
            type=panel.y2scale,
            range=_plotly_range(panel.y2range, panel.y2scale),
            anchor="x" + suffix,
            overlaying="y" + suffix,
            side="right",
        )
    return layout


def render_plotly(spec: FigureSpec):
    """
    Render a FigureSpec as a plotly Figure.

    """

    from plotly import graph_objects as go

    fig = go.Figure(
        layout=dict(
            autosize=False,
            height=spec.height,
            width=spec.width,
            title=spec.title,
            hovermode="closest",
            showlegend=spec.show_legend,
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="black"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    for index, panel in enumerate(spec.panels):
        suffix = "" if index == 0 else str(index + 1)
        for trace in panel.traces:
            fig.add_trace(_plotly_trace(trace, suffix, trace.secondary))
        fig.update_layout(**_plotly_axes_layout(panel, index))
    if spec.scale_toggle:
        fig.update_layout(updatemenus=_plotly_scale_buttons())
    return fig


def _color_limits(z, zmin, zmax):
    """
    Return colour limits that matplotlib accepts: non-finite limits fall back
    to the finite data range, and reversed limits are put in order.

    """

    finite = z[np.isfinite(z)]
    lower = zmin if zmin is not None and np.isfinite(zmin) else None
    upper = zmax if zmax is not None and np.isfinite(zmax) else None
    if lower is None and finite.size:
        lower = float(finite.min())
    if upper is None and finite.size:
        upper = float(finite.max())
    if lower is not None and upper is not None and lower > upper:
        lower, upper = upper, lower
    return lower, upper


def _matplotlib_draw_trace(axes, trace: Trace):
    """
    Draw one Trace on matplotlib axes and return the artist used.

    """

    if trace.kind == "heatmap":
        z = _to_array(trace.z)
        x = np.arange(z.shape[1]) if trace.x is None else _to_array(trace.x)
        y = np.arange(z.shape[0]) if trace.y is None else _to_array(trace.y)
        vmin, vmax = _color_limits(z, trace.zmin, trace.zmax)
        artist = axes.pcolormesh(
            x,
            y,
            np.ma.masked_invalid(z),
            cmap=trace.colorscale.lower(),
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        return artist
    marker = "o" if "markers" in trace.kind else None
    linestyle = "none" if trace.kind == "markers" else "-"
    if trace.dash is not None and linestyle != "none":
        linestyle = "--"
    x_data = [] if trace.x is None else _to_array(trace.x)
    y_data = [] if trace.y is None else _to_array(trace.y)
    n_points = min(len(x_data), len(y_data))
    x_data, y_data = x_data[:n_points], y_data[:n_points]
    lines = axes.plot(
        x_data,
        y_data,
        color=trace.color,
        linewidth=trace.width / 2.0,
        linestyle=linestyle,
        marker=marker,
        markersize=trace.marker_size / 2.0,
        alpha=trace.opacity,
        label=trace.name if trace.show_legend is not False else None,
    )
    return lines[0]


def _matplotlib_style_axes(axes, panel: Panel, secondary: bool):
    """
    Apply labels, scales and ranges of a Panel to its (secondary) axes.

    """

    if secondary:
        axes.set_ylabel(panel.y2label)
        axes.set_yscale(panel.y2scale)
        y2range = _clean_range(panel.y2range, panel.y2scale)
        if y2range is not None:
            axes.set_ylim(y2range)
        return
    axes.set_title(panel.title)
    axes.set_xlabel(panel.xlabel)
    axes.set_ylabel(panel.ylabel)
    axes.set_xscale(panel.xscale)
    axes.set_yscale(panel.yscale)
    xrange = _clean_range(panel.xrange, panel.xscale)
    yrange = _clean_range(panel.yrange, panel.yscale)
    if xrange is not None:
        axes.set_xlim(xrange)
    if yrange is not None:
        axes.set_ylim(yrange)
    if panel.show_grid:
        axes.grid(True, alpha=0.3)
    else:
        axes.grid(False)


def _matplotlib_draw_panel(fig, panel: Panel, show_legend: bool):
    """
    Create the axes of a Panel in the figure and draw all of its traces.

    """

    x0, x1 = panel.x_domain
    y0, y1 = panel.y_domain
    # a gap is left above stacked panels so that tick labels stay visible
    gap = 0.06 if y0 > 0 else 0.0
    rect = [
        0.08 + 0.8 * x0,
        0.1 + 0.8 * y0 + gap,
        0.8 * (x1 - x0),
        0.8 * (y1 - y0) - gap,
    ]
    axes = fig.add_axes(rect)
    axes2 = None
    handles = []
    for trace in panel.traces:
        if trace.secondary and axes2 is None:
            axes2 = axes.twinx()
        target = axes2 if trace.secondary else axes
        artist = _matplotlib_draw_trace(target, trace)
        if trace.kind == "heatmap":
            # fixed colorbar axes keep stacked panels aligned
            colorbar = fig.colorbar(
                artist,
                cax=fig.add_axes(
                    [rect[0] + rect[2] + 0.015, rect[1], 0.015, rect[3]]
                ),
            )
            colorbar.set_label(trace.colorbar_title or "")
        elif trace.name is not None and trace.show_legend is not False:
            handles.append(artist)
    _matplotlib_style_axes(axes, panel, secondary=False)
    if axes2 is not None:
        _matplotlib_style_axes(axes2, panel, secondary=True)
    if show_legend and handles:
        axes.legend(handles=handles, loc="upper left", frameon=False)
    return axes


def render_matplotlib(spec: FigureSpec):
    """
    Render a FigureSpec as a matplotlib Figure (never attached to pyplot
    state beyond creation, so it can be closed safely).

    """

    import matplotlib.pyplot as plt

    fig = plt.figure(
        figsize=(spec.width / 100.0, spec.height / 100.0), dpi=100
    )
    for panel in spec.panels:
        _matplotlib_draw_panel(fig, panel, spec.show_legend)
    if spec.title is not None:
        fig.suptitle(spec.title)
    return fig


def build_figure(spec: FigureSpec, plot_library: Optional[str] = None):
    """
    Render a FigureSpec with the requested (or global) plotting library.

    Parameters
    ----------
    spec: FigureSpec
        The library-independent figure description.
    plot_library: str or None (Default: None)
        'matplotlib', 'plotly' or None to use the global selection.

    Returns
    -------
    A matplotlib Figure or a plotly Figure.

    """

    library = resolve_plot_library(plot_library)
    if library == "plotly":
        return render_plotly(spec)
    return render_matplotlib(spec)


def _split_figure_types(fig_type: str):
    """
    Split a 'png+iframe' style string into a list of lower-case types.

    """

    if fig_type is None:
        return []
    return [_t.strip().lower() for _t in str(fig_type).split("+") if _t]


def _save_plotly(fig, filename: str, fig_type: str, open_iframe: bool):
    """
    Save a plotly figure as html ('iframe') and/or static images.

    """

    from plotly import io as pio

    for _t in _split_figure_types(fig_type):
        if _t == "iframe":
            pio.write_html(fig, filename + "." + _t, auto_open=open_iframe)
        elif _t in STATIC_FIGURE_TYPES:
            pio.write_image(fig, filename + "." + _t)


def _save_matplotlib(fig, filename: str, fig_type: str):
    """
    Save a matplotlib figure as static images; 'iframe' is plotly only.

    """

    for _t in _split_figure_types(fig_type):
        if _t == "iframe":
            logger.warning(
                "The 'iframe' output is only available with plotly, it is "
                "skipped for matplotlib figures."
            )
        elif _t in STATIC_FIGURE_TYPES:
            fig.savefig(filename + "." + _t)


def finalise_figure(
    fig,
    display: bool = True,
    renderer: str = "default",
    save_fig: bool = False,
    fig_type: str = "iframe+png",
    filename: Optional[str] = None,
    default_filename: str = "figure",
    open_iframe: bool = False,
    return_jsonstring: bool = False,
    return_figure: bool = False,
):
    """
    Save, display and return a figure produced by :func:`build_figure`.

    Parameters
    ----------
    fig: matplotlib Figure or plotly Figure
        The figure to finalise; the library is detected from the object.
    display: bool (Default: True)
        Show the figure. matplotlib uses pyplot.show(); plotly uses the
        given renderer.
    renderer: str (Default: 'default')
        plotly renderer, ignored by matplotlib.
    save_fig: bool (Default: False)
        Save the figure in each of the formats in ``fig_type``.
    fig_type: str (Default: 'iframe+png')
        '+' delimited from jpg, png, svg, pdf and iframe (plotly only).
    filename: str or None (Default: None)
        Output name without extension, ``default_filename`` if None.
    open_iframe: bool (Default: False)
        Open the plotly html output in the browser.
    return_jsonstring: bool (Default: False)
        plotly only, return the figure as a json string.
    return_figure: bool (Default: False)
        Return the figure object (matplotlib or plotly).

    Returns
    -------
    The json string, the figure object, or None.

    """

    is_plotly = type(fig).__module__.startswith("plotly")
    if filename is None:
        filename = default_filename

    if save_fig and is_plotly:
        _save_plotly(fig, filename, fig_type, open_iframe)
    elif save_fig:
        _save_matplotlib(fig, filename, fig_type)

    if display and is_plotly:
        if renderer is None or renderer == "default":
            fig.show()
        else:
            fig.show(renderer)
    elif display:
        import matplotlib.pyplot as plt

        plt.show()

    if return_jsonstring and not is_plotly:
        logger.warning(
            "return_jsonstring is only available with plotly, set "
            "return_figure=True to get the matplotlib figure."
        )
    if return_jsonstring and is_plotly:
        return fig.to_json()
    if return_figure:
        return fig
    if not is_plotly:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return None
