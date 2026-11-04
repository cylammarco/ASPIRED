Plotting
========

All diagnostic figures (``inspect_*`` methods and the ``inspect=True``
options) can be drawn with matplotlib or plotly. matplotlib is the default.

Choosing the library
--------------------

Globally:

.. code-block:: python

    import aspired
    aspired.set_plot_library("plotly")   # or "matplotlib"

For a single call, which takes precedence over the global setting:

.. code-block:: python

    onedspec.inspect_reduced_spectrum(plot_library="plotly")

Options that depend on the library
----------------------------------

* ``renderer``, ``return_jsonstring``, the ``iframe`` entry of ``fig_type``
  (html output), ``open_iframe`` and the log/linear scale buttons only exist
  in plotly. matplotlib ignores them (with a warning for ``iframe`` and
  ``return_jsonstring``) and draws the scale set in the figure.
* Static images (``jpg``, ``png``, ``svg``, ``pdf``) are written by
  ``savefig`` for matplotlib and by kaleido for plotly.
* ``return_figure=True`` returns the matplotlib or plotly figure object for
  further customisation. Without it, matplotlib figures are closed after being
  shown or saved.
* Methods that draw one figure per spectrum return a list.
