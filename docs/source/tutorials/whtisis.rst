.. _whtisis:

WHT ISIS
========

1.

2.

3.  And then extract the spectra from the traces by using the ap_extract()
    method. The science spectrum is optimally extracted with an aperture with a
    size of 15 pixel on each side of the trace, the sky is measured by fitting
    a first order polynomial to the sky region of 10 pixels on each side from
    the aperture with a separation of 3 pixels. After the extraction, display
    the results with the default renderer (plotly graph in a browser).

    .. code-block:: python

      science2D.ap_extract(
          apwidth=15,
          optimal=True,
          skysep=3,
          skywidth=10,
          skydeg=1,
          display=True)

      standard2D.ap_extract(
          apwidth=25,
          optimal=True,
          skysep=3,
          skywidth=5,
          skydeg=1,
          optimal=True,
          display=True)
