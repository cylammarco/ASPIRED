.. _quickstart:

Quickstart with default setting
===============================

Using the SPRAT instrument on the Liverpool Telescope as an example quickstart.

1.  In order to perform image reduction, users have to provide a file list of
    the spectra to be reduced. Alternatively, a fits.hdu.image.PrimaryHDU
    object can be supplied For the science spectral image, the file list is
    contained in `examples/sprat_LHS6328.list`

    .. literalinclude:: ../../../examples/sprat_LHS6328.list

    To reduce the image with the built-in reduction method ImageReduction,
    execute the following, the rederer options are those of `plotly's
    <https://plotly.com/python/renderers/#setting-the-default-renderer>`_:

    .. code-block:: python

      science_frame = aspired.ImageReduction('examples/sprat_LHS6328.list')
      science_frame.reduce()
      science_frame.inspect()

    .. raw:: html
      :file: ../_static/1_science_reduced.html

    and for the standard spectral image, the file list is contained in
    `examples/sprat_Hiltner102.list`

    .. literalinclude:: ../../../examples/sprat_Hiltner102.list

    Similar to the science frame, execute the following:

    .. code-block:: python

      standard_frame = aspired.ImageReduction('examples/sprat_Hiltner102.list')
      standard_frame.reduce()
      standard_frame.inspect()

    .. raw:: html
      :file: ../_static/2_standard_reduced.html

2.  To trace the respective brightest spectrum in the science and standard
    frames, run

    .. code-block:: python

      science2D.ap_trace()
      standard2D.ap_trace()

    .. raw:: html
      :file: ../_static/3_science_traced.html

    .. raw:: html
      :file: ../_static/4_standard_traced.html

3.  And then extract the spectra from the traces by using the ap_extract()
    method. The science spectrum is optimally extracted with an aperture with
    the default size of 7 pixel on each side of the trace, the sky is measured
    by fitting a, by default, first order polynomial to the sky region of
    5 pixels on each side from the aperture by default. The aperture and the
    sky regions are separated by 3 pixels by default. After the extraction,
    display the results with the default renderer (plotly graph in a browser).

    .. code-block:: python

      science2D.ap_extract(display=True)
      standard2D.ap_extract(display=True)

    .. raw:: html
      :file: ../_static/5_science_extracted.html

    .. raw:: html
      :file: ../_static/6_standard_extracted.html

4.  Next step is the perform the flux calibration, which requires comparing the
    spectrum of the standard to the literature values. To do this, first we need
    to load the literature template from the built-in library, which contains
    all the iraf and ESO standards.

    .. code-block:: python

      fluxcal = aspired.StandardFlux(target='hiltner102', group='irafirs')
      fluxcal.load_standard()
      fluxcal.inspect_standard()

    .. raw:: html
      :file: ../_static/7_standard.html

5.  Finding arc lines and perform wavelength calibration for each trace

    .. code-block:: python

      wavecal_science = aspired.WavelengthPolyFit(science2D, science_arc)
      wavecal_science.find_arc_lines(display=True)
      wavecal_science.fit(elements=["Xe"])
      wavecal_science.refine_fit(elements=["Xe"], tolerance=5, display=True)

    .. raw:: html
      :file: ../_static/8_science_arc.html

    .. raw:: html
      :file: ../_static/9_standard_arc.html

6.  Collect all the calibrations to apply the wavelength calibration and then
    compute and apply the sensitivity curve to all the spectra

    .. code-block:: python

      science_reduced = aspired.OneDSpec(
          science2D,
          wavecal_science,
          standard2D,
          wavecal_standard,
          fluxcal)
      science_reduced.apply_wavelength_calibration('science+standard')
      science_reduced.compute_sensitivity(display=True)

  .. raw:: html
    :file: ../_static/10_sensitivity_curve.html

7.  Generate the reduced spectra.

    .. code-block:: python

      science_reduced.inspect_reduced_spectrum('science+standard', display=True)

  .. raw:: html
    :file: ../_static/11_science_spectrum.html

  .. raw:: html
    :file: ../_static/12_standard_spectrum.html
