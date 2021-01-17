.. _quickstart:

Quickstart with LT/SPRAT
========================

Using the SPRAT instrument on the Liverpool Telescope as an example quickstart.
We are only showing one science spectrum until the end where all the reduced
science and standard spectra are shown.

0.  Import all the required libraries:

    .. code-block:: python

      import sys
      import numpy as np
      from astropy.io import fits
      from aspired import image_reduction
      from aspired import spectral_reduction
      import plotly.io as pio

      # If using jupyter notebook
      pio.renderers.default = 'notebook'

      # If you want to display it in your default browser
      pio.renderers.default = 'browser'

      # If you want the image
      pio.renderers.default = 'png'

1.  In order to perform image reduction, users have to provide a file list of
    the spectra to be reduced. Alternatively, a fits.hdu.image.PrimaryHDU
    object can be supplied For the science spectral image, the file list is
    contained in `examples/sprat_LHS6328.list`

    .. literalinclude:: ../_static/sprat_LHS6328.list

    To reduce the image with the built-in reduction method ImageReduction,
    execute the following, the rederer options are those of `plotly's
    <https://plotly.com/python/renderers/#setting-the-default-renderer>`_:

    .. code-block:: python

      science_frame = image_reduction.ImageReduction('examples/sprat_LHS6328.list')
      science_frame.reduce()
      science_frame.inspect()

    .. raw:: html
      :file: ../_static/1_science_reduced.html

    and for the standard spectral image, the file list is contained in
    `examples/sprat_Hiltner102.list`

    .. literalinclude:: ../_static/sprat_Hiltner102.list

    Similar to the science frame, execute the following:

    .. code-block:: python

      standard_frame = image_reduction.ImageReduction('examples/sprat_Hiltner102.list')
      standard_frame.reduce()
      standard_frame.inspect()

2.  With the image reduced, we can start performing spectral reduction,
    starting from the 2D spectrum:

    .. code-block:: python

      science2D = spectral_reduction.TwoDSpec(science_frame)
      standard2D = spectral_reduction.TwoDSpec(standard_frame)

3.  To trace the respective brightest spectrum in the science and standard
    frames, run

    .. code-block:: python

      science2D.ap_trace()
      standard2D.ap_trace()

    .. raw:: html
      :file: ../_static/3_science_traced.html

4.  And then extract the spectra from the traces by using the ap_extract()
    method. The science spectrum is optimally extracted with an aperture with
    the default size of 7 pixel on each side of the trace, the sky is measured
    by fitting a, by default, first order polynomial to the sky region of
    5 pixels on each side from the aperture by default. The aperture and the
    sky regions are separated by 3 pixels by default. After the extraction,
    display the results with the default renderer (plotly graph in a browser).

    .. code-block:: python

      science2D.ap_extract()
      standard2D.ap_extract()

    The two spectra from the science frame:

    .. raw:: html
      :file: ../_static/5_science_extracted_1.html

    and the spectrum of the standard frame:

5.  Add the 2D arc. The `extract_arc_spec()` automatrically apply the traces
    found above to extract the spectra of the arcs.

    .. code-block:: python

      science2D.add_arc(science_frame)
      science2D.extract_arc_spec()

      standard2D.add_arc(standard_frame)
      standard2D.extract_arc_spec()

6.  Initialise the OneDSpec for wavelength and flux calibration; get the traces
    and the extracted spectra from the TwoDSpec objects,

    .. code-block:: python

      onedspec = spectral_reduction.OneDSpec()
      onedspec.from_twodspec(science2D, stype='science')
      onedspec.from_twodspec(standard2D, stype='standard')

7.  Identify the arclines from the extracted spectrum of the arc.

    .. code-block:: python

      onedspec.find_arc_lines()

    Then, the position of the peaks of the arc lines, can be found for
    performing wavelength calibration for each trace.

    .. raw:: html
      :file: ../_static/9_science_arc_lines.html

8.  Initialise a calibrator and add element lines to prepare for wavelength
    calibration, set the various calibrator, Hough transform and RANSAC
    properties before performing the Hough Transform that is used for the
    automated wavelength calibration. And finally fit for the solution and
    apply to the spectra.

    .. code-block:: python

      onedspec.initialise_calibrator()
      onedspec.add_atlas(
          ['Xe'],
          min_atlas_wavelength=3500.,
          max_atlas_wavelength=8500.)
      onedspec.set_hough_properties()
      onedspec.set_ransac_properties()
      onedspec.do_hough_transform()
      onedspec.fit()
      onedspec.apply_wavelength_calibration()

9.  Next step is the perform the flux calibration, which requires comparing the
    spectrum of the standard to the literature values. To do this, first we need
    to load the literature template from the built-in library, which contains
    all the iraf and ESO standards.

    .. code-block:: python

      onedspec.load_standard(target='hiltner102')
      onedspec.inspect_standard()

    .. raw:: html
      :file: ../_static/11_literature_standard.html

    .. code-block:: python

      onedspec.compute_sensitivity()
      onedspec.inspect_sensitivity()

    .. raw:: html
      :file: ../_static/12_sensitivity_curve.html

10.  Apply the fluxcalibration and inspect the reduced spectra.

    .. code-block:: python

      onedspec.apply_flux_calibration(
      onedspec.inspect_reduced_spectrum()

    The two science spectra:

    .. raw:: html
      :file: ../_static/13_science_spectrum_0.html

    .. raw:: html
      :file: ../_static/13_science_spectrum_1.html

    and the standard spectrum:

    .. raw:: html
      :file: ../_static/14_standard_spectrum.html
