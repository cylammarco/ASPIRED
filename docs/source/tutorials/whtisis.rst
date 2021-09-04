.. _whtisis:

WHT/ISIS
========

In this example, we reduced a faint low-resolution spectrum of an ultracool white dwarf taken with the Intermediate-dispersion Spectrograph and Imaging System (ISIS) on the William Herschel Telescope (WHT).

0.  Import all the required libraries:

    .. code-block:: python

      import sys
      import numpy as np
      from astropy.io import fits
      from aspired import image_reduction
      from aspired import spectral_reduction

1.  In order to perform image reduction, users have to provide a file list of
    the spectra to be reduced. Alternatively, a fits.hdu.image.PrimaryHDU
    object can be supplied For the science spectral image, the file list is
    contained in `examples/sprat_LHS6328.list`

    .. literalinclude:: ../_static/sprat_LHS6328.list

    To reduce the image with the built-in reduction method ImageReduction,
    execute the following, the rederer options are those of `plotly's
    <https://plotly.com/python/renderers/#setting-the-default-renderer>`_. In
    all the following lines, we are saving the plotly figures as iframes that
    can be viewed and interacted by using a browser:

    .. code-block:: python

      # Set the dispersion direction
      Saxis = 0

      science_frame = image_reduction.ImageReduction()
      science_frame.add_filelist('isis_pso1801p6254.list')
      science_frame.set_properties(saxis=Saxis)
      science_frame.reduce()
      science_frame.inspect(
          filename='reduced_image_pso1801p6254',
          save_iframe=True)

    and for the standard spectral image, the file list is contained in
    `isis_g93m48.list`

    .. literalinclude:: ../_static/sprat_Hiltner102.list

    Similar to the science frame, execute the following:

    .. code-block:: python

      standard_frame = image_reduction.ImageReduction()
      standard_frame.add_filelist('isis_g93m48.list')
      standard_frame.set_properties(saxis=Saxis)
      standard_frame.reduce()
      standard_frame.inspect()

2.  With the image reduced, we can start performing spectral reduction,
    starting from the 2D spectrum with the customised setting to provide
    the appropriate read noise, gain, seeing and spatial masking:

    .. code-block:: python

      # spec mask
      spatial_mask = np.arange(450, 650)

      # initialise the two spectral_reduction.TwoDSpec()
      pso = spectral_reduction.TwoDSpec(
          science_frame,
          spatial_mask=spatial_mask,
          readnoise=4.5,
          cosmicray=False,
          gain=0.98,
          seeing=1.1,
          silence=True)

      g93 = spectral_reduction.TwoDSpec(
          standard_frame,
          spatial_mask=spatial_mask,
          readnoise=4.5,
          cosmicray=False,
          gain=0.98,
          seeing=1.1,
          silence=True)

3.  To trace the respective brightest spectrum in the science and standard
    frames, run

    .. code-block:: python

      pso.ap_trace(save_iframe=True, filename='pso_trace')

      g93.ap_trace(save_iframe=True, filename='g93_trace')

    .. raw:: html
      :file: ../_static/isis_pso_trace.html

4.  And then extract the spectra from the traces by using the ap_extract()
    method. The science spectrum is optimally extracted with an aperture with
    the default size of 15 and 20 pixel on each side of the trace, the sky is 
    measured by fitting a, by default, first order polynomial to the sky region of
    5 pixels on each side from the aperture by default. The aperture and the
    sky regions are separated by 3 pixels by default. After the extraction,
    display the results with the default renderer (plotly graph in a browser).

    .. code-block:: python

      # Optimal extracting spectrum by summing over the aperture along the trace
      pso.ap_extract(
          apwidth=15,
          skysep=3,
          skywidth=5,
          optimal=True,
          display=True,
          save_iframe=True,
          filename='pso_extract')

      g93.ap_extract(
          apwidth=20,
          skysep=3,
          skywidth=5,
          optimal=True,
          display=True,
          save_iframe=True,
          filename='g93_extract')

    The two spectra from the science frame:

    .. raw:: html
      :file: ../_static/isis_pso_extract_0.html

    and the spectrum of the standard frame:

    .. raw:: html
      :file: ../_static/isis_g93_extract_0.html

5.  Add the 2D arc and apply the masks in both the dispersion and spatial
    directions to the image before extracting the spectra of the
    arcs (experimental, as of 17 Jan 2021). The arcs have to be rotated
    manually if the dispersion direction is along the y-axis. Future updates
    will handle the `saxis` automatically.

    .. code-block:: python

      pso.extract_arc_spec(
          display=True,
          save_iframe=True,
          filename='science_arc_spec')

      g93.extract_arc_spec(
          display=True,
          save_iframe=True,
          filename='standard_arc_spec')

6.  Initialise the OneDSpec for wavelength and flux calibration; copy the
    relavent data from the TwoDSpec objects and find the arc lines

    .. code-block:: python

      pso_reduced = spectral_reduction.OneDSpec()
      pso_reduced.from_twodspec(pso, stype='science')
      pso_reduced.from_twodspec(g93, stype='standard')

      pso_reduced.find_arc_lines(
          display=True,
          stype='science+standard',
          save_iframe=True,
          filename='arc_lines')

    Then, the position of the arc line peaks can be found for
    performing wavelength calibration for each trace.

    .. raw:: html
      :file: ../_static/isis_arc_lines_0.html

7.  Initialise a calibrator and add element lines to prepare for wavelength
    calibration, set the various calibrator, Hough transform and RANSAC
    properties before performing the Hough Transform that is used for the
    automated wavelength calibration. And finally fit for the solution and
    apply to the spectra.

    .. code-block:: python

      pso_reduced.initialise_calibrator(stype='science+standard')
      pso_reduced.set_hough_properties(
          min_wavelength=7000.,
          max_wavelength=10500.,
          stype='science+standard')
      pso_reduced.add_atlas(
          elements=["Cu", "Ne", 'Ar'],
          stype='science+standard')
      pso_reduced.do_hough_transform()
      pso_reduced.fit(max_tries=2000, stype='science+standard')
      pso_reduced.apply_wavelength_calibration(stype='science+standard')

8.  Next step is the perform the flux calibration, which requires comparing the
    spectrum of the standard to the literature values. To do this, first we need
    to load the literature template from the built-in library, which contains
    all the iraf and ESO standards.

    .. code-block:: python

      pso_reduced.load_standard(
          target='g93_48',
          library='esohststan',
          cutoff=0.4)
      pso_reduced.inspect_standard(
          save_iframe=True,
          filename='literature_standard')

    .. raw:: html
      :file: ../_static/isis_literature_standard.html

    .. code-block:: python

      pso_reduced.get_sensitivity(kind='cubic')
      pso_reduced.inspect_sensitivity(
          save_iframe=True
          filename='sensitivity')

    .. raw:: html
      :file: ../_static/isis_sensitivity.html

9.  Apply the fluxcalibration and inspect the reduced spectra.

    .. code-block:: python

      pso_reduced.apply_flux_calibration(
      pso_reduced.inspect_reduced_spectrum(
          wave_min=7000.,
          wave_max=10500.,
          stype='science',
          save_iframe=True,
          filename='pso_reduced_spectrum')
      pso_reduced.inspect_reduced_spectrum(
          wave_min=7000.,
          wave_max=10500.,
          stype='standard',
          save_iframe=True,
          filename='g93_reduced_spectrum')

    The two science spectra:

    .. raw:: html
      :file: ../_static/isis_pso_reduced_spectrum_0.html

    and the standard spectrum:

    .. raw:: html
      :file: ../_static/isis_g93_reduced_spectrum.html
