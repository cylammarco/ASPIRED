We aim to track and report as many changes as possible, but this is not an exhaustive list of all the changes.

Version 0.5.0-dev
-----------------

:Date X-X-2022

* New Freatures:

  * Atmospheric extinction corrected spectra are stored separately and has become a separate output option.
  * Telluric corrected spectra are stored separately and has become a separate output option.
  * Atmospheric extinction AND telluric corrected spectra are stored separately and has become a separate output option.
  * Atmospheric extinction correction factor and telluric absorption correction can be exported as FITS and CSV.
  * Resampling is now performed immediately before creating and saving files. All the calibrations are now done in the native pixel/wavelength sampling.
  * Added `TwoDSpec.inspect_line_spread_function()` to view the line spread function used for extraction.
  * Using Horne86+Gaussian for optimal extraction now includes aperture correction.
  * Added recompute_continuum when generating sensitivity curve.

* Major bug fixes:

  * `ap_trace` trace suffers from a rounding-like error which is now fixed
  * `ap_trace` sub-sprectra cleaning was referencing pre-upsampled pixel location instead of the upsampled pixel location
  * Occasional wrong flux calibration is fixed as the sensitivity function switched to the unit of flux/mag per electron

* Minor bug fixes:

  * Fixed a bug that caused some standard star names are never matched
  * Removed some unused package import in ondespec.py
  * `ImageReduction()` is checking if `verbose` is supplied as a boolean at initialisation.
  * Fixed a few typos when exporting FITS files with `spectrum1D`.
  * Fixed typo relating to logging of the setting of airmass.
  * Fixed a bug in sorting peaks by prominence
  * Exporting FITS and CSV will check if data exist to avoid outputting empty files.
  * Header information should be grabbed when data are manually provided.

* (API) changes:

  * get_telluric_correction() to get_telluric_strength() as the function only get the multiplication factor to the template telluric absorption profile.
  * `lowess_frac` is used throughout the woftware to denote the `frac` argument for the `lowess()` function.
  * spectrum1D is now spectrumOneD to avoid confusion with Astropy Specutil spectrum1D

* Other changes:

  * Small change in the Horne86 optimal extraction function, it should have become more efficient.
  * Improved stability in computing the spectral width when using Horne86+Gaussian. 
  * `count`, `count_err`, `count_sky` are strictly in unit of electron count.
  * Sensitivity function in FluxCalibration remains in the unit of flux/mag per electron, however, because `count` is no longer in the unit of electron per second, the exposure time is needed when applying the sensitivity function.
  * `get_continuum` is now checking input array/list size and throw error if they don't match.
  * OneDSpec and spectrum1D expanded to accommodate wavelength_resampled
  * spectrum1D exapanded to handle atmospheric extinction corrected spectrum as an independent data list

Version 0.4.X
-------------

* Major bug fixes

  * Fixed a rounding-like error wwhen finding trace.
  * Fixed a bug leading to wrong background estimation when finding trace.

Version 0.4.6
-------------

:Date 26-July-2022

* Minor bug fixes

  * Fixed a bug leading to failing in re-creating FITS file in the background when it should.

Version 0.4.5
-------------

:Date 18-July-2022

* Minor bug fixes

  * Fixed typo relating to logging of the setting of airmass.
  * Fixed a bug in sorting peaks by prominence

Version 0.4.4
-------------

:Date 3-Apr-2022

* Minor bug fixes:

  * Fixed typos in the image_reduction which led to the failure in bias subraction and flat division (#79).
  * Fixed a few condition handling errors when manually adding frames to an image_reduction object one by one.

Version 0.4.3
-------------

:Date 25-Mar-2022

* Minor bug fixes:

  * FITS header LABEL and EXTNAME are now unique and identical (#77).

Version 0.4.2
-------------

:Date 23-Mar-2022

* Major bug fixes:

  * Telluric correction can be applied to both science and standard spectra in OneDSpec.
  * Telluric profile is copied from a FluxCalibration instance to OneDSpec.

* Minor bug fixes:

  * When spectrum1D failed to save file, a warning is displayed.
  * Standard star names are all compared in lower-case strings.

* Dependency change

  * Plotly dependency changed from orca to kaleido.
  * Astroscrappy >= 1.0.8
  * Rascal >= 0.3.2
  * Astropy >=4.3

* Other changes

  * Installation is now configured with setup.cfg
  * Adopted black style
  * Using pre-commit
  * Fixed coverall report submission issue
  * Using unittest.mock.patch such that image display in tests do not block process

Version 0.4.1
-------------

:Date 6-Nov-2021

* New Features:

    * All image output supoprts iframe, jpg, png, svg and pdf.
    * ImageReduction is initialised to configure the logger only.
    * ImageReduction frame location can be added with add_filelist(), and then added by executing load_data().
    * ImageReduction frames (in type of CCDData or ndarray) can be added with add_light(), add_arc(), add_flat(), add_dark(), add_bias()
    * ImageReduction properties seeting can now be set with set_properties(), set_light_properties(), set_dark_properties(), set_flat_properties(), set_bias_properties(), set_arc_properties(), set_cosmic_properties(), set_detector_properties()
    * Bad mask can be added or created.
    * arc frame added to `ImageReduction()` will propagate to `TwoDSpec()`.
    * `readnoise`, `gain`, `seeing`, `exptime`, and `airmass` can be provided after initialisation.
    * Residual image is generated along with the spectral extraction.
    * Use `lowess()` fit for ap_extract profile allowing optimal extraction of extended source.
    * Added 'TwoDSpec.inspect_residual()
    * Added `OneDSpec.remove_atlas_lines_range()` and `wavelength_calibration.remove_atlas_lines_range()`.
    * Added the parameter `top_n_peaks` to `OneDSpec.find_arc_lines()` and `wavelength_calibration.find_arc_lines()`.
    * Sensitivity curve is computed after applying a `lowess()` fit for continuum subtraction to remove random noise.

* New Experimental Features:

    * Added `TwoDSpec.get_rectification()` and `TwoDSpec.apply_rectification()` to correct the curvature of the frames.
    * Added `OneDSpec.set_atmospheric_extinction()` to choose or provide an atmospheric reddening law.
    * Added `OneDSpec.apply_atmospheric_extinction_correction()` to remove atmospheric reddening.
    * Added `OneDSpec.get_telluric_profile()` to compute Telluric absorption profile.
    * Added `OneDSpec.inspect_telluric_profile()` to display the Telluric absorption profile and how the correction would look like.
    * Added `OneDSpec.apply_telluric_correction()` to apply the Telluric absorptions and modify the state of the flux.

* Dropped Features (see also API changes below):

    * `SAXIS_KEYWORD` is no longer in use.

* Major bug fixes:

    * Loggers are propagated between objects upon initialisations.
    * Sky modelling is sigma-clipping outliers and bad values.
    * ap_trace() is masking out the faint parts of the spectrum when fitting a polynomial to the trace.
    * Jansky conversion was wrong when using the ING standards

* (API) changes:

    * All loggers are now displaying `INFO` level of logs and by default it is print to screen only.
    * ImageReduction.add_filelist() no longer accepts properties.
    * ImageReduction properties has to be added with set_properties().
    * In ImageReduction, individual properties can be added one by one without affecting other existing properties.
    * Arc frame has to be MANUALLY flipped or transposed if it is being added AFTER `TwoDSpec.set_properties()`. If arc frame will be flipped and transposed AUTOMATICALLY if it is added BEFORE `TwoDSpec.set_properties()`.
    * `TwoDSpec.apply_twodspec_mask_to_arc()` is changed to TwoDSpec.apply_mask_to_arc().
    * `TwoDSpec.ap_extract()` is now sigma clipping outliers when modelling the sky.
    * `TwoDSpec.ap_trace()` argument ap_faint is now defined by the percentage of the faintest subspectra.
    * `wavelength_calibration.load_user_atlas()` is changed to `wavelength_calibration.add_user_atlas()`.
    * `OneDSpec.refine_fit()` and `wavelength_calibration.refine_fit()` are changed to `robust_refit()`.
    * `OneDSpec.load_user_atlas()` is changed to `OneDSpec.add_user_atlas()`.
    * `OneDSpec.find_arc_lines()` and `wavelength_calibration.find_arc_lines()` are using the percentage of the (maximum - minimum count) in the arc spectrum (before continuum subtraction) for the `prominence`, whereas `percentile` is the count level threshold AFTER the arc_spec is subtracted by the minimum value of the arc spectrum.
    * `OneDSpec.compute_sensitivity()` is changed to `OneDSpec.get_sensitivity()`.
    * `TwoDSpec.set_properties()` is defaulted to NOT set `airmass`, `gain`, `readnoise`, `seeing`, and `exptime`.
    * `save_iframe()` in various functions is no longer in use, it is merged into `save_fig()`.
    * `display` argument is merged into the `renderer` argument.

* See also the changelogs in `RASCAL v0.3.0 <https://github.com/jveitchmichaelis/rascal/blob/main/CHANGELOG.rst>`__.
