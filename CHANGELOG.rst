Version 0.4.0
-------------

:Date X X 2021

We aim to track and report as many changes as possible, but this is not an exhaustive list of all the changes.

* New Features:
  
    * Added `onedspec.remove_atlas_lines_range()` and `wavelength_calibration.remove_atlas_lines_range()`.
    * Added the parameter `top_n_peaks` to `onedspec.find_arc_lines()` and `wavelength_calibration.find_arc_lines()`.
    * `readnoise`, `gain`, `seeing`, `exptime`, and `airmass` can be provided after initialisation.
    * Bad mask can be added or created.
    * arc frame added to `ImageReduction()` will propagate to `TwoDSpec()`.
    * Sensitivity curve is computed after applying a `lowess()` fit for continuum subtraction to remove random noise.
    * Residual image is generated along with the spectral extraction.
    * Use `lowess()` fit for ap_extract profile allowing optimal extraction of extended source.
    * All image output supoprts iframe, jpg, png, svg and pdf.
    * ImageReduction is initialised to configure the logger only.
    * ImageReduction frame location can be added with add_filelist(), and then added by executing load_data().
    * ImageReduction frames (in type of CCDData or ndarray) can be added with add_light(), add_arc(), add_flat(), add_dark(), add_bias()
    * ImageReduction properties seeting can now be set with set_properties(), set_light_properties(), set_dark_properties(), set_flat_properties(), set_bias_properties(), set_arc_properties(), set_cosmic_properties(), set_detector_properties()


* New Experimental Features:

    * Added `TwoDSpec.compute_rectification()` and `TwoDSpec.apply_rectification()` to correct the curvature of the frames.
    * Added `OneDSpec.apply_atmospheric_extinction_correction()` to remove atmospheric reddening.
    * Added `OneDSpec.apply_telluric_correction()` to correct for the Telluric absorptions.

* Dropped Features (see also API changes below):

    * `SAXIS_KEYWORD` is no longer in use.

* Major bug fixes:

    * Loggers are propagated properly upon object creations.

* (API) changes:

    * `onedspec.load_user_atlas()` is changed to `onedspec.add_user_atlas()`.
    * `wavelength_calibration.load_user_atlas()` is changed to `wavelength_calibration.add_user_atlas()`.
    * `onedspec.find_arc_lines()` and `wavelength_calibration.find_arc_lines()` are using the percentage of the (maximum - minimum count) in the arc spectrum (before continuum subtraction) for the `prominence`, whereas `percentile` is the count level threshold AFTER the arc_spec is subtracted by the minimum value of the arc spectrum.
    * `twodspec.set_properties()` is defaulted to NOT set `airmass`, `gain`, `readnoise`, `seeing`, and `exptime`.
    * `save_iframe()` in various functions is no longer in use, it is merged into `save_fig()`.
    * `display` argument is merged into the `renderer` argument.
    * `OneDSpec.refine_fit()` and `wavelength_calibration.refine_fit()` are changed to `robust_refit()`.
    * Arc frame has to be MANUALLY flipped or transposed if it is being added AFTER `TwoDSpec.set_properties()`. If arc frame will be flipped and transposed AUTOMATICALLY if it is added BEFORE `TwoDSpec.set_properties()`.
    * ImageReduction.add_filelist() no longer accepts properties.
    * ImageReduction properties has to be added with set_properties().
    * Individual properties can be added one by one without affecting other existing properties.
    * twodspec.apply_twodspec_mask_to_arc() is changed to twodspec.apply_mask_to_arc().

* See also the changelogs in `RASCAL v0.3.0 <https://github.com/jveitchmichaelis/rascal/blob/main/CHANGELOG.rst>`__.
