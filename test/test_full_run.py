import os
import numpy as np
from aspired import image_reduction
from aspired import spectral_reduction

base_dir = os.path.dirname(__file__)
abs_dir = os.path.abspath(os.path.join(base_dir, '..'))


def test_full_run():
    # Extract two spectra

    # Line list
    atlas = [
        4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
        4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
        6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32,
        6976.18, 7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65,
        7887.40, 7967.34, 8057.258
    ]
    element = ['Xe'] * len(atlas)

    spatial_mask = np.arange(70, 200)
    spec_mask = np.arange(50, 1024)

    # Science frame
    lhs6328_frame = image_reduction.ImageReduction(
        'test/test_data/sprat_LHS6328.list',
        log_level='DEBUG',
        log_file_folder='test/test_output/')
    lhs6328_frame.reduce()
    lhs6328_frame.save_fits('test/test_output/test_full_run_standard_image',
                            overwrite=True)

    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        log_level='DEBUG',
        log_file_folder='test/test_output/')

    lhs6328_twodspec.ap_trace(nspec=2, display=False)

    # Optimal extraction to get the LSF for force extraction below
    lhs6328_twodspec.ap_extract(apwidth=15,
                                skywidth=10,
                                skydeg=1,
                                optimal=True,
                                display=False,
                                save_iframe=False)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        forced=True,
        variances=lhs6328_twodspec.spectrum_list[1].var,
        display=False,
        save_iframe=False)

    # Aperture extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=5,
        skydeg=1,
        optimal=False,
        display=False,
        save_iframe=True,
        filename='test/test_output/test_full_run_extract')

    # Optimal extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=5,
        skydeg=1,
        optimal=True,
        forced=True,
        variances=1000000.,
        display=False,
        save_iframe=True,
        filename='test/test_output/test_full_run_extract')

    lhs6328_twodspec.save_fits(
        filename='test/test_output/test_full_run_twospec', overwrite=True)

    # Standard frame
    standard_frame = image_reduction.ImageReduction(
        'test/test_data/sprat_Hiltner102.list',
        log_level='DEBUG',
        log_file_folder='test/test_output/')
    standard_frame.reduce()
    standard_frame.save_fits('test/test_output/test_full_run_standard_image',
                             overwrite=True)

    hilt102_twodspec = spectral_reduction.TwoDSpec(
        standard_frame,
        cosmicray=True,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        readnoise=5.7,
        log_level='DEBUG',
        log_file_folder='test/test_output/')

    hilt102_twodspec.ap_trace(nspec=1, resample_factor=10, display=False)

    hilt102_twodspec.ap_extract(apwidth=15,
                                skysep=3,
                                skywidth=5,
                                skydeg=1,
                                optimal=True,
                                display=False,
                                save_iframe=False)

    # Add a 2D arc image
    lhs6328_twodspec.add_arc(lhs6328_frame)
    hilt102_twodspec.add_arc(standard_frame)

    lhs6328_twodspec.apply_twodspec_mask_to_arc()
    hilt102_twodspec.apply_twodspec_mask_to_arc()

    # Extract the 1D arc by aperture sum of the traces provided
    lhs6328_twodspec.extract_arc_spec(display=False)
    hilt102_twodspec.extract_arc_spec(display=False)

    # Handle 1D Science spectrum
    lhs6328_onedspec = spectral_reduction.OneDSpec(
        log_level='DEBUG',
        log_file_folder='test/test_output/')
    lhs6328_onedspec.from_twodspec(lhs6328_twodspec, stype='science')
    lhs6328_onedspec.from_twodspec(hilt102_twodspec, stype='standard')

    # Create the trace and count as FITS BEFORE flux and wavelength calibration
    # This is an uncommon operation, but it should work.
    lhs6328_onedspec.create_fits(output='trace+count',
                                 stype='science+standard')

    # Save the trace and count as FITS BEFORE flux and wavelength calibration
    # This is an uncommon operation, but it should work.
    lhs6328_onedspec.save_fits(output='trace+count',
                               filename='test/test_output/test_full_run',
                               stype='science+standard',
                               overwrite=True)

    # Find the peaks of the arc
    lhs6328_onedspec.find_arc_lines(display=False, stype='science+standard')

    # Configure the wavelength calibrator
    lhs6328_onedspec.initialise_calibrator(stype='science+standard')

    lhs6328_onedspec.set_hough_properties(num_slopes=1000,
                                          xbins=200,
                                          ybins=200,
                                          min_wavelength=3500,
                                          max_wavelength=8500,
                                          stype='science+standard')
    lhs6328_onedspec.set_ransac_properties(filter_close=True,
                                           stype='science+standard')

    lhs6328_onedspec.load_user_atlas(elements=element,
                                     wavelengths=atlas,
                                     stype='science+standard')
    lhs6328_onedspec.do_hough_transform()

    # Solve for the pixel-to-wavelength solution
    lhs6328_onedspec.fit(max_tries=200,
                         stype='science+standard',
                         display=False)

    # Apply the wavelength calibration and display it
    lhs6328_onedspec.apply_wavelength_calibration(stype='science+standard')

    # Get the standard from the library
    lhs6328_onedspec.load_standard(target='hiltner102')

    lhs6328_onedspec.compute_sensitivity(k=3,
                                         method='interpolate',
                                         mask_fit_size=1,
                                         extinction_correction=True)

    lhs6328_onedspec.apply_flux_calibration(stype='science+standard')

    # Apply atmospheric extinction correction
    lhs6328_onedspec.set_atmospheric_extinction(location='orm')
    lhs6328_onedspec.apply_atmospheric_extinction_correction()

    # Create FITS
    lhs6328_onedspec.create_fits(output='trace+count')

    # Modify FITS header for the trace
    lhs6328_onedspec.modify_trace_header(0, 'set', 'COMMENT', 'Hello World!')

    print(lhs6328_onedspec.science_spectrum_list[0].trace_hdulist[0].
          header['COMMENT'])

    # Create more FITS
    lhs6328_onedspec.create_fits(output='trace+count+arc_spec+wavecal',
                                 stype='science+standard',
                                 recreate=False)

    # Check the modified headers are not overwritten
    print(lhs6328_onedspec.science_spectrum_list[0].trace_hdulist[0].
          header['COMMENT'])

    # Save as FITS (and create the ones that were not created earlier)
    lhs6328_onedspec.save_fits(
        output='trace+count+arc_spec+wavecal+wavelength+count_resampled+flux+'
        'flux_resampled',
        filename='test/test_output/test_full_run',
        stype='science+standard',
        recreate=False,
        overwrite=True)

    # Check the modified headers are still not overwritten
    print(lhs6328_onedspec.science_spectrum_list[0].trace_hdulist[0].
          header['COMMENT'])

    # Create more FITS
    lhs6328_onedspec.create_fits(output='trace',
                                 stype='science+standard',
                                 recreate=True)

    # Now the FITS header should be back to default
    print(lhs6328_onedspec.science_spectrum_list[0].trace_hdulist[0].header)

    # save as CSV
    lhs6328_onedspec.save_csv(
        output='trace+count+arc_spec+wavecal+wavelength+count_resampled+flux+'
        'flux_resampled',
        filename='test/test_output/test_full_run',
        stype='science+standard',
        overwrite=True)

    # save figure
    lhs6328_onedspec.inspect_reduced_spectrum(
        filename='test/test_output/test_full_run',
        display=False,
        save_iframe=True)
