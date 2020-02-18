.. _quickstart:

Quickstart
==========

Using the SPRAT instrument on the Liverpool Telescope as an example quickstart.

1. Perform image reduction by providing a file list of the spectra to be reduced:

  for the science spectral image:

  .. literalinclude:: ../../../examples/sprat_LHS6328.list

  .. raw:: html
     :file: ../_static/1_science_reduced.html

  and for the standard spectral image:

  .. literalinclude:: ../../../examples/sprat_Hiltner102.list

  .. raw:: html
     :file: ../_static/2_standard_reduced.html

2. Tracing spectra

  .. raw:: html
     :file: ../_static/3_science_traced.html

  .. raw:: html
     :file: ../_static/4_standard_traced.html

3. Extracting spectra

  .. raw:: html
     :file: ../_static/5_science_extracted.html

  .. raw:: html
     :file: ../_static/6_standard_extracted.html

4. Getting the Standard template

  .. raw:: html
     :file: ../_static/7_standard.html

5. Finding arc lines and perform wavelength calibration for each trace

  .. raw:: html
     :file: ../_static/8_science_arc.html

  .. raw:: html
     :file: ../_static/9_standard_arc.html

6. Collect all the calibrations to apply the wavelength calibration and compute & apply the sensitivity curve to all the spectra

  .. raw:: html
     :file: ../_static/10_sensitivity_curve.html

7. Generate the reduced spectra.

  .. raw:: html
     :file: ../_static/11_science_spectrum.html

  .. raw:: html
     :file: ../_static/12_standard_spectrum.html
