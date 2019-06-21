import difflib
import numpy as np
from scipy import signal
from scipy import interpolate as itp
try:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    warn(AstropyWarning(
        'matplotlib is not present, diagnostic plots cannot be generated.'
        ))

library_list = ['ctio', 'hst', 'oke', 'wd', 'xshooter']

# https://www.eso.org/sci/observing/tools/standards/spectra/hamuystandards.html
ctio = [
    'cd32d9927',
    'cd_34d241',
    'eg21',
    'eg274',
    'feige110',
    'feige56',
    'hilt600',
    'hr1544',
    'hr3454',
    'hr4468',
    'hr4963',
    'hr5501',
    'hr718',
    'hr7596',
    'hr7950',
    'hr8634',
    'hr9087',
    'ltt1020',
    'ltt1788',
    'ltt2415',
    'ltt3218',
    'ltt3864',
    'ltt4364',
    'ltt4816',
    'ltt6248',
    'ltt7379',
    'ltt745',
    'ltt7987',
    'ltt9239',
    'ltt9491'
    ]

# https://www.eso.org/sci/observing/tools/standards/spectra/hststandards.html
hst = [
    'agk81d226',
    'bd28d4211',
    'bd33d2642',
    'bd75d325',
    'bpm16274',
    'feige110',
    'feige34',
    'g191b2b',
    'g93_48',
    'gd108',
    'gd50',
    'grw70d5824',
    'hd49798',
    'hd60753',
    'hd93521',
    'hr153',
    'hr1996',
    'hr4554',
    'hr5191',
    'hr7001',
    'hz2',
    'hz21',
    'hz4',
    'hz44',
    'lb227',
    'lds749b',
    'ngc7293'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/okestandards_rev.html
oke = [
  'bd25d4655',
  'bd28d4211',
  'bd33d2642',
  'bd75d325',
  'feige110',
  'feige34',
  'feige66',
  'feige67',
  'g138_31',
  'g158_100',
  'g191b2b',
  'g193_74',
  'g24_9',
  'g60_54',
  'gd108',
  'gd248',
  'gd50',
  'grw70d5824',
  'hd93521',
  'hz21',
  'hz4',
  'hz44',
  'ltt9491',
  'ngc7293',
  'sa95_42'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/wdstandards.html
wd = [
  'agk81d226_005',
  'alpha_lyr_004',
  'bd_25d4655_002',
  'bd_28d4211_005',
  'bd_33d2642_004',
  'bd_75d325_005',
  'feige110_005',
  'feige34_005',
  'feige66_002',
  'feige67_002',
  'g93_48_004',
  'gd108_005',
  'gd50_004',
  'gd71',
  'grw_70d5824_005',
  'hd93521_005',
  'hz21_005',
  'hz2_005',
  'hz44_005',
  'hz4_004',
  'lb227_004',
  'lds749b_005',
  'ltt9491_002',
  'ngc7293_005'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/Xshooterspec.html
xshooter = [
  'EG274',
  'Feige110',
  'GD153',
  'GD71',
  'LTT3218',
  'LTT7987'
]


def _lookup_standard(target, source):
    '''
    Check if the requested standard and library exist.

    '''
    
    try:
        target_list = eval(source)
    except:
        print('Requested spectrophotometric library does not exist.')
        print()
        return False     
    if target in target_list:
        return True
    else:
        print('Requested target not in the library.')
        print('')
        print('The requrested spectrophotometric library contains:')
        print(target_list)
        print('')
        best_match = difflib.get_close_matches(target, target_list, cutoff=0.4)
        if len(best_match) > 0:
            print('Are you looking for these: ')
            print(best_match)
        return False


def _read_standard(target, source, ftype, display):
    '''
    Read the standard flux/magnitude file. And return the wavelength and
    flux/mag in units of

    wavelength: \AA
    flux:       ergs / cm / cm / s / \AA
    mag:        mag (AB) 

    '''

    if not _lookup_standard(target, source):
        return None
    
    if ftype == 'flux':
        target_name = 'f' + target + '.dat'
        if source != 'xhooter':
            flux_multiplier = 1e-16
        else:
            flux_multiplier = 1.
    elif ftype == 'mag':
        target_name = 'm' + target + '.dat'
        flux_multiplier = 1.
    else:
        print('The type has to be \'flux\' of \'mag\'.')
    
    f = np.loadtxt('standards/' + str(source) + 'stan/' + target_name)
    
    wave = f[:,0]
    fluxmag = f[:,1] * flux_multiplier
    
    if display:
        plt.figure(figsize=(10,6))
        if ftype == 'flux':
            plt.plot(wave, fluxmag, label='Standard Flux')
        else:
            plt.plot(wave, fluxmag, label='Standard Magnitude')
        plt.title(source + ' : ' + target)
        plt.xlabel(r'Wavelength / $\AA$')
        plt.ylabel(r'Flux / ergs cm$^{-2}$ s$^{-1} \AA^{-1}$')
        plt.ylim(ymin=0)
        plt.grid()
        plt.legend()

    return wave, fluxmag


def list_all():
    '''
    List all the built-in Spectrophotometric Standards

    '''
    
    print('CTIO Spectrophotometric Standards')
    print(ctio)
    print('')
    print('HST Spectrophotometric Standards')
    print(hst)
    print('')
    print('Oke Spectrophotometric Standards')
    print(oke)
    print('')
    print('WD Spectrophotometric Standards')
    print(wd)
    print('')
    print('Xshooter Spectrophotometric Standards')
    print(xshooter)
    print('')


def get_sencurve(wave, adu, target, source, ftype='flux', kind='cubic',
                 smooth=True, slength=11, sorder=2, display=False):
    '''
    Get the standard flux or magnitude of the given target and source
    based on the given array of wavelengths. Some standard libraries
    contain the same target with slightly different values.

    Parameters
    ----------
    wave : 1-d numpy array (N)
        wavelength in the unit of A
    flux : 1-d numpy array (N)
        the ADU of the standard spectrum
    target : string
        the name of the standard star
    source : string
        the name of the standard library
        >>> ['ctio', 'hst', 'oke', 'wd', 'xshooter']
    ftype : string
        data to return 'flux' or 'mag'
        (default is 'flux')
    kind : string
        interpolation kind
        >>> [‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
             ‘previous’, ‘next’]
        (default is 'cubic')
    smooth : tuple
        set to smooth the input ADU/flux/mag with scipy.signal.savgol_filter
        (default is True)
    slength : int
        SG-filter window size
    sorder : int
        SG-filter polynomial order

    Returns
    -------
    A scipy interp1d object.

    '''

    # Get the standard flux/magnitude
    std_wave, std_flux = _read_standard(target, source, ftype=ftype, display=False)
    std = itp.interp1d(std_wave, std_flux, kind=kind, fill_value='extrapolate')

    # apply a Savitzky-Golay filter to remove noise and Telluric lines
    if smooth:
        flux = signal.savgol_filter(adu, slength, sorder)
    else:
      flux = adu

    # Get the sensitivity curve
    sensitivity = std(wave) / flux
    sencurve = itp.interp1d(wave, sensitivity)

    # Diagnostic plot
    if display:
        fig, ax1 = plt.subplots(figsize=(10,10))
        ax1.plot(wave, flux, label='ADU')
        ax1.legend()
        ax1.set_xlabel('Wavelength / A')
        if smooth:
            ax1.set_ylabel('Smoothed ADU')
            ax1.set_title('SG(' + str(slength) + ', ' + str(sorder) +
                          ')-Smoothed ' +  source + ' : ' + target)
        else:
            ax1.set_ylabel('ADU')
            ax1.set_title(source + ' : ' + target)

        ax2 = ax1.twinx()
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)
        ax2.plot(wave, sensitivity, color='black', label='Sensitivity Curve')
        ax2.legend()
        ax2.legend(loc='upper left')
        ax2.set_ylabel('Sensitivity Curve')
        ax2.set_yscale('log')

    return sencurve

