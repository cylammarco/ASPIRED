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

from standardlist import *

def _lookup_standard(target, source, cutoff):
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
        best_match = difflib.get_close_matches(target, target_list, cutoff=cutoff)
        if len(best_match) > 0:
            print('Are you looking for these: ')
            print(best_match)
        return False


def _read_standard(target, source, cutoff, ftype, display):
    '''
    Read the standard flux/magnitude file. And return the wavelength and
    flux/mag in units of

    wavelength: \AA
    flux:       ergs / cm / cm / s / \AA
    mag:        mag (AB) 

    '''

    if not _lookup_standard(target, source, cutoff):
        return None

    flux_multiplier = 1.
    if source[:4]=='iraf':
        target_name = target + '.dat'
    else:
        if ftype == 'flux':
            target_name = 'f' + target + '.dat'
            if source != 'xshooter':
                flux_multiplier = 1e-16
        elif ftype == 'mag':
            target_name = 'm' + target + '.dat'
        else:
            print('The type has to be \'flux\' of \'mag\'.')
    
    f = np.loadtxt('standards/' + str(source) + 'stan/' + target_name)
    
    wave = f[:,0]
    if (source[:4]=='iraf') & (ftype == 'flux'):
        fluxmag = 10.**(-(f[:,1] / 2.5)) * 3630.780548 / 3.34e4 / wave**2 
    else:
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
        plt.ylim(bottom=0)
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
    print('ING TN65 & 100 Spectrophotometric Standards')
    print(ing)
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
    print('iraf Blackbody Flux Distribution')
    print(irafbb)
    print('')
    print('iraf KPNO IRS Spectrophotometric Standards')
    print(irafkpno)
    print('')
    print('iraf CTIO Spectrophotometric Standards (1983, 1984)')
    print(irafctio)
    print('')
    print('iraf CTIO Spectrophotometric Standards (1992, 1994)')
    print(irafctionew)
    print('')
    print('iraf CTIO Blue Spectrophotometric Standards (1992, 1994)')
    print(irafctionewblue)
    print('')
    print('iraf CTIO Red Spectrophotometric Standards (1992, 1994)')
    print(irafctionewred)
    print('')
    print('iraf KPNO IIDS Spectrophotometric Standards')
    print(irafiids)
    print('')
    print('iraf KPNO IRS Spectrophotometric Standards')
    print(irafirs)
    print('')
    print('iraf Oke Spectrophotometric Standards (HST Table IV)')
    print(irafoke)
    print('')
    print('iraf Extended KPNO IRS/IIDS Spectrophotometric Standards (Red)')
    print(irafred)
    print('')
    print('iraf Hamuy Spectrophotometric Standards (1992)')
    print(iraf16)
    print('')
    print('iraf Hamuy Spectrophotometric Standards (Red, 1994)')
    print(iraf16red)
    print('')
    print('iraf KPNO Spectrophotometric Standards (1988)')
    print(iraf50)
    print('')
    print('iraf KPNO/Kitt Peak Spectrophotometric Standards (1990)')
    print(iraf50ir)
    print('')
    print('iraf KPNO Hayes Spectrophotometric Standards (1988)')
    print(irafhayes)


def get_sencurve(wave, adu, target, source, exp_time, cutoff=0.4, ftype='flux',
                 kind='cubic', smooth=True, slength=11, sorder=2, display=False):
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
    std_wave, std_flux = _read_standard(
        target, source, cutoff=cutoff, ftype=ftype, display=False
        )
    std = itp.interp1d(std_wave, std_flux, kind=kind, fill_value='extrapolate')

    # apply a Savitzky-Golay filter to remove noise and Telluric lines
    if smooth:
        flux = signal.savgol_filter(adu, slength, sorder)
    else:
        flux = adu

    # adjust for exposure time
    flux = flux / exp_time

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

