import sys
import os
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

library_list = [
    'esoctio', 'esohst', 'ing', 'esooke', 'esowd', 'esoxshooter', 'irafbb',
    'irafkpno', 'irafctio', 'irafctionew', 'irafctionewblue', 'irafctionewred',
    'irafiids', 'irafirs', 'irafoke', 'irafred', 'iraf16', 'iraf16blue',
    'iraf16red', 'iraf50', 'iraf50red', 'irafhayes'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/hamuystandards.html
esoctio = [
    'cd32d9927', 'cd_34d241', 'eg21', 'eg274', 'feige110', 'feige56',
    'hilt600', 'hr1544', 'hr3454', 'hr4468', 'hr4963', 'hr5501', 'hr718',
    'hr7596', 'hr7950', 'hr8634', 'hr9087', 'ltt1020', 'ltt1788', 'ltt2415',
    'ltt3218', 'ltt3864', 'ltt4364', 'ltt4816', 'ltt6248', 'ltt7379', 'ltt745',
    'ltt7987', 'ltt9239', 'ltt9491'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/hststandards.html
esohst = [
    'agk81d226', 'bd28d4211', 'bd33d2642', 'bd75d325', 'bpm16274', 'feige110',
    'feige34', 'g191b2b', 'g93_48', 'gd108', 'gd50', 'grw70d5824', 'hd49798',
    'hd60753', 'hd93521', 'hr153', 'hr1996', 'hr4554', 'hr5191', 'hr7001',
    'hz2', 'hz21', 'hz4', 'hz44', 'lb227', 'lds749b', 'ngc7293'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/okestandards_rev.html
esooke = [
    'bd25d4655', 'bd28d4211', 'bd33d2642', 'bd75d325', 'feige110', 'feige34',
    'feige66', 'feige67', 'g138_31', 'g158_100', 'g191b2b', 'g193_74', 'g24_9',
    'g60_54', 'gd108', 'gd248', 'gd50', 'grw70d5824', 'hd93521', 'hz21', 'hz4',
    'hz44', 'ltt9491', 'ngc7293', 'sa95_42'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/wdstandards.html
esowd = [
    'agk81d226_005', 'alpha_lyr_004', 'bd_25d4655_002', 'bd_28d4211_005',
    'bd_33d2642_004', 'bd_75d325_005', 'feige110_005', 'feige34_005',
    'feige66_002', 'feige67_002', 'g93_48_004', 'gd108_005', 'gd50_004',
    'gd71', 'grw_70d5824_005', 'hd93521_005', 'hz21_005', 'hz2_005',
    'hz44_005', 'hz4_004', 'lb227_004', 'lds749b_005', 'ltt9491_002',
    'ngc7293_005'
]

# https://www.eso.org/sci/observing/tools/standards/spectra/Xshooterspec.html
esoxshooter = ['EG274', 'Feige110', 'GD153', 'GD71', 'LTT3218', 'LTT7987']

# http://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/workflux.html
ing = [
    'g158_100', 'hiltner102', 'l870_2', 'feige15'
    'pg0205_134', 'PG0216_032', 'feige24', 'feige25', 'hd19445', 'pg0310_149',
    'gd50', 'sa95_42', 'hz4', 'lb1240', 'lb227', 'hz2', '40erib', 'hz7',
    'hz15', 'hz14', 'g191_b2b', 'g99_37', 'hiltner600', 'he3', 'l745_46a',
    'g193_74', 'bd75_325', 'bd08_2015', 'pg0823_546', 'lds235b', 'pg0846_249',
    'g47_18', 'pg0934_554', 'pg1939_262', 'sa29-130', 'hd84937', 'gd108',
    'feige34', 'hd93521', 'l970_30', 'ton573', 'pg1121_145', 'ross627', 'eg81',
    'gd140', 'feige56', 'hz21', 'hz29', 'feige66', 'feige67', 'g60-54', 'hz43',
    'hz44', 'wolf485', 'grw70_5824', 'feige92', 'feige98', 'bd26_2606',
    'gd190', 'pg1545_035', 'bd33_2642', 'g138_31', 'ross640', 'pg1708_602',
    'kopff27', 'grw70_8247', 'bd25_3941', 'bd40_4032', 'hd192281', 'g24_9',
    'cygob2_9', 'wolf1346', 'grw73_8031', 'lds749b', 'l1363_3', 'l930_80',
    'bd28_4211', 'bd25_4655', 'bd17_4708', 'ngc7293', 'hd217086', 'g157_34',
    'ltt9491', 'feige110', 'gd248', 'l1512_34'
]

# The following iraf standards refer to:
# https://github.com/iraf-community/iraf/tree/master/noao/lib/onedstds
irafbb = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'L', 'Lprime', 'M']

irafkpno = [
    'hr718', 'hr3454', 'hr3982', 'hr4468', 'hr4534', 'hr5191', 'hr5511',
    'hr7001', 'hr7596', 'hr7950', 'hr8634', 'hr9087', 'hr15318', 'hr74280',
    'hr100889', 'hr188350', 'hr198001', 'hr214923', 'hr224926'
]

irafctio = [
    'bd8', 'bd25', 'bd73632', 'cd32', 'eg11', 'eg21', 'eg26', 'eg31', 'eg54',
    'eg63', 'eg76', 'eg79', 'eg99', 'eg139', 'eg149', 'eg158', 'eg248',
    'eg274', 'f15', 'f25', 'f56', 'f98', 'f110', 'feige15', 'feige25',
    'feige56', 'feige98', 'feige110', 'g2631', 'g9937', 'g16350', 'h600',
    'hz2', 'hz4', 'hz15', 'kopf27', 'l377', 'l1020', 'l1788', 'l2415', 'l2511',
    'l3218', 'l3864', 'l4364', 'l4816', 'l6248', 'l7379', 'l7987', 'l8702',
    'l9239', 'l9491', 'l74546', 'l93080', 'l97030', 'lds235', 'lds749',
    'ltt4099', 'ltt8702', 'rose627', 'w1346', 'w485a', 'wolf1346', 'wolf485a'
]

irafctionew = [
    'cd32', 'eg21', 'eg274', 'f56', 'f110', 'h600', 'l377', 'l745', 'l1020',
    'l1788', 'l2415', 'l2511', 'l3218', 'l3864', 'l4364', 'l4816', 'l6248',
    'l7379', 'l7987', 'l9239', 'l9491',
]

irafctionewblue = [
    'cd32', 'eg21', 'eg274', 'f56', 'f110', 'h600', 'l377', 'l1020', 'l1788',
    'l2415', 'l2511', 'l3218', 'l3864', 'l4364', 'l4816', 'l6248', 'l7379',
    'l7987', 'l9239', 'l9491',
]

irafctionewred = [
    'cd32', 'eg21', 'eg274', 'f56', 'f110', 'h600', 'l377', 'l745', 'l1020',
    'l1788', 'l2415', 'l2511', 'l3218', 'l3864', 'l4364', 'l4816', 'l6248',
    'l7379', 'l7987', 'l9239', 'l9491',
]

irafiids = [
    '40erib', 'amcvn', 'bd7781', 'bd73632', 'bd82015', 'bd253941', 'bd284211',
    'bd332642', 'bd404032', 'eg11', 'eg20', 'eg26', 'eg28', 'eg29', 'eg31',
    'eg33', 'eg39', 'eg42', 'eg50', 'eg54', 'eg63', 'eg67', 'eg71', 'eg76',
    'eg77', 'eg79', 'eg91', 'eg98', 'eg99', 'eg102', 'eg119', 'eg129', 'eg139',
    'eg144', 'eg145', 'eg148', 'eg149', 'eg158', 'eg162', 'eg182', 'eg184',
    'eg193', 'eg247', 'eg248', 'feige15', 'feige24', 'feige25', 'feige34',
    'feige56', 'feige92', 'feige98', 'feige110', 'g88', 'g2610', 'g2631',
    'g4718', 'g9937', 'g12627', 'g14563', 'g16350', 'g191b2b', 'gd128',
    'gd140', 'gd190', 'gh7112', 'grw705824', 'grw708247', 'grw738031', 'he3',
    'hz2', 'hz4', 'hz7', 'hz14', 'hz15', 'hz29', 'hz43', 'hz44', 'kopff27',
    'hiltner102', 'hiltner600', 'l8702', 'l13633', 'l14094', 'l74546a',
    'l93080', 'l97030', 'l140349', 'l151234b', 'lft1655', 'lb227', 'lb1240',
    'lds235b', 'lds749b', 'lp414101', 'ltt4099', 'ltt8702', 'ltt13002',
    'ltt16294', 'ross627', 'ross640', 'sa29130', 'sao131065', 'ton573',
    'wolf1346', 'wolf485a'
]

irafirs = [
    'bd082015', 'bd174708', 'bd253941', 'bd262606', 'bd284211', 'bd332642',
    'bd404032', 'eg50', 'eg71', 'eg139', 'eg158', 'eg247', 'feige15',
    'feige25', 'feige34', 'feige56', 'feige92', 'feige98', 'feige110',
    'g191b2b', 'hd2857', 'hd17520', 'hd19445', 'hd60778', 'hd74721', 'hd84937',
    'hd86986', 'hd109995', 'hd117880', 'hd161817', 'hd192281', 'hd217086',
    'he3', 'hiltner102', 'hiltner600', 'hr7001', 'hz44', 'kopff27', 'wolf1346'
]

irafoke = [
    'bd75325', 'bd284211', 'feige34', 'feige67', 'feige110', 'g249', 'g13831',
    'g191b2b', 'g19374', 'gd108', 'gd248', 'hz21', 'ltt9491', 'eg71', 'eg158',
    'eg247'
]

irafred = [
    '40erib', 'amcvn', 'bd7781', 'bd73632', 'bd174708', 'bd262606', 'eg20',
    'eg33', 'eg50', 'eg54', 'eg63', 'eg67', 'eg76', 'eg79', 'eg91', 'eg98',
    'eg99', 'eg102', 'eg119', 'eg129', 'eg139', 'eg144', 'eg145', 'eg148',
    'eg149', 'eg158', 'eg162', 'eg182', 'eg184', 'eg193', 'eg247', 'eg248',
    'feige24', 'g2610', 'g2631', 'g4718', 'g9937', 'g12627', 'g14563',
    'g16350', 'g191b2b', 'gd140', 'gd190', 'grw705824', 'grw708247',
    'grw738031', 'hd19445', 'hd84937', 'he3', 'hz29', 'hz43', 'hz44', 'l13633',
    'l14094', 'l151234b', 'l74546a', 'l93080', 'l97030', 'lds235b', 'lds749b',
    'lft1655', 'ltt4099', 'ltt8702', 'ltt16294', 'ross627', 'ross640',
    'sa29130', 'sao131065', 'wolf1346', 'wolf485a'
]

iraf16 = [
    'hd15318', 'hd30739', 'hd74280', 'hd100889', 'hd114330', 'hd129956',
    'hd188350', 'hd198001', 'hd214923', 'hd224926', 'hr718', 'hr1544',
    'hr3454', 'hr4468', 'hr4963', 'hr5501', 'hr7596', 'hr7950', 'hr8634',
    'hr9087'
]

iraf16blue = [
    'hd15318', 'hd30739', 'hd74280', 'hd100889', 'hd114330', 'hd129956',
    'hd188350', 'hd198001', 'hd214923', 'hd224926', 'hr718', 'hr1544',
    'hr3454', 'hr4468', 'hr4963', 'hr5501', 'hr7596', 'hr7950', 'hr8634',
    'hr9087'
]

iraf16red = [
    'hd15318', 'hd30739', 'hd74280', 'hd100889', 'hd114330', 'hd129956',
    'hd188350', 'hd198001', 'hd214923', 'hd224926', 'hr718', 'hr1544',
    'hr3454', 'hr4468', 'hr4963', 'hr5501', 'hr7596', 'hr7950', 'hr8634',
    'hr9087'
]

iraf50 = [
    'bd284211', 'cygob2no9', 'eg20', 'eg42', 'eg71', 'eg81', 'eg139', 'eg158',
    'eg247', 'feige34', 'feige66', 'feige67', 'feige110', 'g191b2b', 'gd140',
    'hd192281', 'hd217086', 'hilt600', 'hz14', 'hz44', 'pg0205134',
    'pg0216032', 'pg0310149', 'pg0823546', 'pg0846249', 'pg0934554',
    'pg0939262', 'pg1121145', 'pg1545035', 'pg1708602', 'wolf1346'
]

iraf50red = [
    'bd284211', 'eg71', 'eg139', 'eg158', 'eg247', 'feige34', 'feige66',
    'feige67', 'feige110', 'g191b2b', 'gd140', 'hilt600', 'hz44', 'pg0823546',
    'wolf1346'
]

irafhayes = [
    'bd284211', 'cygob2no9', 'eg42', 'eg71', 'eg81', 'eg139', 'eg158', 'eg247',
    'feige34', 'feige66', 'feige67', 'feige110', 'g191b2b', 'gd140',
    'hd192281', 'hd217086', 'hilt600', 'hz14', 'hz44', 'pg0205134',
    'pg0216032', 'pg0310149', 'pg0823546', 'pg0846249', 'pg0934554',
    'pg0939262', 'pg1121145', 'pg1545035', 'pg1708602', 'wolf1346'
]

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

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path, '..', 'standards', str(source) + 'stan', target_name)

    if source[:4]=='iraf':
      f = np.loadtxt(filepath, skiprows=1)
    else:
      f = np.loadtxt(filepath)
    
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
    sencurve = itp.interp1d(wave, sensitivity, fill_value='extrapolate')

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

