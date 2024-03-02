from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms
from scipy.io import loadmat
import pycwt as wavelet
from pycwt.helpers import find
from scipy.signal import convolve2d
import warnings
import os
from obspy.signal.regression import linear_regression


## Disable Warnings
warnings.filterwarnings('ignore')
## conv2 function
# Returns the two-dimensional convolution of matrices x and y
def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

## nextpow2 function
# Returns the exponents p for the smallest powers of two that satisfy the relation  : 2**p >= abs(x)
def nextpow2(x):
    res = np.ceil(np.log2(x))
    return res.astype('int')

## Smoothing function
# Smooth the dataset
def smoothCFS(cfs, scales, dt, ns, nt):
    """
    Smoothing function
    """
    N = cfs.shape[1]
    npad = int(2 ** nextpow2(N))
    omega = np.arange(1, np.fix(npad / 2) + 1, 1).tolist()
    omega = np.array(omega) * ((2 * np.pi) / npad)
    omega_save = -omega[int(np.fix((npad - 1) / 2)) - 1:0:-1]
    omega_2 = np.concatenate((0., omega), axis=None)
    omega_2 = np.concatenate((omega_2, omega_save), axis=None)
    omega = np.concatenate((omega_2, -omega[0]), axis=None)
    # Normalize scales by DT because we are not including DT in the angular frequencies here.
    # The smoothing is done by multiplication in the Fourier domain.
    normscales = scales / dt

    for kk in range(0, cfs.shape[0]):
        
        F = np.exp(-nt * (normscales[kk] ** 2) * omega ** 2)
        smooth = np.fft.ifft(F * np.fft.fft(cfs[kk - 1], npad))
        cfs[kk - 1] = smooth[0:N]
    # Convolve the coefficients with a moving average smoothing filter across scales.
    H = 1 / ns * np.ones((ns, 1))

    cfs = conv2(cfs, H)
    return cfs




## xwt function
def xwt(trace_ref, trace_current, fs, ns=3, nt=0.25, vpo=12, freqmin=0.1, freqmax=8.0, nptsfreq=100):
    """
    Wavelet coherence transform (WCT).
​
    The WCT finds regions in time frequency space where the two time
    series co-vary, but do not necessarily have high power.
    
    Modified from https://github.com/Qhig/cross-wavelet-transform
​
    Parameters
    ----------
    trace_ref, trace_current : numpy.ndarray, list
        Input signals.
    fs : float
        Sampling frequency.
    ns : smoothing parameter. 
        Default value is 3
    nt : smoothing parameter. 
        Default value is 0.25
    vpo : float,
        Spacing parameter between discrete scales. Default value is 12.
        Higher values will result in better scale resolution, but
        slower calculation and plot.
        
    freqmin : float,
        Smallest frequency
        Default value is 0.1 Hz
    freqmax : float,
        Highest frequency
        Default value is 8.0 Hz
    nptsfreq : int,
        Number of frequency points between freqmin and freqmax.
        Default value is 100 points
    
    ----------        
    TODO.    
    normalize (boolean, optional) :
        If set to true, normalizes CWT by the standard deviation of
        the signals.
    Phase unwrapping
​
    Returns
    """
    # Choosing a Morlet wavelet with a central frequency w0 = 6
    mother = wavelet.Morlet(6.)
    # nx represent the number of element in the trace_current array
    nx = np.size(trace_current)
    x_reference = np.transpose(trace_ref)
    x_current = np.transpose(trace_current)
    # Sampling interval
    dt = 1 / fs
    # Spacing between discrete scales, the default value is 1/12
    dj = 1 / vpo 
    # Number of scales less one, -1 refers to the default value which is J = (log2(N * dt / so)) / dj.
    J = -1
    # Smallest scale of the wavelet, default value is 2*dt
    s0 = 2 * dt  # Smallest scale of the wavelet, default value is 2*dt

    # Creation of the frequency vector that we will use in the continuous wavelet transform 
    freqlim = np.linspace(freqmax, freqmin, num=nptsfreq, endpoint=True, retstep=False, dtype=None, axis=0)

    # Calculation of the two wavelet transform independently
    # scales are calculated using the wavelet Fourier wavelength
    # fft : Normalized fast Fourier transform of the input trace
    # fftfreqs : Fourier frequencies for the calculated FFT spectrum.
    ###############################################################################################################
    ###############################################################################################################
    cwt_reference, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(x_reference, dt, dj, s0, J, mother, freqs=freqlim)
    cwt_current, _, _, _, _, _ = wavelet.cwt(x_current, dt, dj, s0, J, mother, freqs=freqlim)
    ###############################################################################################################
    ###############################################################################################################

    scales = np.array([[kk] for kk in scales])
    invscales = np.kron(np.ones((1, nx)), 1 / scales)
    
    cfs2 = smoothCFS(invscales * abs(cwt_current) ** 2, scales, dt, ns, nt)
    cfs1 = smoothCFS(invscales * abs(cwt_reference) ** 2, scales, dt, ns, nt)
    
    crossCFS = cwt_reference * np.conj(cwt_current)
    WXamp = abs(crossCFS)
    # cross-wavelet transform operation with smoothing
    crossCFS = smoothCFS(invscales * crossCFS, scales, dt, ns, nt)
    WXspec = crossCFS / (np.sqrt(cfs1) * np.sqrt(cfs2))
    WXangle = np.angle(WXspec)
    Wcoh = abs(crossCFS) ** 2 / (cfs1 * cfs2)
    pp = 2 * np.pi * freqs
    pp2 = np.array([[kk] for kk in pp])
    WXdt = WXangle / np.kron(np.ones((1, nx)), pp2)


    return WXamp, WXspec, WXangle, Wcoh, WXdt, freqs, coi



## dv/v measurement function
def get_dvv(freqs, tvec, WXamp, Wcoh, delta_t, lag_min, lag_max, freqmin=0.1, freqmax=2.0):
    """
    Measure velocity variations (dv/v) from the Wavelet coherence transform (WCT).
    
    Parameters
    ----------
    freqs :
        frequencies used in CWT
    tvec : numpy.ndarray
        time vector of the CCFs
    WXamp : numpy.ndarray
        amplitude product of two CWT in time-frequency domain
    Wcoh :  numpy.ndarray
        wavelet coherence
    delta_t : numpy.ndarray
        time difference between the two inputs in the time-frequency domain.
    lag_min : float
        lower limit of the analyzed lag tim on the CCF
    lag_max : float
        higher limit of the analyzed lag tim on the CCF
    freqmin :
        lower limit of the analyzed frequency range
    freqmax :
        higher limit of the analyzed frequency range
    RETURNS:
    ------------------
    dvv*100 : estimated dv/v in %
    err*100 : error of dv/v estimation in %
    wf : weighting function used for the linear regressions
    
    """
    inx = np.where((freqs>=freqmin) & (freqs<=freqmax)) #TODO don't hardcode frequency range
    dvv, err = np.zeros(inx[0].shape), np.zeros(inx[0].shape) # Create empty vectors vor dvv and err
    
    t=tvec
    
    
    ## Better weight function
    weight_func = np.log(np.abs(WXamp))/np.log(np.abs(WXamp)).max()
    zero_idx = (np.where((Wcoh<=0.65) | (delta_t>0.1))) #TODO get values from db
    wf = weight_func+abs(np.nanmin(weight_func))
    wf = wf/wf.max()
    wf[zero_idx] = 0
    
    ## Coda selection
    tindex = np.where(((t >= -lag_max) & (t <= -lag_min)) | ((t >= lag_min) & (t <= lag_max)))[0] # Index of the coda
    # loop through freq for linear regression
    for ii, ifreq in enumerate(inx[0]): # Loop through frequencies index
        if len(tvec)>2: # check time vector size
            if not np.any(delta_t[ifreq]): # check non-empty dt array
                continue
            delta_t[ifreq][tindex]=np.nan_to_num(delta_t[ifreq][tindex])
            w = wf[ifreq] # weighting function for the specific frequency
            w[~np.isfinite(w)] = 1.0
            #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False) # if note forcing through origin
            m, em = linear_regression(tvec[tindex], delta_t[ifreq][tindex], w[tindex], intercept_origin=True) #Forcing through origin
            dvv[ii], err[ii] = -m, em
        else:
            print('not enough points to estimate dv/v for wct')
            dvv[ii], err[ii]=np.nan, np.nan         
    
    return dvv*100, err*100, wf




def do_plot(time, WXamp, WXspec, WXangle, Wcoh, WXdt, freqs, coi, w, sta, date, comp):   
    save_dir = "WCT/Figure"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    cmap = "plasma"

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    dt = ax1.pcolormesh(time, freqs, WXdt, cmap="seismic_r", edgecolors='none', vmin=-0.1,vmax=0.1)
    plt.colorbar(dt, ax=ax1)
    ax1.plot(time, 1/coi, 'w--', linewidth=2)
    ax1.set_ylim(freqs[-1], freqs[0])
    ax1.set_xlim(0,50)
    ax1.set_title('Smoothed Time difference', fontsize=13)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')

    wc = ax2.pcolormesh(time, freqs, Wcoh, cmap=cmap, edgecolors='none', vmin=0.6, vmax=1)
    plt.colorbar(wc, ax=ax2)
    ax2.plot(time, 1/coi, 'w--', linewidth=2)
    ax2.set_ylim(freqs[-1], freqs[0])
    ax2.set_xlim(0,50)
    ax2.set_title('Wavelet Coherence', fontsize=13)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')

    la = ax3.pcolormesh(time, freqs, np.log(WXamp), cmap=cmap, edgecolors='none')
    #plt.clim([-50, 0])
    plt.colorbar(la, ax = ax3)
    ax3.plot(time, 1/coi, 'w--', linewidth=2)
    ax3.set_ylim(freqs[-1], freqs[0])
    ax3.set_xlim(0,50)
    ax3.set_title('(Logarithmic) Amplitude', fontsize=13)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')

    weigh = ax4.pcolormesh(time, freqs, w, cmap=cmap, edgecolors='none')
    plt.colorbar(weigh, ax=ax4)
    ax4.plot(time, 1/coi, 'w--', linewidth=2)
    ax4.set_ylim(freqs[-1], freqs[0])
    ax4.set_xlim(0,50)
    ax4.set_title('Weighting function', fontsize=13)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')



    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"{} {}_{}.png".format(sta.replace(":","_"), comp, date)),dpi=300)
#    plt.show()
    plt.close(fig)
    
    return
