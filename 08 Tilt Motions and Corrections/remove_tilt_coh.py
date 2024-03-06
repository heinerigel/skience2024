import math as M
import numpy as np
import scipy as sp

#import matplotlib.pyplot as plt

def nearestPow2(x):
    a = M.pow(2, M.ceil(np.log2(x)))
    b = M.pow(2, M.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b

def p2r(radii,angles):
    return radii*np.exp(1j*angles)

def r2p(x):
    return np.abs(x), np.angle(x)

def transfer_function(tr_r, tr_s, dt, ndat):
    """
    Calculates the transfer function between the source and the response channel.
    Returns the complex transfer function, the autospectral densities, and a corresponding frequency vector.

    .. type tr_r: numpy.array
    .. param tr_r: Data array of the response channel
    .. type tr_s: numpy.array
    .. param tr_s: Data array of the source channel
    .. type dt: float
    .. param dt: sample spacing
    .. type ndat: int
    .. param ndat: number of data samples.

    return
    .. type freq: numpy.array
    .. param freq: array of frequancies
    .. type coh: numpy.array
    .. param coh: complex coherency between source and reciever
    """

    nfft = int(nearestPow2(ndat))
    nfft *= 2
    gr = np.zeros(nfft)
    gs = np.zeros(nfft)
    gr[0:ndat] = tr_r[:]
    gs[0:ndat] = tr_s[:]
    # perform ffts
    Gr = np.fft.rfft(gr)*dt
    Gs = np.fft.rfft(gs)*dt
    freq = np.fft.rfftfreq(nfft, dt)
    # perform autospectral and crossspectral densities
    w = np.blackman(100)
    XY = (Gr*Gs.conjugate())
    XX = (Gr*Gr.conjugate())
    YY = (Gs*Gs.conjugate())
    cross = np.convolve(XY,w,mode='same')
    coh = cross/np.sqrt(np.convolve(XX,w,mode='same')*np.convolve(YY,w,mode='same'))

    return freq, coh

def remove_coh(tr_r, tr_s, coh, dt, ndat, fmin, fmax, parallel=True):
    """
    Removes tilt noise from translation recordings. only parts of the spectra 
    with significant coherency are used (> 0.5)
    
    .. type tr_r: np.array
    .. param tr_r: Data array of the response channel
    .. type tr_s: np.array
    .. param tr_s: Data array of the source channel
    .. type coh: numpy.array
    .. param coh: complex coherency between source and reciever
    .. type dt: float
    .. param dt: sample spacing
    .. type ndat: int
    .. param ndat: number of data samples.
    .. type fmin: float
    .. param fmin: minimum frequency of bandwidth used for correction
                    if fmin = -1 no lower frequency limit is used
    .. type fmax: float
    .. param fmax: maximum frequency of bandwidth used for correction
                    if fmax = -1 no upper frequency limit is used
    .. type parallel: bolean
    .. param parallel: 'True' if tilt and acceleration axis are parallel
                       'False' if tilt and acceleration axis are anti parallel

    return:
    
    .. type tr_r: numpy.array
    .. param tr_r: data array of tilt corrected response

    """

    thresh = 0.5

    sig = -1
    if parallel:
        sig = 1
    nfft = int(nearestPow2(ndat))
    nfft *= 2
    gr = np.zeros(nfft)
    gs = np.zeros(nfft)
    gr[0:ndat] = tr_r[:]
    gs[0:ndat] = tr_s[:]
    # perform ffts
    Gr = np.fft.rfft(gr)* dt
    Gs = np.fft.rfft(gs)* dt
    freq = np.fft.rfftfreq(nfft, dt)
    cohe,ang = r2p(coh)

    ind = np.where(cohe >= thresh)
    cohe[ind[0]]=1.
    ind = np.where(cohe < thresh)
    cohe[ind[0]]=0.
    # use only intervall specified by [fmin, fmax]
    if fmin > 0 and fmax > 0:
        indl = np.where(freq < fmin)
        indh = np.where(freq > fmax)
        cohe[indl] = 0.
        cohe[indh] = 0.
    # remove noise and return autospectral density of corrected data
    Gr_ = np.zeros(Gr.shape, dtype=complex)
    Gs_a,Gs_p = r2p(Gs)
    Gs_a *= cohe
    Gs = p2r(Gs_a,Gs_p)
    corr = Gs[:] * 9.81
    Gr_[:] = Gr[:] -  sig * Gs[:] * 9.81
    tr_r = np.fft.irfft(Gr_)[0:ndat]/dt

    return tr_r, corr

def remove_tilt_coh(response, source, delta, npts, fmin, fmax, parallel):
    """
    This method applies the tilt subtraction in the frequency domain, applied only to 
    regions of hich coherenvy (> 0.5)
    
    .. type response: numpy.array
    .. param response: Data array of the response channel (acc)
    .. type source: numpy.array
    .. param source: Data array of the source channel (angle)
    .. type delta: float
    .. param delta: sample spacing
    .. type npts: int
    .. param npts: number of data samples.
    .. type fmin: float
    .. param fmin: minimum frequency of bandwidth used for correction
                    if fmin = -1 no lower frequency limit is used
    .. type fmax: float
    .. param fmax: maximum frequency of bandwidth used for correction
                    if fmax = -1 no upper frequency limit is used
    .. type parallel: bolean
    .. param parallel: 'True' if tilt and acceleration axis are parallel
                       'False' if tilt and acceleration axis are anti parallel

    return:
    
    .. type response_corr: numpy.array
    .. param response_corr: data array of tilt corrected response (acc)

    """
    
    sig = -1
    if parallel:
        sig = 1
    
    freq, coh = \
               transfer_function(response, source, delta, npts)
    response_corr, corr = \
               remove_coh(response, source, coh, delta, npts, fmin, fmax, parallel)

    return response_corr, freq, corr, coh

