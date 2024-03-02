# -*- coding: utf-8 -*-
"""
@authors: A.Maggi 2016 > Orignial code and porting from Matlab
          C. Hibert after 22/05/2017 > Original code from Matlab and addition of spectrogram attributes and other stuffs + comments 

This function computes the attributes of a seismic signal later used to perform identification through machine
learning algorithms.

- Example: from ComputeAttributes_CH_V1 import calculate_all_attributes 
        
           all_attributes = calculate_all_attributes(Data,sps,flag)


- Inputs: "Data" is the raw seismic signal of the event (cutted at the onset and at the end of the signal)
          "sps" is the sampling rate of the seismic signal (in samples per second)
          "flag" is used to indicate if the input signal is 3C (flag==1) or 1C (flag==0).
          /!\ 3C PROCESSING NOT FULLY IMPLEMENTED YET /!\ 
          
- Output: "all_attributes" is an array of the attribute values for the input signal, ordered as detailed on lines 69-137

- Tweaks: Possibility to adapt the frequncy bands used to compute the energies and the Kurtosis envelopes 
          (attributes #13-#22) on lines 293-294


- References: 
    
        Maggi, A., Ferrazzini, V., Hibert, C., Beauducel, F., Boissier, P., & Amemoutou, A. (2017). Implementation of a 
        multistation approach for automated event classification at Piton de la Fournaise volcano. 
        Seismological Research Letters, 88(3), 878-891.
        
        Provost, F., Hibert, C., & Malet, J. P. (2017). Automatic classification of endogenous landslide seismicity 
        using the Random Forest supervised classifier. Geophysical Research Letters, 44(1), 113-120.
        
        Hibert, C., Provost, F., Malet, J. P., Maggi, A., Stumpf, A., & Ferrazzini, V. (2017). Automatic identification 
        of rockfalls and volcano-tectonic earthquakes at the Piton de la Fournaise volcano using a Random Forest 
        algorithm. Journal of Volcanology and Geothermal Research, 340, 130-142.
 
"""

import numpy as np
from scipy.signal import hilbert, lfilter, butter, spectrogram
from scipy.stats import kurtosis, skew
import sys
sys.path.append(".")
from detect_peaks import detect_peaks
import matplotlib.pyplot as plt


# -----------------------------------#
#            Main Function           #
# -----------------------------------#
 

def calculate_all_attributes(Data,sps,flag):

    # for 3C make sure is in right order (Z then horizontals)
    
    if flag==1:
        NATT = 62
        
    if flag==0:
        NATT = 58
            
        
    all_attributes = np.empty((1, NATT), dtype=float)

    env = envelope(Data,sps)
    
    TesMEAN, TesMEDIAN, TesSTD, env_max = get_TesStuff(env)
    
    RappMaxMean, RappMaxMedian = get_RappMaxStuff(TesMEAN, TesMEDIAN)
   
    AsDec, DistDecAmpEnv = get_AsDec(Data, env, sps)
   
    KurtoEnv, KurtoSig, SkewnessEnv, SkewnessSig =\
        get_KurtoSkewStuff(Data, env)
    
    CorPeakNumber, INT1, INT2, INT_RATIO = get_CorrStuff(Data, sps)
   
    ES, KurtoF = get_freq_band_stuff(Data, sps)
    
    MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1, Fquart3,\
        NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1, gamma2,\
        gammas = get_full_spectrum_stuff(Data, sps)
    
    if flag==1: #If signal is 3C then compute polarisation parameter
        rectilinP, azimuthP, dipP, Plani =\
            get_polarization_stuff(Data, env)

    SpecKurtoMaxEnv, SpecKurtoMedianEnv, RATIOENVSPECMAXMEAN, RATIOENVSPECMAXMEDIAN, \
    DISTMAXMEAN , DISTMAXMEDIAN, NBRPEAKMAX, NBRPEAKMEAN, NBRPEAKMEDIAN, RATIONBRPEAKMAXMEAN, \
    RATIONBRPEAKMAXMED, NBRPEAKFREQCENTER, NBRPEAKFREQMAX, RATIONBRFREQPEAKS, DISTQ2Q1, DISTQ3Q2, DISTQ3Q1 \
    = get_pseudo_spectral_stuff(Data, sps)

    # waveform
    all_attributes[0, 0] = np.mean(duration(Data,sps))  # 1  Duration of the signal
    all_attributes[0, 1] = np.mean(RappMaxMean)         # 2  Ratio of the Max and the Mean of the normalized envelope
    all_attributes[0, 2] = np.mean(RappMaxMedian)       # 3  Ratio of the Max and the Median of the normalized envelope
    all_attributes[0, 3] = np.mean(AsDec)               # 4  Ascending time/Decreasing time of the envelope
    all_attributes[0, 4] = np.mean(KurtoSig)            # 5  Kurtosis Signal
    all_attributes[0, 5] = np.mean(KurtoEnv)            # 6  Kurtosis Envelope
    all_attributes[0, 6] = np.mean(np.abs(SkewnessSig)) # 7  Skewness Signal
    all_attributes[0, 7] = np.mean(np.abs(SkewnessEnv)) # 8  Skewness envelope
    all_attributes[0, 8] = np.mean(CorPeakNumber)       # 9  Number of peaks in the autocorrelation function
    all_attributes[0, 9] = np.mean(INT1)                #10  Energy in the 1/3 around the origin of the autocorr function
    all_attributes[0, 10] = np.mean(INT2)               #11  Energy in the last 2/3 of the autocorr function
    all_attributes[0, 11] = np.mean(INT_RATIO)          #12  Ratio of the energies above
    all_attributes[0, 12] = np.mean(ES[0])              #13  Energy of the seismic signal in the 0.03-1Hz FBand
    all_attributes[0, 13] = np.mean(ES[1])              #14  Energy of the seismic signal in the 1-4Hz FBand
    all_attributes[0, 14] = np.mean(ES[2])              #15  Energy of the seismic signal in the 4-8Hz FBand
    all_attributes[0, 15] = np.mean(ES[3])              #16  Energy of the seismic signal in the 8-12Hz FBand
    all_attributes[0, 16] = np.mean(ES[4])              #17  Energy of the seismic signal in the 10-Nyquist F FBand
    all_attributes[0, 17] = np.mean(KurtoF[0])          #18  Kurtosis of the signal in the 0.03-1Hz FBand
    all_attributes[0, 18] = np.mean(KurtoF[1])          #19  Kurtosis of the signal in the 1-4Hz FBand
    all_attributes[0, 19] = np.mean(KurtoF[2])          #20  Kurtosis of the signal in the 4-8Hz FBand
    all_attributes[0, 20] = np.mean(KurtoF[3])          #21  Kurtosis of the signal in the 8-12Hz FBand
    all_attributes[0, 21] = np.mean(KurtoF[4])          #22  Kurtosis of the signal in the 10-Nyf Hz FBand
    all_attributes[0, 22] = np.mean(DistDecAmpEnv)      #23  Difference bewteen decreasing coda amplitude and straight line
    all_attributes[0, 23] = np.mean(env_max/duration(Data,sps)) # 24  Ratio between max envlelope and duration

    # spectral
    all_attributes[0, 24] = np.mean(MeanFFT)            #25  Mean FFT
    all_attributes[0, 25] = np.mean(MaxFFT)             #26  Max FFT
    all_attributes[0, 26] = np.mean(FmaxFFT)            #27  Frequence at Max(FFT)
    all_attributes[0, 27] = np.mean(FCentroid)          #28  Fq of spectrum centroid
    all_attributes[0, 28] = np.mean(Fquart1)            #29  Fq of 1st quartile
    all_attributes[0, 29] = np.mean(Fquart3)            #30  Fq of 3rd quartile
    all_attributes[0, 30] = np.mean(MedianFFT)          #31  Median Normalized FFT spectrum
    all_attributes[0, 31] = np.mean(VarFFT)             #32  Var Normalized FFT spectrum
    all_attributes[0, 32] = np.mean(NpeakFFT)           #33  Number of peaks in normalized FFT spectrum
    all_attributes[0, 33] = np.mean(MeanPeaksFFT)       #34  Mean peaks value for peaks>0.7
    all_attributes[0, 34] = np.mean(E1FFT)              #35  Energy in the 1 -- NyF/4 Hz (NyF=Nyqusit Freq.) band
    all_attributes[0, 35] = np.mean(E2FFT)              #36  Energy in the NyF/4 -- NyF/2 Hz band
    all_attributes[0, 36] = np.mean(E3FFT)              #37  Energy in the NyF/2 -- 3*NyF/4 Hz band
    all_attributes[0, 37] = np.mean(E4FFT)              #38  Energy in the 3*NyF/4 -- NyF/2 Hz band
    all_attributes[0, 38] = np.mean(gamma1)             #39  Spectrim centroid
    all_attributes[0, 39] = np.mean(gamma2)             #40  Spectrim gyration radio
    all_attributes[0, 40] = np.mean(gammas)             #41  Spectrim centroid width
    
    # Pseudo-Spectro.
    all_attributes[0, 41] = np.mean(SpecKurtoMaxEnv)    #42  Kurto of the envelope of the maximum energy on spectros
    all_attributes[0, 42] = np.mean(SpecKurtoMedianEnv) #43  Kurto of the envelope of the median energy on spectros
    all_attributes[0, 43] = np.mean(RATIOENVSPECMAXMEAN)#44  Ratio Max DFT(t)/ Mean DFT(t)
    all_attributes[0, 44] = np.mean(RATIOENVSPECMAXMEDIAN)#45  Ratio Max DFT(t)/ Median DFT(t)
    all_attributes[0, 45] = np.mean(DISTMAXMEAN)        #46  Nbr peaks Max DFTs(t)
    all_attributes[0, 46] = np.mean(DISTMAXMEDIAN)      #47  Nbr peaks Mean DFTs(t)
    all_attributes[0, 47] = np.mean(NBRPEAKMAX)         #48  Nbr peaks Median DFTs(t)
    all_attributes[0, 48] = np.mean(NBRPEAKMEAN)        #49  Ratio Max/Mean DFTs(t)
    all_attributes[0, 49] = np.mean(NBRPEAKMEDIAN)      #50  Ratio Max/Median DFTs(t)
    all_attributes[0, 50] = np.mean(RATIONBRPEAKMAXMEAN)#51  Nbr peaks X centroid Freq DFTs(t)
    all_attributes[0, 51] = np.mean(RATIONBRPEAKMAXMED) #52  Nbr peaks X Max Freq DFTs(t)
    all_attributes[0, 52] = np.mean(NBRPEAKFREQCENTER)  #53  Ratio Freq Max/X Centroid DFTs(t)
    all_attributes[0, 53] = np.mean(NBRPEAKFREQMAX)     #54  Mean distance bewteen Max DFT(t) Mean DFT(t)
    all_attributes[0, 54] = np.mean(RATIONBRFREQPEAKS)  #55  Mean distance bewteen Max DFT Median DFT
    all_attributes[0, 55] = np.mean(DISTQ2Q1)           #56  Distance Q2 curve to Q1 curve (QX curve = envelope of X quartile of DTFs)
    all_attributes[0, 56] = np.mean(DISTQ3Q2)           #57  Distance Q3 curve to Q2 curve
    all_attributes[0, 57] = np.mean(DISTQ3Q1)           #58  Distance Q3 curve to Q1 curve
    
    # polarisation
    if flag==1:
        all_attributes[0, 58] = rectilinP
        all_attributes[0, 59] = azimuthP
        all_attributes[0, 60] = dipP
        all_attributes[0, 61] = Plani

    return all_attributes
    
def get_attribute_names():
    attributes = {
    0: 'Duration of the signal',
    1: 'Ratio of the Max and the Mean of the normalized envelope',
    2: 'Ratio of the Max and the Median of the normalized envelope',
    3: 'Ascending time/Decreasing time of the envelope',
    4: 'Kurtosis Signal',
    5: 'Kurtosis Envelope',
    6: 'Skewness Signal',
    7: 'Skewness envelope',
    8: 'Number of peaks in the autocorrelation function',
    9: 'Energy in the 1/3 around the origin of the autocorr function',
    10: 'Energy in the last 2/3 of the autocorr function',
    11: 'Ratio of the energies above',
    12: 'Energy of the seismic signal in the 0.03-1Hz FBand',
    13: 'Energy of the seismic signal in the 1-4Hz FBand',
    14: 'Energy of the seismic signal in the 4-8Hz FBand',
    15: 'Energy of the seismic signal in the 8-12Hz FBand',
    16: 'Energy of the seismic signal in the 10-Nyquist F FBand',
    17: 'Kurtosis of the signal in the 0.03-1Hz FBand',
    18: 'Kurtosis of the signal in the 1-4Hz FBand',
    19: 'Kurtosis of the signal in the 4-8Hz FBand',
    20: 'Kurtosis of the signal in the 8-12Hz FBand',
    21: 'Kurtosis of the signal in the 10-Nyf Hz FBand',
    22: 'Difference bewteen decreasing coda amplitude and straight line',
    23: 'Ratio between max envlelope and duration',
    24: 'Mean FFT',
    25: 'Max FFT',
    26: 'Frequence at Max(FFT)',
    27: 'Fq of spectrum centroid',
    28: 'Fq of 1st quartile',
    29: 'Fq of 3rd quartile',
    30: 'Median Normalized FFT spectrum',
    31: 'Var Normalized FFT spectrum',
    32: 'Number of peaks in normalized FFT spectrum',
    33: 'Mean peaks value for peaks>0.7',
    34: 'Energy in the 1 -- NyF/4 Hz (NyF=Nyqusit Freq.) band',
    35: 'Energy in the NyF/4 -- NyF/2 Hz band',
    36: 'Energy in the NyF/2 -- 3*NyF/4 Hz band',
    37: 'Energy in the 3*NyF/4 -- NyF/2 Hz band',
    38: 'Spectrim centroid',
    39: 'Spectrim gyration radio',
    40: 'Spectrim centroid width',
    41: 'Kurto of the envelope of the maximum energy on spectros',
    42: 'Kurto of the envelope of the median energy on spectros',
    43: 'Ratio Max DFT(t)/ Mean DFT(t)',
    44: 'Ratio Max DFT(t)/ Median DFT(t)',
    45: 'Nbr peaks Max DFTs(t)',
    46: 'Nbr peaks Mean DFTs(t)',
    47: 'Nbr peaks Median DFTs(t)',
    48: 'Ratio Max/Mean DFTs(t)',
    49: 'Ratio Max/Median DFTs(t)',
    50: 'Nbr peaks X centroid Freq DFTs(t)',
    51: 'Nbr peaks X Max Freq DFTs(t)',
    52: 'Ratio Freq Max/X Centroid DFTs(t)',
    53: 'Mean distance bewteen Max DFT(t) Mean DFT(t)',
    54: 'Mean distance bewteen Max DFT Median DFT',
    55: 'Distance Q2 curve to Q1 curve (QX curve = envelope of X quartile of DTFs)',
    56: 'Distance Q3 curve to Q2 curve',
    57: 'Distance Q3 curve to Q1 curve'
    }
    return attributes

# -----------------------------------#
#        Secondary Functions         #
# -----------------------------------#
    
def duration(Data,sps):

    dur = len(Data) / sps

    return dur


def envelope(Data,sps):
    
    env = np.abs(hilbert(Data))

    return env


def get_TesStuff(env):

    CoefSmooth=3
    light_filter = np.ones(CoefSmooth) / float(CoefSmooth)

#    for i in range(ntr):
    env_max = np.max(env)
    tmp = lfilter(light_filter, 1, env/env_max)
    TesMEAN = np.mean(tmp)
    TesMEDIAN = np.median(tmp)
    TesSTD = np.std(tmp)

    return TesMEAN, TesMEDIAN, TesSTD, env_max


def get_RappMaxStuff(TesMEAN, TesMEDIAN):


    npts = 1
    RappMaxMean = np.empty(npts, dtype=float)
    RappMaxMedian = np.empty(npts, dtype=float)

    #for i in range(npts):
    RappMaxMean = 1./TesMEAN
    RappMaxMedian = 1./TesMEDIAN

    return RappMaxMean, RappMaxMedian


def get_AsDec(Data, env, sps):

    strong_filter = np.ones(int(sps)) / float(sps)

    smooth_env = lfilter(strong_filter, 1, env)
    imax = np.argmax(smooth_env)
    
    if float(len(Data) - (imax+1))>0:
        AsDec = (imax+1) / float(len(Data) - (imax+1))
    else:
        AsDec = 0 
    
    dec = Data[imax:]
    lendec = len(dec)
    if lendec > 0:
        DistDecAmpEnv = np.abs(np.mean(np.abs(hilbert(dec / np.max(Data))) -
            (1 - ((1 / float(lendec)) * (np.arange(lendec)+1)))))
    else:
        DistDecAmpEnv = 0

    return AsDec, DistDecAmpEnv


def get_KurtoSkewStuff(Data, env):

    ntr = 1

    KurtoEnv = np.empty(ntr, dtype=float)
    KurtoSig = np.empty(ntr, dtype=float)
    SkewnessEnv = np.empty(ntr, dtype=float)
    SkewnessSig = np.empty(ntr, dtype=float)
    CoefSmooth = 3
    
    light_filter = np.ones(CoefSmooth) / float(CoefSmooth)

#   for i in range(ntr):
    env_max = np.max(env)
    data_max = np.max(Data)
    tmp = lfilter(light_filter, 1, env/env_max)
    KurtoEnv = kurtosis(tmp, fisher=False)
    SkewnessEnv = skew(tmp)
    KurtoSig = kurtosis(Data / data_max, fisher=False)
    SkewnessSig = skew(Data / data_max)

    return KurtoEnv, KurtoSig, SkewnessEnv, SkewnessSig


def get_CorrStuff(Data,sps):

   
    strong_filter = np.ones(int(sps)) / float(sps)
    min_peak_height = 0.4

    ntr=1
    CorPeakNumber = np.empty(ntr, dtype=int)
    INT1 = np.empty(ntr, dtype=float)
    INT2 = np.empty(ntr, dtype=float)
    INT_RATIO = np.empty(ntr, dtype=float)

#    for i in range(ntr):
    cor = np.correlate(Data, Data, mode='full')
    cor = cor / np.max(cor)

    # find number of peaks
    cor_env = np.abs(cor)

    cor_smooth = lfilter(strong_filter, 1, cor_env)
    cor_smooth2 = lfilter(strong_filter, 1, cor_smooth/np.max(cor_smooth))
    ipeaks=detect_peaks(cor_smooth2,min_peak_height)
    
    CorPeakNumber =len(ipeaks)

    # integrate over bands
    npts = len(cor_smooth)
    ilag_0 = np.argmax(cor_smooth)+1
    ilag_third = ilag_0 + npts/6

    
    max_cor = np.max(cor_smooth)
    int1 = np.trapz(cor_smooth[int(ilag_0):int(ilag_third)+1]/max_cor)
    int2 = np.trapz(cor_smooth[int(ilag_third):]/max_cor)
    int_ratio = int1 / int2

    INT1 = int1
    INT2 = int2
    INT_RATIO = int_ratio

    return CorPeakNumber, INT1, INT2, INT_RATIO


def get_freq_band_stuff(Data,sps):

    NyF = sps / 2
#    ntr=1

    # lower bounds of the different tested freq. bands
    FFI = np.array([0.03, 1, 4, 8, 10])
    
    # higher bounds of the different tested freq. bands
    FFE = np.array([ 1, 4, 8, 12, NyF-0.01])

#    ntr = len(st)
    nf = len(FFI)

    ES = np.empty(nf, dtype=float)
    KurtoF = np.empty(nf, dtype=float)

#    for i in range(ntr):
    for j in range(nf):
#        tr = Data
        Fb, Fa = butter(2, [FFI[j]/NyF, FFE[j]/NyF], 'bandpass')
        data_filt = lfilter(Fb, Fa, Data)
           
        ES[j] = np.log10(np.trapz(np.abs(hilbert(data_filt))))
        KurtoF[j] = kurtosis(data_filt, fisher=False)

    return ES, KurtoF


def get_full_spectrum_stuff(Data,sps):

    NyF = sps / 2.0

    ntr = 1
    MeanFFT = np.empty(ntr, dtype=float)
    MaxFFT = np.empty(ntr, dtype=float)
    FmaxFFT = np.empty(ntr, dtype=float)
    MedianFFT = np.empty(ntr, dtype=float)
    VarFFT = np.empty(ntr, dtype=float)
    FCentroid = np.empty(ntr, dtype=float)
    Fquart1 = np.empty(ntr, dtype=float)
    Fquart3 = np.empty(ntr, dtype=float)
    NpeakFFT = np.empty(ntr, dtype=float)
    MeanPeaksFFT = np.empty(ntr, dtype=float)
    E1FFT = np.empty(ntr, dtype=float)
    E2FFT = np.empty(ntr, dtype=float)
    E3FFT = np.empty(ntr, dtype=float)
    E4FFT = np.empty(ntr, dtype=float)
    gamma1 = np.empty(ntr, dtype=float)
    gamma2 = np.empty(ntr, dtype=float)
    gammas = np.empty(ntr, dtype=float)

    b = np.ones(300) / 300.0

#    for i in range(ntr):
    data = Data
    npts = 2560
    n = nextpow2(2*npts-1)
    Freq1 = np.linspace(0, 1, int(n/2)) * NyF
    
    FFTdata = 2 * np.abs(np.fft.fft(data, n=n)) / float(npts * npts)
    FFTsmooth = lfilter(b, 1, FFTdata[0:int(len(FFTdata)/2)])
    FFTsmooth_norm = FFTsmooth / max(FFTsmooth)
    
    MeanFFT = np.mean(FFTsmooth_norm)
    MedianFFT = np.median(FFTsmooth_norm)
    VarFFT = np.var(FFTsmooth_norm, ddof=1)
    MaxFFT = np.max(FFTsmooth)
    iMaxFFT = np.argmax(FFTsmooth)
    FmaxFFT = Freq1[iMaxFFT]
    
    xCenterFFT = np.sum((np.arange(len(FFTsmooth_norm))) *
                                FFTsmooth_norm) / np.sum(FFTsmooth_norm)
    i_xCenterFFT = int(np.round(xCenterFFT))

    xCenterFFT_1quart = np.sum((np.arange(i_xCenterFFT+1)) *
                                  FFTsmooth_norm[0:i_xCenterFFT+1]) /\
            np.sum(FFTsmooth_norm[0:i_xCenterFFT+1])
    
    i_xCenterFFT_1quart = int(np.round(xCenterFFT_1quart))

    xCenterFFT_3quart = np.sum((np.arange(len(FFTsmooth_norm) -
                                              i_xCenterFFT)) *
                                   FFTsmooth_norm[i_xCenterFFT:]) /\
            np.sum(FFTsmooth_norm[i_xCenterFFT:]) + i_xCenterFFT+1
       
    i_xCenterFFT_3quart = int(np.round(xCenterFFT_3quart))

    FCentroid = Freq1[i_xCenterFFT]
    Fquart1 = Freq1[i_xCenterFFT_1quart]
    Fquart3 = Freq1[i_xCenterFFT_3quart]

    min_peak_height = 0.75
    ipeaks = detect_peaks(FFTsmooth_norm,min_peak_height,100)

    NpeakFFT = len(ipeaks)
    sum_peaks=0
    
    for ll in range(0,len(ipeaks)):
        sum_peaks+=FFTsmooth_norm[ipeaks[ll]]
        
    if NpeakFFT>0:
        MeanPeaksFFT = sum_peaks / float(NpeakFFT)
    else:
        MeanPeaksFFT= 0

    npts = len(FFTsmooth_norm)
    
    E1FFT = np.trapz(FFTsmooth_norm[0:int(npts/4)])
    E2FFT = np.trapz(FFTsmooth_norm[int(npts/4-1):int(2*npts/4)])
    E3FFT = np.trapz(FFTsmooth_norm[int(2*npts/4-1):int(3*npts/4)])
    E4FFT = np.trapz(FFTsmooth_norm[int(3*npts/4-1):int(npts)])

    moment = np.empty(3, dtype=float)

    for j in range(3):
        moment[j] = np.sum(Freq1**j * FFTsmooth_norm[0:int(n/2)]**2)
        
    gamma1 = moment[1]/moment[0]
    gamma2 = np.sqrt(moment[2]/moment[0])
    gammas = np.sqrt(np.abs(gamma1**2 - gamma2**2))

    return MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1,\
        Fquart3, NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1,\
        gamma2, gammas


def get_polarization_stuff(st, env):

    sps = st[0].stats.sampling_rate
    strong_filter = np.ones(int(sps)) / float(sps)
    smooth_env = lfilter(strong_filter, 1, env[0])
    imax = np.argmax(smooth_env)
    end_window = int(np.round(imax/3.))

    xP = st[2].data[0:end_window]
    yP = st[1].data[0:end_window]
    zP = st[0].data[0:end_window]

    MP = np.cov(np.array([xP, yP, zP]))
    w, v = np.linalg.eig(MP)

    indexes = np.argsort(w)
    DP = w[indexes]
    pP = v[:, indexes]

    rectilinP = 1 - ((DP[0] + DP[1]) / (2*DP[2]))
    azimuthP = np.arctan(pP[1, 2] / pP[0, 2]) * 180./np.pi
    dipP = np.arctan(pP[2, 2] / np.sqrt(pP[1, 2]**2 + pP[0, 2]**2)) * 180/np.pi
    Plani = 1 - (2 * DP[0]) / (DP[1] + DP[2])

    return rectilinP, azimuthP, dipP, Plani


def get_pseudo_spectral_stuff(Data, sps):
    
    ntr=1
    SpecKurtoMaxEnv = np.empty(ntr, dtype=float)
    SpecKurtoMedianEnv = np.empty(ntr, dtype=float)
    RATIOENVSPECMAXMEAN = np.empty(ntr, dtype=float)
    RATIOENVSPECMAXMEDIAN = np.empty(ntr, dtype=float)
    DISTMAXMEAN = np.empty(ntr, dtype=float)
    DISTMAXMEDIAN = np.empty(ntr, dtype=float)
    NBRPEAKMAX = np.empty(ntr, dtype=float)
    NBRPEAKMEAN  = np.empty(ntr, dtype=float)
    NBRPEAKMEDIAN = np.empty(ntr, dtype=float)
    RATIONBRPEAKMAXMEAN = np.empty(ntr, dtype=float)
    RATIONBRPEAKMAXMED = np.empty(ntr, dtype=float)
    NBRPEAKFREQCENTER = np.empty(ntr, dtype=float)
    NBRPEAKFREQMAX = np.empty(ntr, dtype=float)
    RATIONBRFREQPEAKS = np.empty(ntr, dtype=float)
    DISTQ2Q1 = np.empty(ntr, dtype=float)
    DISTQ3Q2 = np.empty(ntr, dtype=float)
    DISTQ3Q1 = np.empty(ntr, dtype=float)
    
    # Spectrogram parametrisation
    SpecWindow = 10 # Window legnth
    noverlap = int(0.90 * SpecWindow) # Overlap
    n = 2048 
    Freq=np.linspace(0,sps,int(n/2)) # Sampling of frequency array
    b_filt = np.ones(10) / 10.0 # Smoothing param

    # Spectrogram computation from DFT (Discrete Fourier Transform on a moving window)
    f, t, spec = spectrogram(Data, fs=sps, window='boxcar',
                                     nperseg=SpecWindow, nfft=n, noverlap=noverlap,
                                     scaling='spectrum')
    
    smooth_spec = lfilter(b_filt, 1, np.abs(spec), axis=1) #smoothing

    # Envelope of the maximum of each DFT constituting the spectrogram
    SpecMaxEnv,SpecMaxFreq = smooth_spec[0:800,:].max(0),smooth_spec[0:800,:].argmax(0)
    
    # Envelope of the mean of each DFT constituting the spectrogram
    SpecMeanEnv=smooth_spec.mean(0)
    
    # Envelope of the median of each DFT constituting the spectrogram
    SpecMedianEnv=np.median(smooth_spec,0)
    
    # Envelope of different quartiles of each DFT
    CentoiX=np.empty(np.size(smooth_spec,1), dtype=float)
    CentoiX1=np.empty(np.size(smooth_spec,1), dtype=float)
    CentoiX3=np.empty(np.size(smooth_spec,1), dtype=float)
    
    # Envelope of the frequencies corresponding to different quartiles of the DFT
    for v in range(0,np.size(smooth_spec,1)):
        CentroIndex=np.around(centeroidnpX(smooth_spec[0:800,v]))
        CentoiX[v]=(Freq[int(CentroIndex)])
        CentoiX1[v]=(Freq[int(np.around(centeroidnpX(smooth_spec[0:int(CentroIndex),v])))])
        CentoiX3[v]=(Freq[int(np.around(centeroidnpX(smooth_spec[int(CentroIndex):800,v])+CentroIndex))])
        
    # Tranform into single values
    SpecKurtoMaxEnv=kurtosis(SpecMaxEnv / SpecMaxEnv.max(axis=0))
    SpecKurtoMedianEnv=kurtosis(SpecMedianEnv / SpecMedianEnv.max(axis=0))
    RATIOENVSPECMAXMEAN = np.mean(SpecMaxEnv / SpecMeanEnv)
    RATIOENVSPECMAXMEDIAN = np.mean(SpecMaxEnv / SpecMedianEnv)
    DISTMAXMEAN = np.mean(np.abs(SpecMaxEnv - SpecMeanEnv))
    DISTMAXMEDIAN = np.mean(np.abs(SpecMaxEnv - SpecMedianEnv))
    NBRPEAKMAX = len(detect_peaks(SpecMaxEnv / SpecMaxEnv.max(axis=0),0.75))
    NBRPEAKMEAN  = len(detect_peaks(SpecMeanEnv / SpecMeanEnv.max(axis=0),0.75))
    NBRPEAKMEDIAN = len(detect_peaks(SpecMedianEnv / SpecMedianEnv.max(axis=0),0.75))
    
    if NBRPEAKMEAN>0:
        RATIONBRPEAKMAXMEAN = np.divide(NBRPEAKMAX, NBRPEAKMEAN) 
    else:
        RATIONBRPEAKMAXMEAN=0
        
    if NBRPEAKMEDIAN>0:
        RATIONBRPEAKMAXMED = np.divide(NBRPEAKMAX, NBRPEAKMEDIAN)
    else:
        RATIONBRPEAKMAXMED=0
        
    NBRPEAKFREQCENTER = len(detect_peaks(CentoiX / CentoiX.max(axis=0),0.75))
    NBRPEAKFREQMAX = len(detect_peaks(SpecMaxFreq / SpecMaxFreq.max(axis=0),0.75))
    
    if NBRPEAKFREQCENTER>0:
        RATIONBRFREQPEAKS = NBRPEAKFREQMAX / NBRPEAKFREQCENTER
    else:
        RATIONBRFREQPEAKS=0
        
    DISTQ2Q1 = np.mean(abs(CentoiX-CentoiX1))
    DISTQ3Q2 = np.mean(abs(CentoiX3-CentoiX))
    DISTQ3Q1 = np.mean(abs(CentoiX3-CentoiX1))
                        
    return SpecKurtoMaxEnv, SpecKurtoMedianEnv, RATIOENVSPECMAXMEAN, RATIOENVSPECMAXMEDIAN, \
    DISTMAXMEAN , DISTMAXMEDIAN, NBRPEAKMAX, NBRPEAKMEAN, NBRPEAKMEDIAN, RATIONBRPEAKMAXMEAN, \
    RATIONBRPEAKMAXMED, NBRPEAKFREQCENTER, NBRPEAKFREQMAX, RATIONBRFREQPEAKS, DISTQ2Q1, DISTQ3Q2, DISTQ3Q1


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def l2filter(b, a, x):

    # explicit two-pass filtering with no bells or whistles

    x_01 = lfilter(b, a, x)
    x_02 = lfilter(b, a, x_01[::-1])
    x_02 = x_02[::-1]

def centeroidnpX(arr):
    length = np.arange(1,len(arr)+1)
    CentrX=np.sum(length*arr)/np.sum(arr)
    return CentrX

