#!/usr/bin/env python
#import sys
#sys.path.append('/Users/thompsong/src/kitchensinkGT/LIB')
import os
import numpy as np
from scipy.stats import describe
from scipy.interpolate import interp1d
from obspy import Stream, Trace
from obspy.signal.quality_control import MSEEDMetadata 
#import libseisGT
from libseisGT import add_to_trace_history, clean_trace
import pandas as pd
from math import pi

"""
Functions for computing data quality metrics and statistical metrics (such as amplitude, energy and frequency) 
on Stream/Trace objects.

In terms of order of application:

1. Read the raw data.
2. Fix Trace IDs.
3. Compute QC metrics (and potentially remove bad Traces).
4. Correct the data (and save corrected data as MSEED/StationXML).
5. Compute statistical metrics.

"""

def process_trace(tr, inv=None, quality_threshold=0.0):
# function tr, quality_factor, snr = compute_metrics(tr)
# This function wraps all others
    # Here we compute simple metrics on each trace (and used to write them to NET.STA.CHAN files). 
    # These metrics are:
    #     1. duration of signal
    #     2. signal amplitude
    #     3. noise amplitude
    #     4. signal-to-noise ratio
    
    if not 'history' in tr.stats:
        tr.stats['history'] = list()    
        
    """ RAW DATA QC METRICS """
    try:
        qcTrace(tr)
    except:
        print('qcTrace failed on %s for raw trace' % tr.id)
        tr.stats['quality_factor'] = -1
    else:        
    	tr.stats["quality_factor"] = trace_quality_factor(tr) #0 = blank trace, 1 = has some 0s and -1s, 3 = all looks good
    	tr.stats.quality_factor -= tr.stats.metrics['num_gaps']
    	tr.stats.quality_factor -= tr.stats.metrics['num_overlaps']
    	tr.stats.quality_factor *= tr.stats.metrics['percent_availability']/100.0
    	tr.stats.metrics["twin"] = tr.stats.npts /  tr.stats.sampling_rate # before or after detrending  
    

    if tr.stats.quality_factor > quality_threshold: # only clean traces better than the threshold

        # Check for spikes - been seeing this in 1998 data
        check_for_spikes(tr)

        """ CLEAN (DETREND, BANDPASS, CORRECT) TRACE """
        clean_trace(tr, taperFraction=0.05, filterType="bandpass", freq=[0.5, 30.0], corners=6, zerophase=False, inv=inv)
    
        # Update other stats
        try:
            qcTrace(tr)
        except:
            print('qcTrace failed on %s for cleaned trace' % tr.id)
    return tr.stats.quality_factor

def check_for_spikes(tr):
    if not 'metrics' in tr.stats:
        if not 'history' in tr.stats:
            tr.stats['history'] = list()    
        
        """ RAW DATA QC METRICS """
        try:
            qcTrace(tr)
        except:
            print('qcTrace failed on %s for raw trace' % tr.id)
            tr.stats['quality_factor'] = -1
        else:        
            tr.stats["quality_factor"] = trace_quality_factor(tr) #0 = blank trace, 1 = has some 0s and -1s, 3 = all looks good
            tr.stats.quality_factor -= tr.stats.metrics['num_gaps']
            tr.stats.quality_factor -= tr.stats.metrics['num_overlaps']
            tr.stats.quality_factor *= tr.stats.metrics['percent_availability']/100.0
            tr.stats.metrics["twin"] = tr.stats.npts /  tr.stats.sampling_rate # before or after detrending  
    m = tr.stats.metrics
    peak2peak = m['sample_max']-m['sample_min']
    positive_spike_metric = (m['sample_upper_quartile']-m['sample_min'])/peak2peak
    negative_spike_metric = (m['sample_max']-m['sample_lower_quartile'])/peak2peak
    if positive_spike_metric < 0.01:
        print('Positive spike(s) suspected on %s' % tr.id)
        tr.stats['quality_factor'] = -1
    if negative_spike_metric < 0.01:
        print('Negative spike(s) suspected on %s' % tr.id)  
        tr.stats['quality_factor'] = -1
    
def qcTrace(tr):
    """ qcTrace(tr) DATA QUALITY CHECKS """
    
    """ Useful MSEED metrics
    {'start_gap': None, 'end_gap': None, 'num_gaps': 0, 
     'sum_gaps': 0, 'max_gap': None, 'num_overlaps': 0, 
     'sum_overlaps': 0, 'max_overlap': None, 'quality': 'D', 
     'sample_min': -22404, 'sample_max': 9261, 
     'sample_mean': -3854.7406382978725, 'sample_median': -3836.0, 
     'sample_lower_quartile': -4526.0, 'sample_upper_quartile': -3105.0, 
     'sample_rms': 4426.1431329789848, 
     'sample_stdev': 2175.2511682727431, 
     'percent_availability': 100.0}           
    """
    if len(tr.data)>0:
        tmpfilename = '%s%s.mseed' % (tr.id, tr.stats.starttime.isoformat())
        tr.write(tmpfilename)
        try:
            mseedqc = MSEEDMetadata([tmpfilename]) 
        except:
            pass
        else:
            tr.stats['metrics'] = mseedqc.meta
            os.remove(tmpfilename)
            add_to_trace_history(tr, 'MSEED metrics computed (similar to ISPAQ/MUSTANG).')
    else:
        tr.stats['quality_factor'] = -100

def _detectClipping(tr, countThresh = 10):
    upper_clipped = False
    lower_clipped = False
    y = tr.data
    mu = np.nanmax(y)
    md = np.nanmin(y)
    countu = (tr.data == mu).sum()
    countd = (tr.data == md).sum()
    if countu >= countThresh:
        add_to_trace_history(tr, 'Trace %s appears to be clipped at upper limit %e (count=%d)' % (tr.id, mu, countu) )    
        upper_clipped = True
    if countd >= countThresh:
        add_to_trace_history(tr, 'Trace %s appears to be clipped at lower limit %e (count=%d)' % (tr.id, mu, countu) )       
        lower_clipped = True
    return upper_clipped, lower_clipped

    
def _get_islands(arr, mask):
    mask_ = np.concatenate(( [False], mask, [False] ))
    idx = np.flatnonzero(mask_ [1:] != mask_ [:-1])
    return [arr[idx[i]:idx[i+1] + 1] for i in range(0, len(idx), 2)]

def _FindMaxLength(lst):
    maxList = max(lst, key = len)
    maxLength = max(map(len, lst))      
    return maxList, maxLength  

def trace_quality_factor(tr):
    # trace_quality_factor(tr)
    # a good trace has quality factor 3, one with 0s and -1s has 1, bad trace has 0
    quality_factor = 1.0
    is_bad_trace = False
    
    # ignore traces with few samples
    if tr.stats.npts < 100:
        add_to_trace_history(tr, 'Not enough samples')
        is_bad_trace = True
    
    # ignore traces with weirdly low sampling rates
    if tr.stats.sampling_rate < 19.99:
        add_to_trace_history(tr, 'Sampling rate too low')
        is_bad_trace = True

    # ignore blank trace
    anyData = np.count_nonzero(tr.data)
    if anyData==0:
        add_to_trace_history(tr, 'Trace is blank')
        is_bad_trace = True
    
    # check for bit level noise
    u = np.unique(tr.data)
    num_unique_values = u.size
    if num_unique_values > 10:
        quality_factor += np.log10(num_unique_values)
    else:
        add_to_trace_history(tr, 'bit level noise suspected')
        is_bad_trace = True

    # check for sequences of 0 or 1
    trace_good_flag = _check0andMinus1(tr.data)
    if not trace_good_flag:
        add_to_trace_history(tr, 'sequences of 0 or -1 found')
        is_bad_trace = True
    
    # replacement for check0andMinus1
    seq = tr.data
    islands = _get_islands(seq, np.r_[np.diff(seq) == 0, False]) 
    try:
        maxList, maxLength = _FindMaxLength(islands)
        add_to_trace_history(tr, 'longest flat sequence found: %d samples' % maxLength)
        if maxLength >= tr.stats.sampling_rate:
            is_bad_trace = True
    except:
        is_bad_trace = True
        
    # time to exit?
    if is_bad_trace:
        return 0.0
        
    # check if trace clipped - but so far I don't see clipped trace as terminal
    upperClipped, lowerClipped = _detectClipping(tr) # can add another function to interpolate clipped values
    if upperClipped:
        quality_factor /= 2.0
    if lowerClipped:
        quality_factor /= 2.0
         
    # check for outliers - but I haven't tuned this yet, so not making it a decision between quality 0.0 and continue
    outlier_count, outlier_indices = _mad_based_outlier(tr, thresh=50.0)
    #print('Outliers: %d' % outlier_count)
    if outlier_count == 0:
        quality_factor += 1.0
    else:
        add_to_trace_history(tr, '%d outliers found' % outlier_count)
        tr.stats['outlier_indices'] = outlier_indices    
       
    return quality_factor    

def _mad_based_outlier(tr, thresh=3.5):
    tr2 = tr.copy()
    tr2.detrend()
    points = tr2.data
    if len(points.shape) == 1:
        points = points[:,None]
    #points = np.absolute(points)
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    
    outlier_indices = np.array(np.where(modified_z_score > thresh))
    outlier_count = outlier_indices.size
    '''
    if outlier_count > 0:
        print('size diff = %d, median = %e, med_abs_deviation = %e ' % (diff.size, median, med_abs_deviation))
        mzs = sorted(modified_z_score)
        print(mzs[-10:])
    '''
    
    return outlier_count, outlier_indices


def _check0andMinus1(liste):
# function bool_good_trace = check0andMinus1(tr.data)
    liste=list(liste)
    listStr=''.join(str(i) for i in liste)
    if  "000000000000" in listStr or "-1-1-1-1-1-1-1-1" in listStr :
        return False
    else:
        return True  

    
def signaltonoise(tr):
# function snr, highval, lowval = signaltonoise(tr)
    # Here we just make an estimate of the signal-to-noise ratio
    #
    # Normally the trace should be pre-processed before passing to this routine, e.g.
    # * remove ridiculously large values
    # * remove any sequences of 0 from start or end
    # * detrend
    # * bandpass filter
    #
    # Processing:
    #    1. ensure we have at least 1 seconds
    #    2. take absolute values
    #    3. compute the maximum of each 1-s of data, call this time series M
    #    4. compute 95th and 5th percentile of M, call these M95 and M5
    #    5. estimate signal-to-noise ratio as M95/M5
    #
    # Correction. The above seems like a poor algorithm. Why not instead just take the ratio of the amplitudes of 
    # the "loudest" second and the "quietest" second.
    
    highval = 0.0
    lowval = 0.0
    snr = 0.0
    a = tr.data

    fsamp = int(tr.stats.sampling_rate)
    npts = tr.stats.npts
    numseconds = int(npts/fsamp)
    if numseconds > 1:
        a = a[0:int(fsamp * numseconds - 1)]             # remove any fractional second from end of trace
        abs = np.absolute(a)                             # take absolute value of a        
        abs_resize = np.resize(abs, (fsamp, numseconds)) # resize so that each second is one row
        M = np.max(abs_resize,axis=0)                    # find max of each second / row
        """ old algorithm
        highval = np.nanpercentile(M,95)                    # set highval to 95th percentile, to represent signal amplitude
        lowval = np.nanpercentile(M,5)                      # set lowval to 5th percentile, to represent noise amplitude
        """
        # new algorithm
        highval = np.max(M)
        lowval = np.min(M)
        snr = highval / lowval
    tr.stats.metrics['snr'] = snr
    tr.stats.metrics['signal_level'] = highval
    tr.stats.metrics['noise_level'] = lowval


def choose_best_traces(st, MAX_TRACES=8, include_seismic=True, include_infrasound=False, include_uncorrected=False):

    priority = np.array([float(tr.stats.quality_factor) for tr in st])      
    for i, tr in enumerate(st):           
        if tr.stats.channel[1]=='H':
            if include_seismic:
                if tr.stats.channel[2] == 'Z':
                    priority[i] *= 2
            else:
                priority[i] = 0
        if tr.stats.channel[1]=='D':
            if include_infrasound:
                priority[i] *= 2 
            else:
                priority[i] = 0
        if not include_uncorrected:
            if 'units' in tr.stats:
                if tr.stats.units == 'Counts':
                    priority[i] = 0
            else:
                priority[i] = 0

    n = np.count_nonzero(priority > 0.0)
    n = min([n, MAX_TRACES])
    j = np.argsort(priority)
    chosen = j[-n:]  
    return chosen        
        
def select_by_index_list(st, chosen):
    st2 = Stream()
    for i, tr in enumerate(st):
        if i in chosen:
            st2.append(tr)
    return st2


def _ssam(tr, f, S):
    if not f.size==S.size:
        return
    # use actual amplitude, not dB. 
    ssamValues = []
    #tr.stats['spectral_amplitude'] = S
    #tr.stats['spectral_frequencies'] = f
    freqs = np.arange(0.0, 16.0, 1.0)
    for fmin in freqs:
        f_indexes = np.intersect1d(np.where(f>=fmin), np.where(f<fmin+1.0))
        S_selected = S[f_indexes]
        ssamValues.append(np.nanmean(S_selected) )
    tr.stats['ssam'] = {'f': freqs, 'A': ssamValues}

    
def _band_ratio(tr, freqlims = [1, 6, 11]):    
    # compute band ratio as log2(amplitude > 6 Hz/amplitude < 6 Hz)
    # After Rodgers et al., 2015: https://doi.org/10.1016/j.jvolgeores.2014.11.012
    A = None
    if 'ssam' in tr.stats:
        A = np.array(tr.stats.ssam.A)
        f = tr.stats.ssam.f
    elif 'spectrum' in tr.stats:
        f = tr.stats.spectrum['F']
        A = tr.stats.spectrum['A']
    if len(A)>0:
        f_indexes_low = np.intersect1d(np.where(f>freqlims[0]), np.where(f<freqlims[1]))
        f_indexes_high = np.intersect1d(np.where(f>freqlims[1]), np.where(f<freqlims[2]))       
        A_low = A[f_indexes_low]        
        A_high = A[f_indexes_high] 
        bandratio = {}
        bandratio['freqlims'] = freqlims
        bandratio['RSAM_high'] = sum(A_high)
        bandratio['RSAM_low'] = sum(A_low)
        bandratio['RSAM_ratio'] = np.log2(sum(A_high)/sum(A_low))
        if not 'bandratio' in tr.stats:
            tr.stats.bandratio = []
        tr.stats.bandratio.append(bandratio)
        

        
def ampeng(tr):
    if not 'detrended' in tr.stats.history:
        tr.detrend(type='linear')
        add_to_trace_history(tr, 'detrended')    
    y = np.absolute(tr.data)
    maxy = y.max()
    maxy_i = y.argmax()
    maxy_t = maxy_i / tr.stats.sampling_rate
    stationEng = np.sum(np.square(y)) / tr.stats.sampling_rate
    return maxy_t, maxy, stationEng
    
def ampengfft(tr, outdir=None):
    """ 
    Measure peakamp and energy on a Trace and add to tr.stats.metrics
    Call ssam to add ssam dict to tr.stats
    
    TO DO:
            # 1. For continuous data, I will want to break into 1-minute chunks before computing spectra and metrics.
            # 2. compute RSAM? Where is my code from New Zealand / ObsPy core? Or I could just do RSAM in two frequency bands, the same ones used in band_ratio calculation. 
    
    """
    
    # if we use absolute function, we don't need to check if maxy > -miny
    if not 'detrended' in tr.stats.history:
        tr.detrend(type='linear')
        add_to_trace_history(tr, 'detrended')
        
    if not 'metrics' in tr.stats:
        tr.stats.metrics = {}
        
    maxy_t, maxy, stationEng = ampeng(tr)    
    tr.stats.metrics['peakamp'] = maxy
    tr.stats.metrics['peaktime'] = maxy_t
    tr.stats.metrics['energy'] = stationEng

    # estimate signal-to-noise ratio (after detrending)
    signaltonoise(tr)    
    add_to_trace_history(tr, 'Signal to noise measured and added to tr.stats.metrics.')
    
    # SciPy stats
    scipystats = describe(tr.data, nan_policy = 'omit')._asdict()
    for item in ['skewness', 'kurtosis']:
        tr.stats.metrics[item]=scipystats[item]
    add_to_trace_history(tr, 'scipy.stats metrics added to tr.stats.metrics.')
    
    # add spectral data
    if 'spectrum' in tr.stats:
        _ssam(tr, tr.stats.spectrum['F'], tr.stats.spectrum['A'])
        _band_ratio(tr, freqlims = [1.0, 6.0, 11.0])
        _band_ratio(tr, freqlims = [0.8, 4.0, 16.0]) 
        if outdir:
            _save_esam(tr, outdir)
        add_to_trace_history(tr, 'ampengfft')
    else:
        print("""No tr.stats.spectrum. Cannot compute SSAM data. You need to use:
                        iwsobj = IceWeb.icewebSpectrogram(stream=st)
                        iwsobj = iwsobj.precompute()
                        iwsobj.compute_amplitude_spectrum(compute_bandwidth=True)""")
        add_to_trace_history(tr, 'ampeng')


    # To do:
    # an alternative version might be to plot the sum of spectral data by event type per day,
    # then plot those sums as an event spectrogram - against a daily sampling rate.
    # 
    # from either version could then pick the biggest events       

def _save_esam(tr, outdir):
    """ interpolate F and A to a fixed F2 vector. """
    F = tr.stats.spectrum['F']
    A = tr.stats.spectrum['A']
    
    # interpolate onto regular frequency vector
    # actual frequency interval is 0.09765625 for 100 Hz data with 512 NFFT
    F2 = np.arange(0.0, 20.0, 0.1)
    fn = interp1d(F, A)
    A2 = fn(F2)
    
    # save to file    
    esamfile = tr.stats.starttime.strftime(os.path.join(outdir,'ESAM%Y%m%d.csv'))
    if not os.path.exists(esamfile):
        fptr = open(esamfile, 'w')
        columns = ['id', 'time']    
        for thisF in F2:
            columns.append(str(thisF))
        sep = ','
        fptr.write( sep.join(columns)+'\n' )        
    else:   
        fptr = open(esamfile, 'a')

    fptr.write('%s, %s, ' % (tr.id, tr.stats.starttime.__str__())  )
    for thisA in A2:
        fptr.write('%.2e, ' % thisA)
    fptr.write('\n')
    fptr.close()
    


#### METRICS BELOW ARE NOT ADDED TO TRACE OBJECTS        

def max_3c(st):
    """ max of a 3-component seismogram """
    N = len(st)/3
    m = []

    if N.is_integer():
        st.detrend()
        for c in range(int(N)):
            y1 = st[c*3+0].data
            y2 = st[c*3+1].data
            y3 = st[c*3+2].data
            y = np.sqrt(np.square(y1) + np.square(y2) + np.square(y3))
            m.append(max(y))
    return m 

def peak_amplitudes(st):   
    """ Peak Ground Motion. Should rename it peakGroundMotion """
    
    seismic1d_list = []
    seismic3d_list = []
    infrasound_list = []
    
    #ls.clean_stream(st, taper_fraction=0.05, freqmin=0.05, causal=True)
               
    # velocity, displacement, acceleration seismograms
    stV = st.select(channel='[ESBH]H?') 
    stD = stV.copy().integrate()
    for tr in stD:
        add_to_trace_history(tr, 'integrated')    
    stA = stV.copy().differentiate()
    for tr in stA:
        add_to_trace_history(tr, 'differentiated') 
     
    # Seismic vector data  
    stZ = stV.select(channel="[ESBH]HZ")
    for tr in stZ:
        thisID = tr.id[:-1]
        st3cV = stV.select(id = '%s[ENZ12RT]' % thisID)
        if len(st3cV)==3:
            st3cD = stD.select(id = '%s[ENZ12RT]' % thisID)
            st3cA = stA.select(id = '%s[ENZ12RT]' % thisID)
            md = ls.max_3c(st3cD)
            mv = ls.max_3c(st3cV)
            ma = ls.max_3c(st3cA)
            d = {'traceID':thisID, 'PGD':md[0], 'PGV':mv[0], 'PGA':ma[0], 'calib':tr.stats.calib, 'units':tr.stats.units}
            seismic3d_list.append(d)              
    seismic3d = pd.DataFrame(seismic3d_list)
    
    # Seismic 1-c data
    peakseismo1cfile = os.path.join(eventdir, 'summary_seismic_1c.csv')
    for c in range(len(stV)):
        md = max(abs(stD[c].data))
        mv = max(abs(stV[c].data))
        ma = max(abs(stA[c].data))  
        d = {'traceID':stV[c].id, 'PGD':md[0], 'PGV':mv[0], 'PGA':ma[0], 'calib':stV[c].stats.calib, 'units':stV[c].stats.units}
        seismic1d_list.append(d)    
    seismic1d = pd.DataFrame(seismic1d_list)        
            
    # Infrasound data
    peakinfrafile = os.path.join(eventdir, 'summary_infrasound.csv')
    stP = st.select(channel="[ESBH]D?")
    stPF = stP.copy().filter('bandpass', freqmin=1.0, freqmax=20.0, corners=2, zerophase=True)    
    for c in range(len(stP)):
        mb = max(abs(stP[c].data))
        mn = max(abs(stPF[c].data)) 
        d = {'traceID':stP[c].id, 'PP':mb[0], 'PPF':mn[0], 'calib':stP[c].stats.calib, 'units':stP[c].stats.units}
        infrasound_list.append(d)  
    infrasound = pd.DataFrame(infrasound_list)    
    
    return (seismic3d, seismic1d, infrasound)



def eventStatistics(st):
    # create an empty list of dictionaries
    list_of_dicts = []

    # for each trace, add a new dictionary to the list
    for tr in st:
        thisrow = dict()  # each dictionary will become a row of a dataframe

        # if we use absolute function, we don't need to check if maxy > -miny
        tr.detrend()
        y = np.absolute(tr.data)

        # we did this before
        maxy = y.max()
        maxy_i = y.argmax()
        maxy_t = tr.stats.starttime + maxy_i / tr.stats.sampling_rate

        # add new elements to dictionary
        thisrow['id'] = tr.id
        thisrow['sample'] = maxy_i
        thisrow['time'] = maxy_t
        thisrow['peakamp'] = maxy

        # add a new measurement: energy
        thisrow['energy'] = np.sum(np.square(y)) / tr.stats.sampling_rate

        # add row (dict) to list
        list_of_dicts.append(thisrow)

    # Convert list of dicts to dataframe
    df = pd.DataFrame.from_dict(list_of_dicts)

    return df

def Eseismic2magnitude(Eseismic, correction=3.7):
    # after equation 7 in Hanks and Kanamori 1979, where moment is substitute with energy
    # energy in Joules rather than ergs, so correction is 3.7 rather than 10.7
    if isinstance(Eseismic, list): # list of stationEnergy
        mag = [] 
        for thisE in Eseismic:
            mag.append(Eseismic2magnitude(thisE, correction=correction))
    else:
        mag = np.log10(Eseismic)/1.5 - correction
    return mag

def magnitude2Eseismic(mag, correction=3.7):
    # after equation 7 in Hanks and Kanamori 1979, where moment is substitute with energy
    # energy in Joules rather than ergs, so correction is 3.7 rather than 10.7   
    if isinstance(mag, list): # list of stationEnergy
        Eseismic = [] 
        for thismag in mag:
            Eseismic.append(magnitude2Eseismic(thismag, correction=correction))
    else:
        Eseismic = np.power(10, 1.5 * mag + correction)
    return Eseismic

def Mlrichter(val, R, a=1.6, b=-0.15, g=0):
    """
    MLRICHTER Compute a Richter local magnitude
      ml=Mlrichter(peakamp, R, a, b, g)

       peakamp = maximum amplitude measured from seismic trace
       R = distance from earthquake to station (in km)
       this Ml formula only for stations 0.1-200 km from quake
   
       equation is:
           ml = log10(peakamp) + a * log10(R) + b;

       from Lahr hypoellipse manual Chapter 4.2, defaults are a=1.6 & b=-0.15
       this also agrees with Wikipedia for distances less than 200 km
   
       g is station correction

       Note: I erroneously had b=0.15 when running Seisan 2 Pandas, so my values are too high by 0.3
       But I need to work out real a and b values for Montserrat, and determine station corrections, g
    """
    if isinstance(val,Stream): # Stream
        ml = []
        for tr in val:
            ml.append(Mlrichter(tr, R, a=a, b=b, g=g))
    elif isinstance(val,Trace): # Trace
        peaktime, peakamp, stationEnergy = ampeng(tr)
        ml = Mlrichter(peakamp, R, a=a, b=b, g=g)
    elif isinstance(val, list): # list of stationEnergy
        ml = []
        for thisval in val:
            ml.append(Mlrichter(tr, R, a=a, b=b, g=g))
    else: 
        peakamp = val
        ml = np.log10(peakamp) + a * np.log10(R) + b + g; 
        
    return ml

def Eseismic_Boatwright(val, R, rho_earth=2000, c_earth=2500, S=1.0, A=1.0):
    # val can be a Stream, Trace, a stationEnergy or a list of stationEnergy
    # R in m
    # Following values assumed by Johnson and Aster, 2005:
    # rho_earth 2000 kg/m^3
    # c_earth 2500 m/s
    # A is attenuation = 1
    # S is site response = 1
    #
    # These equations seem to be valid for body waves only, that spread like hemispherical waves in a flat earth.
    # But if surface waves dominate, they would spread like ripples on a pond, so energy density of wavefront like 2*pi*R
    if isinstance(val,Stream): # Stream
        Eseismic = []
        for tr in val:
            Eseismic.append(Eseismic_Boatwright(tr, R, rho_earth, c_earth, S, A))
    elif isinstance(val,Trace): # Trace
        stationEnergy = compute_stationEnergy(val) 
        Eseismic = Eseismic_Boatwright(stationEnergy, R, rho_earth, c_earth, S, A)
    elif isinstance(val, list): # list of stationEnergy
        Eseismic = []
        for thisval in val:
            Eseismic.append(Eseismic_Boatwright(thisval, R, rho_earth, c_earth, S, A))
    else: # stationEnergy
        Eseismic = 2 * pi * (R ** 2) * rho_earth * c_earth * (S ** 2) * val / A 
    return Eseismic

def Eacoustic_Boatwright(val, R, rho_atmos=1.2, c_atmos=340, z=100000):
    # val can be a Stream, Trace, a stationEnergy or a list of stationEnergy
    # R in m
    # Following values assumed by Johnson and Aster, 2005:
    # rho_atmos 1.2 kg/m^3
    # c_atmos 340 m/s  
    # z is just an estimate of the atmospheric vertical scale length - the height of ripples of infrasound energy spreading globally
    if isinstance(val,Stream): # Stream
        Eacoustic = []
        for tr in val:
            Eacoustic.append(Eacoustic_Boatwright(tr, R, rho_atmos, c_atmos))
    elif isinstance(val, Trace): # Trace
        stationEnergy = compute_stationEnergy(val) 
        Eacoustic = Eacoustic_Boatwright(stationEnergy, R, rho_atmos, c_atmos)
    elif isinstance(val, list): # list of stationEnergy
        for thisval in val:
            Eacoustic.append(Eacoustic_Boatwright(thisval, R, rho_atmos, c_atmos, S, A))
    else:
        if R > 100000: # beyond distance z (e.g. 100 km), assume spreading like 2*pi*R
            E_if_station_were_at_z = 2 * pi * (z ** 2) / (rho_atmos * c_atmos) * val
            Eacoustic = E_if_station_were_at_z* R/1e5
        else:
            Eacoustic = 2 * pi * R ** 2 / (rho_atmos * c_atmos) * val
    return Eacoustic
    
def VASR(Eacoustic, Eseismic):
    # From Johnson and Aster 2005
    eta = Eacoustic / Eseismic
    return eta

def compute_stationEnergy(val):
    # seismic: eng is sum of velocity trace (in m/s) squared, divided by samples per second
    # infrasound: eng is sum of pressure trace (in Pa) squared, divided by samples per second
    if isinstance(val, Stream):
        stationEnergy =[]
        for tr in val:
            stationEnergy.append(compute_stationEnergy(tr))
    if isinstance(val, Trace):
        tr2 = val.copy()
        tr2.detrend()
        y = tr2.data
        stationEnergy = np.sum(y ** 2)*tr2.stats.delta
    return stationEnergy

def attenuation(tr, R, Q=50, c_earth=2500):
    s = tr.stats
    if 'spectrum' in s: 
        peakF = s['spectrum']['peakF']
        exponent = - ((pi) * peakF * R) / (c_earth * Q)
        A = np.exp(exponent)
        return A
    else:
        return None


       
