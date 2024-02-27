#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import glob
import pandas as pd
import obspy
from obspy.core import Stream, read 
import numpy as np
from obspy.core.utcdatetime import UTCDateTime
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import datetime
#from sys import exit
from obspy.signal.trigger import z_detect, trigger_onset
from obspy.geodetics.base import gps2dist_azimuth, kilometers2degrees
from obspy.taup import TauPyModel


# Glenn Thompson, Feb 2021

#######################################################################
##                Trace  tools                                       ##
#######################################################################

def add_to_trace_history(tr, str):
    if not 'history' in tr.stats:
        tr.stats['history'] = list()
    if not str in tr.stats.history:
        tr.stats.history.append(str)

def update_trace_filter(tr, filtertype, freq, zerophase):
    if not filter in tr.stats:
        tr.stats['filter'] = {'freqmin':0, 'freqmax':tr.stats.sampling_rate/2, 'zerophase': False}
    if filtertype == 'highpass':    
        tr.stats.filter["freqmin"] = max([freq, tr.stats.filter["freqmin"]])
    if filtertype == 'bandpass':
        tr.stats.filter["freqmin"] = max([freq[0], tr.stats.filter["freqmin"]]) 
        tr.stats.filter["freqmax"] = min([freq[1], tr.stats.filter["freqmax"]])
    if filtertype == 'lowpass':
        tr.stats.filter["freqmax"] = min([freq, tr.stats.filter["freqmax"]])
    tr.stats.filter['zerophase'] = zerophase

def clip_trace(tr, AMP_LIMIT = 10000000):     
# function tr = clip_trace(tr, maxamp)
    # remove absurdly large values
    a = tr.data
    np.clip(a, -AMP_LIMIT, AMP_LIMIT, out=a)
    np.where(a == AMP_LIMIT, 0, a)
    tr.data = a    
    
def smart_merge_traces(trace_pair):
    """
    Clever way to merge overlapping traces. Uses all non-zero data values from both.
    """
    this_tr = trace_pair[0] 
    other_tr = trace_pair[1]

    error_flag = False


    if not (this_tr.id == other_tr.id):
        print('Different trace IDs. Cannot merge.')
        error_flag = True

    if not (this_tr.stats.sampling_rate == other_tr.stats.sampling_rate):
        print('Different sampling rates. Cannot merge.')
        error_flag = True

    if (abs(this_tr.stats.starttime - other_tr.stats.starttime) > this_tr.stats.delta/4):
        print('Different start times. Cannot merge.')
        error_flag = True

    if (abs(this_tr.stats.endtime - other_tr.stats.endtime) > this_tr.stats.delta/4):
        print('Different end times. Cannot merge.')
        error_flag = True

    if error_flag: # traces incompatible, so return the trace with the most non-zero values
        this_good = np.count_nonzero(this_tr.data)
        #print(this_tr.stats)
        other_good = np.count_nonzero(other_tr.data)
        #print(other_tr.stats)
        if other_good > this_good:
            return other_tr
        else:
            return this_tr

    else: # things are good
        indices = np.where(other_tr.data == 0)
        other_tr.data[indices] = this_tr.data[indices]
        return other_tr
        
def pad_trace(tr, seconds):
    if seconds>0.0:
        y = tr.data
        tr.stats['originalStartTime'] = tr.stats.starttime
        tr.stats['originalEndTime'] = tr.stats.endtime
        npts_pad = int(tr.stats.sampling_rate * seconds)
        y_prepend = np.flip(y[0:npts_pad])
        y_postpend = np.flip(y[-npts_pad:])
        y = np.concatenate( [y_prepend, y, y_postpend ] )
        padStartTime = tr.stats.starttime - npts_pad * tr.stats.delta
        tr.data = y
        add_to_trace_history(tr, 'padded')
        tr.stats.starttime = padStartTime

def unpad_trace(tr):
    s = tr.stats
    if 'originalStartTime' in s:
        tr.trim(starttime=s.originalStartTime, endtime=s.originalEndTime, pad=False)
        add_to_trace_history(tr, 'unpadded') 
    
        
def clean_trace(tr, taperFraction=0.05, filterType="bandpass", freq=[0.1, 20.0], corners=2, zerophase=True, inv=None):
    """
    Clean Trace object in place.
    clean_trace(tr, taperFraction=0.05, filterType="bandpass", freq=[0.1, 20.0], corners=2, zerophase=True, inv=None)
    """

    if not 'history' in tr.stats:
        tr.stats['history'] = list()    
    
    # remove absurd values
    clip_trace(tr) # could add function here to correct for clipping - algorithms exist
    
    # save the start and end times for later 
    startTime = tr.stats.starttime
    endTime = tr.stats.endtime
    #print(tr.id, startTime, endTime)
    
    # get trace max and min
    #amp_before = (max(tr.data)-min(tr.data))/2
    #stdev_before = np.std(tr.data)
        
    # pad the Trace
    #y = tr.data
    npts = tr.stats.npts
    npts_pad = int(taperFraction * npts)
    npts_pad_seconds = npts_pad * tr.stats.delta
    if npts_pad_seconds < 10.0: # impose a minimum pad length of 10-seconds
        #npts_pad = int(10.0 / tr.stats.delta)
        npts_pad_seconds = 10.0
    """
    y_prepend = np.flip(y[0:npts_pad])
    y_postpend = np.flip(y[-npts_pad:])
    y = np.concatenate( [y_prepend, y, y_postpend ] )
    padStartTime = startTime - npts_pad * tr.stats.delta
    tr.data = y
    add_to_trace_history(tr, 'padded')
    tr.stats.starttime = padStartTime
    """
    
    pad_trace(tr, npts_pad_seconds)
    max_fraction = npts_pad / tr.stats.npts
    
    # clean
    if not 'detrended' in tr.stats.history:
        tr.detrend('linear')
        add_to_trace_history(tr, 'detrended')
        
    if not 'tapered' in tr.stats.history:
        tr.taper(max_percentage=max_fraction, type="hann") 
        add_to_trace_history(tr, 'tapered')        
    
    if filterType == 'bandpass':
        tr.filter(filterType, freqmin=freq[0], freqmax=freq[1], corners=corners, zerophase=zerophase)
    else:    
        tr.filter(filterType, freq=freq, corners=corners, zerophase=zerophase)
    update_trace_filter(tr, filterType, freq, zerophase)
    add_to_trace_history(tr, filterType)    
        

    '''
    if not 'deconvolved' in tr.stats.history:
        if not 'units' in tr.stats:
            tr.stats['units'] = 'Counts'   
    if inv and tr.stats['units'] == 'Counts':
        tr.remove_response(inventory=inv, output="VEL") 
        tr.stats['units'] = 'm/s'
        add_to_trace_history(tr, 'deconvolved')
    if not inv and tr.stats.calib and not 'calibrated' in tr.stats.history:
        if not tr.stats.calib==1.0:
            tr.data = tr.data * tr.stats.calib
            tr.stats['units'] = 'm/s'
            add_to_trace_history(tr, 'calibrated')
          
    '''
    
    if not 'units' in tr.stats:
        tr.stats['units'] = 'Counts'   
        
    if tr.stats['units'] == 'Counts' and not 'calibrated' in tr.stats.history:
        if inv:
            try:
                tr.remove_response(inventory=inv)
            except:
                print('No matching response info found for %s' % tr.id)
            else:
                add_to_trace_history(tr, 'calibrated')
                # update the calib value
                tr.stats.calib = _get_calib(tr, inv)
                
        elif not tr.stats.calib==1.0:
            tr.data = tr.data * tr.stats.calib
            add_to_trace_history(tr, 'calibrated') 
        if 'calibrated' in tr.stats.history:           
            if tr.stats.channel[1]=='H':
                tr.stats['units'] = 'm/s'
            if tr.stats.channel[1]=='N':
                tr.stats['units'] = 'm/s2'                
            if tr.stats.channel[1]=='D':
                tr.stats['units'] = 'Pa'  
            
    # remove the pad
    #tr.trim(starttime=startTime, endtime=endTime, pad=False)
    #add_to_trace_history(tr, 'unpadded')
    unpad_trace(tr)
    
    #amp_after = (max(tr.data)-min(tr.data))/2
    #stdev_after = np.std(tr.data)
    #tr.stats.calib = amp_after / amp_before
    #tr.stats.calib = stdev_before / stdev_after
    #if 'calibrated' in tr.stats.history:
    #    tr.stats.calib = amp_after / amp_before
                        
def _get_calib(tr, this_inv):
    calib = 1.0
    for station in this_inv.networks[0].stations:
        if station.code == tr.stats.station:
            for channel in station.channels:
                if channel.code == tr.stats.channel:
                    calib_freq, calib_value = channel.response._get_overall_sensitivity_and_gain()
    return calib_value
        
        
    
#######################################################################
##                Stream tools                                       ##
#######################################################################

def get_seed_band_code(sr, shortperiod = False):
    bc = '_'
    if sr >= 1 and sr < 10:
        bc = 'M'
    if sr >= 10 and sr < 80:
        if shortperiod:
            bc = 'S'
        else:
            bc = 'B'
    if sr >= 80:
        if shortperiod:
            bc = 'E'
        else:
            bc = 'H'
    return bc
        
def fix_seed_band_code(st, shortperiod = False):
    for tr in st:
        sr = tr.stats.sampling_rate
        bc = libseisGT.get_seed_band_code(sr, shortperiod=shortperiod)
        if not bc == tr.stats.channel[0]:
            tr.stats.channel = bc + tr.stats.channel[1:]
            add_to_trace_history(tr, 'bandcode_fixed') 


                  
def smart_merge(st):
    # need to loop over st and find traces with same ids
    ##### GOT HERE
    newst = Stream()
    all_ids = []
    for tr in st:
        if not tr.id in all_ids:
            all_ids.append(tr.id)

    for this_id in all_ids: # loop over all nsl combinations
        these_traces = st.copy().select(id=this_id).sort() # find all traces for this nsl combination
        
        # remove duplicates
        traces_to_remove = []
        for c in range(len(these_traces)-1):
            s0 = these_traces[c].stats
            s1 = these_traces[c].stats
            if s0.starttime == s1.starttime and s0.endtime == s1.endtime and s0.sampling_rate == s1.sampling_rate:
                traces_to_remove.append(c)
                
        print(these_traces)
        print(traces_to_remove)
        if traces_to_remove:
            for c in traces_to_remove:
                these_traces.remove(these_traces[c])
                
        if len(these_traces)==1: # if only 1 trace, append it, and go to next trace id
            newst.append(these_traces[0]) 
            continue
        
        # must have more than 1 trace
        try: # try regular merge now duplicates removed
            merged_trace = these_traces.copy().merge()
            print('- regular merge of these traces success')
        except:
            print('- regular merge of these traces failed')   
            # need to try merging traces in pairs instead
            N = len(these_traces)
            these_traces.sort() # sort the traces
            for c in range(N-1): # loop over traces in pairs
                appended = False
                
                # choose pair
                if c==0:
                    trace_pair = these_traces[0:2]
                else:
                    trace_pair = Stream(traces=[merged_trace, these_traces[c+1] ] )
                                        
                # merge these two traces together    
                try: # standard merge
                    merged_trace = trace_pair.copy().merge()
                    print('- regular merge of trace pair success')
                except: # smart merge
                    print('- regular merge of trace pair failed')
                    try:
                        min_stime, max_stime, min_etime, max_etime = ls.Stream_min_starttime(trace_pair)
                        trace_pair.trim(starttime=min_stime, endtime=max_etime, pad=True, fill_value=0)
                        merged_trace = Stream.append(smart_merge_traces(trace_pair)) # this is a trace, not a Stream
                    except:
                        print('- smart_merge of trace pair failed')
                        
            # we have looped over all pairs and merged_trace should now contain everything
            # we should only have 1 trace in merged_trace
            print(merged_trace)
            if len(merged_trace)==1:
                try:
                    newst.append(merged_trace[0])
                    appended = True
                except:
                    pass
                
            if not appended:
                print('\n\nTrace conflict\n')
                trace_pair.plot()
                for c in range(len(trace_pair)):
                    print(c, trace_pair[c])
                choice = int(input('Keep which trace ? '))
                newst.append(trace_pair[choice])  
                appended = True                
                             
    return newst 
        
        
def Stream_min_starttime(all_traces):
    """
    Take a Stream object, and return the minimum starttime

    Created for CALIPSO data archive from Alan Linde.
    """ 

    min_stime = UTCDateTime(2099, 12, 31, 0, 0, 0.0)
    max_stime = UTCDateTime(1900, 1, 1, 0, 0, 0.0)
    min_etime = UTCDateTime(2099, 12, 31, 0, 0, 0.0)
    max_etime = UTCDateTime(1900, 1, 1, 0, 0, 0.0)    
    for this_tr in all_traces:
        if this_tr.stats.starttime < min_stime:
            min_stime = this_tr.stats.starttime
        if this_tr.stats.starttime > max_stime:
            max_stime = this_tr.stats.starttime  
        if this_tr.stats.endtime < min_etime:
            min_etime = this_tr.stats.endtime
        if this_tr.stats.endtime > max_etime:
            max_etime = this_tr.stats.endtime              
    return min_stime, max_stime, min_etime, max_etime


def removeInstrumentResponse(st, preFilter = (1, 1.5, 30.0, 45.0), outputType = "VEL", inventory = None):  
    """
    Remove instrument response - assumes inventories have been added to Stream object
    Written for Miami Lakes
    
    This function may be obsolete. Use clean_trace instead.
    """
    try:
        st.remove_response(output=outputType, pre_filt=preFilter)
    except:
        for tr in st:
            try:
                if inventory:
                    tr.remove_response(output=outputType, pre_filt=preFilter, inventory=inventory)
                else:
                    tr.remove_response(output=outputType, pre_filt=preFilter)
            except:
                print("- Not able to correct data for %s " %  tr.id)
                st.remove(tr)
    return

def detect_network_event(st, minchans=None, threshon=3.5, threshoff=1.0, sta=0.5, lta=5.0, pad=0.0):
    """
    Run a full network event detector/associator 
    
    Note that if you run a 5-s LTA, you need at least 5-s of noise before the signal.
    
    Output is a list of dicts like:
    
    {'cft_peak_wmean': 19.561900329259956,
 'cft_peaks': [19.535644192544272,
               19.872432918501264,
               19.622171410201297,
               19.217352795792998],
 'cft_std_wmean': 5.4565629691954713,
 'cft_stds': [5.292458320417178,
              5.6565387957966404,
              5.7582248973698507,
              5.1190298631982163],
 'coincidence_sum': 4.0,
 'duration': 4.5299999713897705,
 'stations': ['UH3', 'UH2', 'UH1', 'UH4'],
 'time': UTCDateTime(2010, 5, 27, 16, 24, 33, 190000),
 'trace_ids': ['BW.UH3..SHZ', 'BW.UH2..SHZ', 'BW.UH1..SHZ', 'BW.UH4..SHZ']}
 
 
    Any trimming of the Stream object can then by done with trim_to_event.
 
    """
    from obspy.signal.trigger import coincidence_trigger
    if pad>0.0:
        for tr in st:
            pad_trace(tr, pad)
            
    if not minchans:
        N = len(st)
        minchans = int(N/3)
        if minchans<3:
            minchans=N
    print('minchans=',minchans)
    trig = coincidence_trigger("recstalta", threshon, threshoff, st, minchans, sta=sta, lta=lta, details=True) # 0.5s, 10s
    ontimes = [t['time'] for t in trig]
    offtimes = [t['time']+t['duration'] for t in trig]
    
    if pad>0.0:
        for tr in st:
            unpad_trace(tr)        
    return trig, ontimes, offtimes
    

def add_channel_detections(st, lta=5.0, threshon=0.5, threshoff=0.0, max_duration=120):
    """ 
    Runs a single channel detection on each Trace. No coincidence trigger/association into event.
        
    Take a Stream object and run an STA/LTA on each channel, adding a triggers list to Trace.stats.
    This should be a list of 2-element numpy arrays with trigger times on and off as UTCDateTime
    
    Note that if you run a 5-s LTA, you need at least 5-s of noise before the signal.

    """
    for tr in st:
        tr.stats['triggers'] = []
        Fs = tr.stats.sampling_rate
        cft = z_detect(tr.data, int(lta * Fs)) 
        triggerlist = trigger_onset(cft, threshon, threshoff, max_len = max_duration * Fs)
        for trigpair in triggerlist:
            trigpairUTC = [tr.stats.starttime + samplenum/Fs for samplenum in trigpair]
            tr.stats.triggers.append(trigpairUTC)


def get_event_window(st, pretrig=30, posttrig=30):
    """ 
    Take a Stream object and run an STA/LTA on each channel. Return first trigger on and last trigger off time.
    Assumes that tr.stats has triggers lists added already with add_channel_detections
    Any trimming of the Stream taking into account pretrig and posttrig seconds can by done with trim_to_event
    """

    mintime = []
    maxtime = []
    
    for tr in st:
        if 'triggers' in tr.stats:
            if len(tr.stats.triggers)==0:
                continue
            trigons = [thistrig[0] for thistrig in tr.stats.triggers]
            trigoffs = [thistrig[1] for thistrig in tr.stats.triggers]   
            mintime.append(min(trigons))
            maxtime.append(max(trigoffs))           
    
    N = int(len(mintime)/2)
    if len(mintime)>0:
        return sorted(mintime)[N], sorted(maxtime)[N]
    else:
        return None, None
    

def trim_to_event(st, mintime, maxtime, pretrig=10, posttrig=10):
    """ Trims a Stream based on mintime and maxtime which could come from detect_network_event or get_event_window """
    st.trim(starttime=mintime-pretrig, endtime=maxtime+posttrig)
    
def plot_seismograms(st, outfile=None, bottomlabel=None, ylabels=None):
    """ Create a plot of a Stream object similar to Seisan's mulplt """
    fh = plt.figure(figsize=(8,12))
    
    # get number of stations
    stations = []
    for tr in st:
        stations.append(tr.stats.station)
    stations = list(set(stations))
    n = len(stations)
    
    # start time as a Unix epoch
    startepoch = st[0].stats.starttime.timestamp
    
    # create empty set of subplot handles - without this any change to one affects all
    axh = []
    
    # loop over all stream objects
    colors = 'kgrb'
    channels = 'ZNEF'
    linewidths = [0.25, 0.1, 0.1]
    for i in range(n):
        # add new axes handle for new subplot
        #axh.append(plt.subplot(n, 1, i+1, sharex=ax))
        if i>0:
            #axh.append(plt.subplot(n, 1, i+1, sharex=axh[0]))
            axh.append(fh.add_subplot(n, 1, i+1, sharex=axh[0]))
        else:
            axh.append(fh.add_subplot(n, 1, i+1))
        
        # find all the traces for this station
        this_station = stations[i]
        these_traces = st.copy().select(station=this_station)
        for this_trace in these_traces:
            this_component = this_trace.stats.channel[2]
            line_index = channels.find(this_component)
            #print(this_trace.id, line_index, colors[line_index], linewidths[line_index])
        
            # time vector, t, in seconds since start of record section
            t = np.linspace(this_trace.stats.starttime.timestamp - startepoch,
                this_trace.stats.endtime.timestamp - startepoch,
                this_trace.stats.npts)
            #y = this_trace.data - offset
            y = this_trace.data
            
            # PLOT THE DATA
            axh[i].plot(t, y, linewidth=linewidths[line_index], color=colors[line_index])
            axh[i].autoscale(enable=True, axis='x', tight=True)
   
        # remove yticks because we will add text showing max and offset values
        #axh[i].yaxis.set_ticks([])

        # remove xticklabels for all but the bottom subplot
        if i < n-1:
            axh[i].xaxis.set_ticklabels([])
        else:
            # for the bottom subplot, also add an xlabel with start time
            if bottomlabel:
                plt.xlabel(bottomlabel)
            else:
                plt.xlabel("Starting at %s" % (st[0].stats.starttime) )

        # default ylabel is station.channel
        if ylabels:
            plt.ylabel(ylabels[i])
        else:
            plt.ylabel(this_station, rotation=90)
            
    # change all font sizes
    plt.rcParams.update({'font.size': 8})
    
    plt.subplots_adjust(wspace=0.1)
    
    # show the figure
    if outfile:
        plt.savefig(outfile, bbox_inches='tight')    
    else:
        plt.show()
    
    
def mulplt(st, bottomlabel='', ylabels=[], MAXPANELS=6):
    """ Create a plot of a Stream object similar to Seisan's mulplt """
    fh = plt.figure()
    n = np.min([MAXPANELS, len(st)])
    
    # start time as a Unix epoch
    startepoch = st[0].stats.starttime.timestamp
    
    # create empty set of subplot handles - without this any change to one affects all
    axh = []
    
    # loop over all stream objects
    for i in range(n):
        # add new axes handle for new subplot
        #axh.append(plt.subplot(n, 1, i+1, sharex=ax))
        axh.append(plt.subplot(n, 1, i+1))
        
        # time vector, t, in seconds since start of record section
        t = np.linspace(st[i].stats.starttime.timestamp - startepoch,
            st[i].stats.endtime.timestamp - startepoch,
            st[i].stats.npts)
            
        # We could detrend, but in case of spikes, subtracting the median may be better
        #st[i].detrend()
        offset = np.median(st[i].data)
        y = st[i].data - offset
        
        # PLOT THE DATA
        axh[i].plot(t, y)
   
        # remove yticks because we will add text showing max and offset values
        axh[i].yaxis.set_ticks([])

        # remove xticklabels for all but the bottom subplot
        if i < n-1:
            axh[i].xaxis.set_ticklabels([])
        else:
            # for the bottom subplot, also add an xlabel with start time
            if bottomlabel=='':
                plt.xlabel("Starting at %s" % (st[0].stats.starttime) )
            else:
                plt.xlabel(bottomlabel)

        # default ylabel is station.channel
        if ylabels==[]:
            plt.ylabel(st[i].stats.station + "." + st[i].stats.channel, rotation=0)
        else:
            plt.ylabel(ylabels[i])

        # explicitly give the maximum amplitude and offset(median)
        plt.text(0, 1, "max=%.1e offset=%.1e" % (np.max(np.abs(y)), offset),
            horizontalalignment='left',
            verticalalignment='top',transform=axh[i].transAxes)
            
    # change all font sizes
    plt.rcParams.update({'font.size': 8})
    
    # show the figure
    plt.show()
    #st.mulplt = types.MethodType(mulplt,st)    
    
    return fh, axh
   
   
    
def plot_stream_types(st, eventdir, maxchannels=10): 
    # assumes input traces are in velocity or pressure units and cleaned
      
    # velocity, displacement, acceleration seismograms
    stV = st.copy().select(channel='[ESBH]H?') 
    if len(stV)>maxchannels:
        stV = stV.select(channel="[ESBH]HZ") # just use vertical components then
        if len(stV)>maxchannels:
            stV=stV[0:maxchannels]
    stD = stV.copy().integrate()   
    stA = stV.copy().differentiate()

    # Infrasound data
    stP = st.copy().select(channel="[ESBH]D?")
    if len(stP)>maxchannels:
        stP=stP[0:maxchannels]
        
    # Plot displacement seismogram
    if stD:
        dpngfile = os.path.join(eventdir, 'seismogram_D.png')
        stD.plot(equal_scale=False, outfile=dpngfile)    
    
    # Plot velocity seismogram
    if stV:
        vpngfile = os.path.join(eventdir, 'seismogram_V.png')
        stV.plot(equal_scale=False, outfile=vpngfile)
         
    # Plot acceleration seismogram
    if stA:
        apngfile = os.path.join(eventdir, 'seismogram_A.png')
        stA.plot(equal_scale=False, outfile=apngfile)
        
    # Plot pressure acoustograms
    if stP:
        ppngfile = os.path.join(eventdir, 'seismogram_P.png')
        stP.plot(equal_scale=False, outfile=ppngfile)     
    
    
#######################################################################    
########################         WFDISC tools                        ##
#######################################################################


        
def index_waveformfiles(wffiles):
    """ 
    Take a list of seismic waveform data files and return a dataframe similar to a wfdisc table

    Created for CALIPSO data archive from Alan Linde.
    """

    wfdisc_df = pd.DataFrame()
    traceids = []
    starttimes = []
    endtimes = []
    sampling_rates = []
    calibs = []
    ddirs = []
    dfiles = []
    npts = []
    for wffile in sorted(wffiles):
        dfile = os.path.basename(wffile)
        ddir = os.path.dirname(wffile)
        try:
            this_st = read(wffile)
            print('Read %s\n' % wffile)
        except:
            print('Could not read %s\n' % wffile)
            next
        else:
            for this_tr in this_st:
                r = this_tr.stats
                traceids.append(this_tr.id)
                starttimes.append(r.starttime)
                endtimes.append(r.endtime)
                sampling_rates.append(r.sampling_rate)
                calibs.append(r.calib)
                ddirs.append(ddir)
                dfiles.append(dfile)
                npts.append(r.npts)
    if wffiles:
        wfdisc_dict = {'traceID':traceids, 'starttime':starttimes, 'endtime':endtimes, 'npts':npts, 
                       'sampling_rate':sampling_rates, 'calib':calibs, 'ddir':ddirs, 'dfile':dfiles}
        #print(wfdisc_dict)
        wfdisc_df = pd.DataFrame.from_dict(wfdisc_dict)  
        wfdisc_df.sort_values(['starttime'], ascending=[True], inplace=True)
    return wfdisc_df



def wfdisc_to_BUD(wfdisc_df, TOPDIR, put_away):
    """ 
    Read a wfdisc-like dataframe, read the associated files, and write out as BUD format


    Created for CALIPSO data archive from Alan Linde.
    """

    unique_traceIDs = wfdisc_df['traceID'].unique().tolist()
    print(unique_traceIDs)
    
    successful_wffiles = list()

    for traceID in unique_traceIDs:
        print(traceID)
        
        trace_df = wfdisc_df[wfdisc_df['traceID']==traceID]
        
        # identify earliest start time and latest end time for this channel
        #print(trace_df.iloc[0]['starttime'])
        #print(trace_df.iloc[-1]['endtime'])
        minUTC = trace_df.starttime.min()
        maxUTC = trace_df.endtime.max()
        start_date = minUTC.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = maxUTC.replace(hour=23, minute=59, second=59, microsecond=999999)
        this_date = start_date

        while this_date <= end_date: 
            all_traces = Stream()
        
            # loop from earliest start day to latest end day
            subset_df = trace_df[(trace_df['starttime'] < this_date+86400) & (trace_df['endtime'] >= this_date)]
            #print(subset_df)
            
            if len(subset_df.index)==0:
                next
        
            for index, row in subset_df.iterrows():
                wffile = os.path.join(row['ddir'], row['dfile'])
                start_at = max([this_date, row['starttime']])
                end_at = min([this_date+86400, row['endtime']])
                print('- ',wffile,': START AT:', start_at, ', END AT: ',end_at)
                try:
                    this_st = read(wffile, starttime=start_at, endtime=end_at)

                except:
                    print(' Failed\n')
                    next
                else:
                    print(' Succeeded\n')
                    #raise Exception("Stopping here")
                    if end_at == row['endtime']:
                        successful_wffiles.append(wffile)   
                    for this_tr in this_st:
                        if this_tr.id == traceID:
                            #print(tr.stats)
                            all_traces = all_traces.append(this_tr)
            #print(st.__str__(extended=True)) 
            try:
                all_traces.merge(fill_value=0)
            except:
                print('Failed to merge ', all_traces)
            print(all_traces.__str__(extended=True))
        
            # Check that we really only have a single trace ID before writing the BUD files
            error_flag = False
            for this_tr in all_traces:
                if not this_tr.id == traceID:
                    error_flag = True
            if not error_flag:
                try:
                    Stream_to_BUD(TOPDIR, all_traces)
                except:
                    print('Stream_to_BUD failed for ', all_traces)
            
            this_date += 86400
            
    for wffile in successful_wffiles:
        ddir = os.path.dirname(wffile)
        dbase = "%s.PROCESSED" % os.path.basename(wffile)
        newwffile = os.path.join(ddir, dbase)
        print('move %s %s' % (wffile, newwffile))
        if os.path.exists(wffile) and put_away:
            shutil.move(wffile, newwffile)

            
            
def process_wfdirs(wfdirs, filematch, put_away=False):
    """ 
    Process a directory containing waveform data files in any format readable by ObsPy.
    Build a wfdisc-like dataframe indexing those waveform files.
    Convert them to a BUD archive.

    Created for CALIPSO data archive from Alan Linde.
    """

    for wfdir in wfdirs:
        print('Processing %s' % wfdir)
        wffiles = glob.glob(os.path.join(wfdir, filematch))
        if wffiles:
            #print(wffiles)
            wfdisc_df = index_waveformfiles(wffiles)
            #print(wfdisc_df)
            if not wfdisc_df.empty:
                wfdisc_to_BUD(wfdisc_df, TOPDIR, put_away)  
    print('Done.')



#######################################################################
##                BUD tools                                          ##
#######################################################################



def Stream_to_BUD(TOPDIR, all_traces):
    """ 
    Take a Stream object and write it out in IRIS/PASSCAL BUD format. 
    
    Example:
    
        Stream_to_BUD('RAW', all_traces)
    Where all_traces is a Stream object with traces from 2020346

    Creates a BUD directory structure that looks like:

        DAYS
        ├── BHP2
        │   ├── 1R.BHP2..EH1.2020.346
        │   ├── 1R.BHP2..EH2.2020.346
        │   └── 1R.BHP2..EHZ.2020.346
        ├── BHP4
        │   ├── 1R.BHP4..EH1.2020.346
        │   ├── 1R.BHP4..EH2.2020.346
        │   └── 1R.BHP4..EHZ.2020.346
        ├── FIREP
        │   ├── 1R.FIREP..EH1.2020.346
        │   ├── 1R.FIREP..EH2.2020.346
        │   └── 1R.FIREP..EHZ.2020.346
        └── TANKP
            ├── 1R.TANKP..EH1.2020.346
            ├── 1R.TANKP..EH2.2020.346
            └── 1R.TANKP..EHZ.2020.346
        
    where BHP2, BHP4, FIREP and TANKP are station names, 1R is network name, 
    location is blank, channels are EH[Z12], year is 2020 and day of year is 346.

    Created for ROCKETSEIS data conversion and modified for CALIPSO data archive from Alan Linde.   
    """
    
    all_traces = Stream_to_24H(all_traces)
    
    daysDir = os.path.join(TOPDIR, 'DAYS')

    for this_tr in all_traces:
        YYYY = this_tr.stats.starttime.year
        JJJ = this_tr.stats.starttime.julday
        stationDaysDir = os.path.join(daysDir, this_tr.stats.station)
        if not os.path.exists(stationDaysDir):
            os.makedirs(stationDaysDir)
            #print(stationDaysDir)
        mseedDayBasename = "%s.%04d.%03d" % (this_tr.id, YYYY, JJJ  )
        mseedDayFile = os.path.join(stationDaysDir, mseedDayBasename)
        #print(mseedDayFile)
        if os.path.exists(mseedDayFile):
            this_tr = Trace_merge_with_BUDfile(this_tr, mseedDayFile)

        this_tr.write(mseedDayFile, format='MSEED') 


    
def BUD_load_day(BUDDIR, year, jday):
    """
    Load all files corresponding to this year and day from a BUD archive


    Created for CALIPSO data archive from Alan Linde.
    """

    all_stations = glob.glob(os.path.join(BUDDIR, '*'))
    all_traces = Stream()
    for station_dir in all_stations:
        all_files = glob.glob(os.path.join(station_dir, '*.%04d.%03d' % (year, jday)))
        for this_file in all_files:
            try:
                these_traces = read(this_file)
            except:
                print('Cannot read %s' % this_file)
            else:
                for this_tr in these_traces:
                    all_traces.append(this_tr)
    return all_traces



def Stream_to_dayplot(TOPDIR, all_traces):
    """ 
    Take a Stream object, pad it to 24-hours, plot it, and save to a PNG file. 
    
    Example: 
        Stream_to_dayplot('RAW', all_traces)

    Creates: 
    
        DAYS
        ├── 1R.2020.346.png


    Make sure that all_traces[0] contains full trace-id metadata. 

    Created for ROCKETSEIS project.

    """    

    daysDir = os.path.join(TOPDIR, 'DAYPLOTS')
    os.makedirs(daysDir)
    NETWORK = all_traces[0].stats.network
    stime = all_traces[0].stats.starttime
    YYYY = stime.year
    JJJ = stime.yearday
    pngfile = os.path.join(daysDir, '%s.%s.%s.png' % (NETWORK, YYYYJJJ[0:4], YYYYJJJ[4:])  )   
    all_traces.plot(equal_scale=False, outfile=pngfile);
    return


def Stream_to_24H(all_traces):
    """
    Take a Stream object, merge all traces with common ids and pad out to 24-hour-long traces

    Created for ROCKETSEIS data conversion and modified for CALIPSO data archive from Alan Linde. 
    """

    all_traces.merge(fill_value=0)
    min_stime, max_stime, min_etime, max_etime = Stream_min_starttime(all_traces)
    
    desired_stime = UTCDateTime(min_stime.year, min_stime.month, min_stime.day, 0, 0, 0.0)
    desired_etime = desired_stime + 86400
    
    days = Stream()
    while True:
        
        this_st = all_traces.copy()
        this_st.trim(starttime=desired_stime, endtime=desired_etime, pad=True, fill_value=0)
        for this_tr in this_st:
            days.append(this_tr)
        desired_stime += 86400
        desired_etime += 86400
        if desired_etime > max_etime + 86400:
            break
    return days



def Trace_merge_with_BUDfile(this_tr, budfile):
    """
    Clever way to merge overlapping traces into a BUD file. Uses all non-zero data values from both.

    Created for CALIPSO data archive from Alan Linde, when needed to upgrade Stream_to_BUD.
    """

    other_st = read(budfile)
    error_flag = False
    
    if len(other_st)>1:
        print('More than 1 trace in %s. Cannot merge.' % budfile)
        error_flag = True
        
    other_tr = other_st[0]
    if not (this_tr.id == other_tr.id):
        print('Different trace IDs. Cannot merge.')
        error_flag = True
        
    if not (this_tr.stats.sampling_rate == other_tr.stats.sampling_rate):
        print('Different sampling rates. Cannot merge.')
        error_flag = True
        
    if (abs(this_tr.stats.starttime - other_tr.stats.starttime) > this_tr.stats.delta/4):
        print('Different start times. Cannot merge.')  
        error_flag = True

    if (abs(this_tr.stats.endtime - other_tr.stats.endtime) > this_tr.stats.delta/4):
        print('Different end times. Cannot merge.')  
        error_flag = True
        
    if error_flag: # traces incompatible, so return the trace with the most non-zero values
        this_good = np.count_nonzero(this_tr.data)
        #print(this_tr.stats)
        other_good = np.count_nonzero(other_tr.data)
        #print(other_tr.stats)
        if other_good > this_good:
            return other_tr
        else:
            return this_tr
    
    else: # things are good
        indices = np.where(other_tr.data == 0)
        other_tr.data[indices] = this_tr.data[indices]
        return other_tr

######################################################################
##                  Modeling  tools                                 ##
######################################################################


def predict_arrival_times(station, quake):
    """ calculate predicted travel times based on IASP91 model  - see https://docs.obspy.org/packages/obspy.taup.html
        Input: station and quake both are dicts with lat and lon keys
        Output: a phases dict is added to statihttps://www.facebook.com/on, with phase name keys and predicted arrival times """
    model = TauPyModel(model="iasp91")
    
    [dist_in_m, az1, az2] = gps2dist_azimuth(quake['lat'], quake['lon'], station['lat'], station['lon'])
    station['distance'] = kilometers2degrees(dist_in_m/1000)
    arrivals = model.get_travel_times(source_depth_in_km=quake['depth'],distance_in_degree=station['distance'])
    # https://docs.obspy.org/packages/autogen/obspy.taup.helper_classes.Arrival.html#obspy.taup.helper_classes.Arrival
    
    phases = dict()
    for a in arrivals:
        phasetime = quake['otime'] + a.time
        phases[a.name] = phasetime.strftime('%H:%M:%S')
        if a.name == 'S':
            Rtime = quake['otime'] + a.time/ ((0.8453)**0.5)
            phases['Rayleigh'] = Rtime.strftime('%H:%M:%S')
    station['phases'] = phases
    
    return station

def syngine2stream(station, lat, lon, GCMTeventID, mseedfile):
    """ Generate synthetics for a GCMT event, save into an mseedfile, return as Stream object """
    if os.path.exists(mseedfile):
        synth_disp = read(mseedfile)
    else:
        synth_disp = read("http://service.iris.edu/irisws/syngine/1/query?"
                  "format=miniseed&units=displacement&dt=0.02&"
                  "receivercenterlat=%f&receivercenterlon=%f&"
                  "eventid=GCMT:%s" % (lat, lon, GCMTeventID))
        for c in range(len(synth_disp)):
            synth_disp[c].stats.centerlat = lat
            synth_disp[c].stats.centerlon = lon
        synth_disp.write(mseedfile)
    return synth_disp

