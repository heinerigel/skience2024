import pickle
import os
from pathlib import Path
from obspy import UTCDateTime
from obspy.core.event import Event, Catalog, Origin
from obspy.core.event.magnitude import Amplitude
from obspy.core.event.base import CreationInfo, WaveformStreamID
from obspy.core.event.magnitude import Magnitude, StationMagnitude
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
SECS_PER_DAY = 86400

def tree(dir_path: Path, prefix: str=''):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """    

    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '
    
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir(): # extend the prefix and recurse:
            extension = branch if pointer == tee else space 
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)

class VolcanoSeismicCatalog(Catalog):
    def __init__(self, events=None, streams=None, triggers=None, miniseedfiles=None, classifications=None, triggerMethod=None, \
                 threshON=None, threshOFF=None, sta=None, lta=None, max_secs=None, \
                 starttime=None, endtime=None,
                 pretrig=None, posttrig=None, **kwargs):
        self.events = []
        self.streams = []
        self.triggers = []
        self.miniseedfiles = []
        self.classifications = []
        if events:
            self.events = events
        if streams:
            self.streams = streams
        if miniseedfiles:
            self.miniseedfiles = miniseedfiles          
        if triggers:
            self.triggers = triggers 
        if classifications:
            self.classifications = classifications   
        self.triggerParams = {'method':triggerMethod, 'threshON':threshON, 'threshOFF':threshOFF, 'sta':sta, \
                              'lta':lta, 'max_secs':max_secs, 'pretrig':pretrig, 'posttrig':posttrig}   
        self.starttime = starttime
        self.endtime = endtime
        self.comments = kwargs.get("comments", [])
        self._set_resource_id(kwargs.get("resource_id", None))
        self.description = kwargs.get("description", "")
        self._set_creation_info(kwargs.get("creation_info", None))        

    def addEvent(self, this_trigger, this_stream, this_event):
        self.streams.append(this_stream)
        self.triggers.append(this_trigger)
        self.append(this_event) 

    def get_times(self): # not optimal implementation as I am assuming 1 origin per event & ignoring preferred origin
        times = []
        for this_event in self.events:
            for origin in this_event['origins']:
                times.append(origin['time'])
        return times

    def plot_streams(self):
        for i,st in enumerate(self.streams):
            print('\nEVENT NUMBER: ',f'{i+1}', 'time: ', f'{st[0].stats.starttime}', '\n')
            st.plot(equal_scale=False)
    
    def concat(self, other):
        self.events.extend(other.events)
        self.triggers.extend(other.triggers)
        self.streams.extend(other.streams)

    def write_events(self, outdir, net='MV', xmlfile=None):
        if xmlfile:
            self.write(os.path.join(outdir, xmlfile), format="QUAKEML")  
        times = self.get_times()
        for i, st in enumerate(self.streams): # write streams if they exist
            if not st:
                continue
            mseedfile = os.path.join(outdir, 'WAV', net, times[i].strftime('%Y'), times[i].strftime('%m'), times[i].strftime('%Y%m%dT%H%M%S.mseed'))
            mseeddir = os.path.dirname(mseedfile)
            if not os.path.isdir(mseeddir):
                os.makedirs(mseeddir)
            self.miniseedfiles.append(mseedfile)
            '''
            if not xmlfile:
                qmlfile = os.path.join(outdir, 'REA', net, times[i].strftime('%Y'), times[i].strftime('%m'), times[i].strftime('%Y%m%dT%H%M%S.xml'))
                self.events[i].write(qmlfile, format='QUAKEML')
            '''
            #print(f'Writing {mseedfile}')
            st.write(mseedfile, format='mseed')

    def to_dataframe(self):
        times = [t.datetime for t in self.get_times()]
        #pretrig = self.triggerParams['pretrig']
        #posttrig = self.triggerParams['posttrig']
        #durations = [this_trig['duration'] for this_trig in self.triggers]
        magnitudes = []
        lats = []
        longs = []
        depths = []
        for eventObj in self.events:
            for m in eventObj.magnitudes:
                magnitudes.append(eventObj.magnitudes[0]['mag'])
            if len(eventObj.origins)>0:
                orObj = eventObj.origins[0]
                lats.append(orObj.latitude)
                longs.append(orObj.longitude)
                depths.append(orObj.depth)
            else:
                lats.append(None)
                longs.append(None)
                depths.append(None)
        
        df = pd.DataFrame()
        df['datetime'] = times
        df['magnitude'] = pd.Series(magnitudes)
        df['energy'] = pd.Series([magnitude2energy(m) for m in magnitudes])
        df['latitude'] = pd.Series(lats)
        df['longitude'] = pd.Series(longs)
        df['depth'] = pd.Series(depths)
        #df['duration'] = pd.Series(durations)
        df['filename'] = [t.strftime('%Y%m%dT%H%M%S') for t in self.get_times()]
        if len(self.triggers)>0:
            if isinstance(self.triggers, list):
                durations = []
                for this_trig in self.triggers:
                    durations.append(this_trig['duration'])
                df['duration'] = pd.Series(durations)
        else:
            df['duration'] = None        
        if len(self.classifications)>0:
            if isinstance(self.classifications, list):
                df['classification'] = pd.Series(self.classifications)
        else:
            df['classification'] = None
        return df

    def save(self, outdir, outfile, net='MV'):
        self.write_events(outdir, net=net, xmlfile=outfile + '.xml')
        #df = self.catalog2dataframe()
        #df.to_pickle(os.path.join(outdir, outfile + '_df.pkl'))
        pklfile = os.path.join(outdir, outfile + '_vars.pkl')
        picklevars = {} 
        picklevars['triggers'] = self.triggers
        picklevars['miniseedfiles'] = self.miniseedfiles
        picklevars['classifications'] = self.classifications
        picklevars['triggerParams'] = self.triggerParams
        picklevars['comments'] = self.comments
        picklevars['description'] = self.description
        picklevars['starttime'] = self.starttime.strftime('%Y/%m/%d %H:%M:%S')
        picklevars['endtime'] = self.endtime.strftime('%Y/%m/%d %H:%M:%S')     
        try:
            with open(pklfile, "wb") as fileptr:
                pickle.dump(picklevars, fileptr, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)
      
    '''    
    def save(self, pklfile):
        print(f'Writing {pklfile}')
        try:
            with open(pklfile, "wb") as f:
                pickle.dump(self, fileptr, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)
    '''
            
    def plot_eventrate(self, binsize=pd.Timedelta(days=1), time_limits=None):
        # input times are obspy.UTCDateTime but all converted to datetime.datetime inside function
        #times = self.get_times()
    
        df = self.to_dataframe()
        dfsum = plot_eventrate(df, binsize=binsize, time_limits=time_limits)

        '''
        df['counts']=pd.Series([1 for i in range(len(df))])
        if time_limits:
            sdate= time_limits[0]
            edate = time_limits[1]
            mask = (df['datetime'] >= sdate) & (df['datetime'] <= edate)
            df = df.loc[mask]
            
        #df['counts']=pd.Series([1 for i in range(len(df))])
        dfsum = df.set_index('datetime').resample(binsize).sum() 
        dfsum['cumcounts'] = dfsum['counts'].cumsum()
        dfsum['cumenergy'] = dfsum['energy'].cumsum()
        numevents = len(df)
        if numevents > 0:
            fig1, axs = plt.subplots(2, 1, sharex=True)
            #df.plot(ax=axs[0], x='datetime', y='magnitude', kind='scatter', style='o', xlabel='Time', ylabel='magnitude', rot=90, grid=True)
            ##axs[0].set_xticklabels = []
            
            dfsum.plot.line(ax=axs[0], y='counts', style='b', ylabel='Counts')
            dfsum.plot.line(ax=axs[0], y='cumcounts', secondary_y=True, style='g', ylabel='Cumulative', rot=90, grid=True)
            axs[0].set_xticklabels = []

            # add subplot - energy versus time
            dfsum.plot.line(ax=axs[1], y='energy', style='b', ylabel='Energy')
            dfsum.plot.line(ax=axs[1], y='cumenergy', secondary_y=True, style='g', ylabel='Cumulative', rot=90, grid=True)
        '''
        return dfsum


def plot_eventrate(df, binsize=pd.Timedelta(days=1), time_limits=None):
    df['counts']=pd.Series([1 for i in range(len(df))])
    if time_limits:
        sdate= time_limits[0]
        edate = time_limits[1]
        mask = (df['datetime'] >= sdate) & (df['datetime'] <= edate)
        df = df.loc[mask]
        
    #df['counts']=pd.Series([1 for i in range(len(df))])
    dfsum = df.set_index('datetime').resample(binsize).sum() 
    dfsum['cumcounts'] = dfsum['counts'].cumsum()
    dfsum['cumenergy'] = dfsum['energy'].cumsum()
    numevents = len(df)
    if numevents > 0:
        fig1, axs = plt.subplots(2, 1, sharex=True)
        #df.plot(ax=axs[0], x='datetime', y='magnitude', kind='scatter', style='o', xlabel='Time', ylabel='magnitude', rot=90, grid=True)
        ##axs[0].set_xticklabels = []
        
        dfsum.plot.line(ax=axs[0], y='counts', style='b', ylabel='Counts')
        dfsum.plot.line(ax=axs[0], y='cumcounts', secondary_y=True, style='g', ylabel='Cumulative', rot=90, grid=True)
        axs[0].set_xticklabels = []

        # add subplot - energy versus time
        dfsum.plot.line(ax=axs[1], y='energy', style='b', ylabel='Energy')
        dfsum.plot.line(ax=axs[1], y='cumenergy', secondary_y=True, style='g', ylabel='Cumulative', rot=90, grid=True)

        plt.show()
    return dfsum

# SCAFFOLD: need to reconstruct a catalog object from a dataframe
from obspy.core.event import read_events
def load_catalog(catdir, catfile, net='MV'):
    qmlfile = os.path.join(catdir, catfile + '.xml')
    if os.path.exists(qmlfile):
        catObj = read_events(qmlfile)
        self = VolcanoSeismicCatalog( events=catObj.events.copy())
        pklfile = os.path.join(catdir, catfile + '_vars.pkl')
        if os.path.isfile(pklfile):
            try:
                with open(pklfile, 'rb') as fileptr:
                    picklevars = pickle.load(fileptr)  
            except Exception as ex:
                print(f"Error reading {pklfile}:", ex)
        self.triggers = picklevars['triggers']
        self.miniseedfiles = picklevars['miniseedfiles']
        self.classifications = picklevars['classifications']
        self.triggerParams = picklevars['triggerParams']
        self.comments = picklevars['comments']
        self.description = picklevars['description']
        self.starttime = UTCDateTime.strptime(picklevars['starttime'],'%Y/%m/%d %H:%M:%S')
        self.endtime = UTCDateTime.strptime(picklevars['endtime'],'%Y/%m/%d %H:%M:%S')
        return self
    else:
        print(qmlfile, ' not found')
        return None

    '''    
    EVENTS_DIR = os.path.dirname(pklfile)

    if os.path.exists(pklfile):
        print(f'Loading {pklfile}')
        df = pd.read_pickle(pklfile)
        mseedfiles = [os.path.join(EVENTS_DIR, 'WAV', net, filename[0:4], filename[4:6], filename + '.mseed') for filename in df['filename']]
        #print(mseedfiles)
        print(df)
        catObj = read_events(qmlfile)
        volcanoSeismicCatObj = VolcanoSeismicCatalog( events=catObj.events.copy(), miniseedfiles = mseedfiles )
                        triggerMethod=None, threshON=threshON, threshOFF=threshOFF, \
                       sta=sta_secs, lta=lta_secs, max_secs=max_secs, \
                       pretrig=pretrig, posttrig=posttrig, starttime=stream[0].stats.starttime, endtime=stream[0].stats.endtime) 
    '''


def triggers2catalog(trig, triggerMethod, threshON, threshOFF, sta_secs, lta_secs, max_secs, stream=None, pretrig=None, posttrig=None ):
    if stream:
        cat = VolcanoSeismicCatalog(triggerMethod=None, threshON=threshON, threshOFF=threshOFF, \
                       sta=sta_secs, lta=lta_secs, max_secs=max_secs, \
                       pretrig=pretrig, posttrig=posttrig, starttime=stream[0].stats.starttime, endtime=stream[0].stats.endtime) 
    else:
        cat = VolcanoSeismicCatalog(triggerMethod=None, threshON=threshON, threshOFF=threshOFF, \
                       sta=sta_secs, lta=lta_secs, max_secs=max_secs)        

    for thistrig in trig:
        origin_object = Origin(time=thistrig['time'])
        amplitude_objects = []
        magnitude_objects = []
        stationmag_objects = []
        sta_mags = []
        #print(stream)
        if stream:
            this_st = stream.copy().trim(starttime=thistrig['time']-pretrig, endtime=thistrig['time']+thistrig['duration']+posttrig)
            #print(this_st)
            for i, seed_id in enumerate(thistrig['trace_ids']):
                sta_amp = np.nanmax(np.absolute(this_st[i].data))
                amp_obj = Amplitude(snr=thistrig['cft_peaks'][i], generic_amplitude=sta_amp, \
                                    unit='dimensionless', waveform_id = WaveformStreamID(seed_string=this_st[i].id) )
                amplitude_objects.append(amp_obj)
                sta_mag = np.log10(sta_amp)-2.5 # SCAFFOLD: not a real magnitude
                sta_mags.append(sta_mag)
                stationmag_objects.append(StationMagnitude(mag=sta_mag, mag_type='M', waveform_id = WaveformStreamID(seed_string=this_st[i].id) ))
            avg_mag = np.nanmean(sta_mags)
            networkmag_object = Magnitude(mag=avg_mag, mag_type='M')
            magnitude_objects.append(networkmag_object)
        else:
            this_st = None
        info = CreationInfo(author="coincidence_trigger", creation_time=UTCDateTime())
        this_event = Event(EventType="not reported", creation_info=info, origins=[origin_object], \
                           amplitudes=amplitude_objects, magnitudes=magnitude_objects, station_magnitudes=stationmag_objects)
        cat.addEvent(thistrig, this_st, this_event)
        
    return cat

        
def magnitude2energy(mag):
	energy = np.power(10, 1.5 * mag)
	return energy

def energy2magnitude(energy):
	mag = np.log10(energy)/1.5
	return mag
    
def real_time_optimization(band='all'):
    corners = 2
    if band=='VT':
        # VT + false
        sta_secs = 1.4
        lta_secs = 7.0
        threshON = 2.4
        threshOFF = 1.2
        freqmin = 3.0
        freqmax = 18.0
    elif band=='LP':
        # LP + false
        sta_secs = 2.3
        lta_secs = 11.5
        threshON = 2.4
        threshOFF = 1.2
        freqmin = 0.8
        freqmax = 10.0        
    elif band=='all':
        # all = LP + VT + false
        sta_secs = 2.3
        lta_secs = 11.5
        threshON = 2.4
        threshOFF = 1.2
        freqmin = 1.5
        freqmax = 12.0        
    threshOFF = threshOFF / threshON
        
    return sta_secs, lta_secs, threshON, threshOFF, freqmin, freqmax, corners



def classify_event(dfratio, rsamObj, this_trig, sampling_interval=2.56):
    df = dfratio.set_index('datetime')
    count = {'r':0, 'e':0, 'l':0, 'h':0, 't':0}
    for col in df.columns:
        if col in this_trig['trace_ids']:
            try:
                if len(df)>3:
                    c = np.polyfit( range(len(df)), df[col], 1) 
                    if c[0]>0.0:
                        count['e'] += int(c[0]* 20)
                    elif c[0]<0.0:
                        count['h'] += int(abs(c[0])* 20)
            except:
                pass
            for el in df[col]:
                if el < -1.0:
                    count['l'] += 1
                elif el > 0.0:
                    count['t'] += 1
                else:
                    count['h'] += 1
                    count['r'] += 1
    for seed_id in rsamObj.dataframes:
        df2 = rsamObj.dataframes[seed_id]
        maxi = np.argmax(df2['median'])
        r = maxi/len(df2) 
        if r > 0.3 and r < 0.7:
            count['r'] += 4
            count['l'] += 2
        elif r < 0.3:
            count['t'] += 2
            count['h'] += 1
            count['e'] += 2
        else:
            count['t'] -= 3
            count['h'] -= 2
            count['l'] -= 1         
    
    duration = this_trig['duration']
    if duration < 5.0:
        count['t'] += 4
        count['h'] += 3
        count['l'] += 1
        count['r'] -= 5
        count['e'] -= 10    
    if duration < 10.0:
        count['t'] += 3
        count['h'] += 2
        count['l'] += 1
        count['r'] -= 1
        count['e'] -= 2
    elif duration < 20.0:
        count['t'] += 2
        count['h'] += 3
        count['l'] += 2
        count['e'] -= 1
    elif duration < 30.0:
        count['t'] += 1
        count['h'] += 2
        count['l'] += 3 
        count['r'] += 1
    elif duration < 40.0:
        count['h'] += 1
        count['l'] += 2 
        count['r'] += 2 
        count['e'] += 1
    elif duration < 50.0:
        count['l'] += 1 
        count['r'] += 4 
        count['e'] += 4
    else:
        count['r'] += 3
        count['e'] += 5

    total = sum(count.values())
    for k in count:
        count[k] = count[k]/total
    #print(count)
    subclass = max(zip(count.values(), count.keys()))[1]

    return subclass

def compute_fdominant(st, evtime, duration, inv, show_response_plots=False, show_stream_plots=False, show_frequency_plots=False, interactive=False, return_dominantF=False, pre_filt = [0.25, 0.5, 25, 50]):
    for tr in st:
        tr.stats['units'] = 'Counts'
    st.detrend('linear')
    #st.taper(0.05, type='hann')
    if show_stream_plots:
        st.plot();
    
    vel = st.copy()
    vel.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL", plot=show_response_plots) 
    for tr in vel:
        tr.stats['units'] = 'm/s'
    if show_stream_plots:
        vel.plot();
    vel.trim(starttime=evtime, endtime=evtime+duration)
    
    disp = st.copy()
    disp.remove_response(inventory=inv, pre_filt=pre_filt, output="DISP", plot=show_response_plots) 
    for tr in disp:
        tr.stats['units'] = 'm'
    if show_stream_plots:
        disp.plot();
    disp.trim(starttime=evtime, endtime=evtime+duration)

    # Generate a Velocity Seismic Amplitude Measurement (VSAM) object - units must be 'm/s' else will not work
    vsamObj = VSAM(stream=vel, sampling_interval=2.56)
    #print(vsamObj)  
    if show_frequency_plots:
        try:
            vsamObj.plot(metrics=['fratio'])
        except:
            if interactive:
                input('<ENTER> to continue')    
            return None
            
    if return_dominantF:
        # Generate a Displacement Seismic Amplitude Measurement (DSAM) object - units must be 'm' else will not work
        dsamObj = DSAM(stream=disp, sampling_interval=2.56)
        #print(dsamObj)so
        #dsamObj.plot()
    
        N = len(st)
        vsam_stream = vsamObj.to_stream()
        dsam_stream = dsamObj.to_stream()
        dfdom = pd.DataFrame()
        dfdom['datetime'] = [t.datetime for t in vsam_stream[0].times('utcdatetime')]
        for c in range(N):
            fdominant = np.divide(np.absolute(vsam_stream[c].data), np.absolute(dsam_stream[c].data)) / (2 * np.pi)
            dfdom[vsam_stream[c].id] = fdominant
        if show_frequency_plots:
            dfdom.plot(x='datetime', ylabel='Dominant Frequency (Hz)')
        dfdom['mean'] = dfdom.mean(axis=1, numeric_only=True)
    else:
        dfdom = None


if __name__ == "__main__":
    print('Tree listing of current directory')
    for line in tree(Path.cwd().joinpath('.')):
        print(line)