from pathlib import Path
from obspy import UTCDateTime
from obspy.core.event import Event, Catalog, Origin
from obspy.core.event.magnitude import Amplitude
from obspy.core.event.base import CreationInfo, WaveformStreamID
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, events=None, streams=None, triggers=None, triggerMethod=None, \
                 threshON=None, threshOFF=None, sta=None, lta=None, max_secs=None, \
                 starttime=None, endtime=None,
                 pretrig=None, posttrig=None, **kwargs):
        self.streams = []
        self.triggers = []
        if not events:
            self.events = []
        else:
            self.events = events
        if not streams:
            self.streams = []
        else:
            self.streams = streams
        if not triggers:
            self.triggers = []
        else:
            self.triggers = triggers 
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
            
    def plot_eventrate(self):
        times = self.get_times()
        counts = np.cumsum(np.ones(len(times)))
        times.insert(0, self.starttime) 
        times.append(self.endtime)
        counts = np.insert(counts, 0, 0)
        counts = np.append(counts, counts[-1])
        plt.figure()
        plt.plot([t.datetime for t in times], counts)

    def concat(self, other):
        self.events.extend(other.events)
        self.triggers.extend(other.triggers)
        self.streams.extend(other.streams)

    def write_events(self, xmlfile):
        self.write(xmlfile, format="QUAKEML")  
        times = self.get_times()
        for i, st in enumerate(streams):
            mseedfile = times[i].strftime('%Y%m%dT%H%M%S.mseed')
            st.write(mseedfile, format='mseed')

def triggers2volcanoseismiccatalog(trig, triggerMethod, threshON, threshOFF, sta_secs, lta_secs, max_secs, stream=None, pretrig=None, posttrig=None ):
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
        if stream:
            this_st = stream.copy().trim(starttime=thistrig['time']-pretrig, endtime=thistrig['time']+thistrig['duration']+posttrig)
            for i, seed_id in enumerate(thistrig['trace_ids']):
                amp_obj = Amplitude(snr=thistrig['cft_peaks'][i], generic_amplitude=np.nanmax(np.absolute(this_st[i].data)), \
                                    unit='dimensionless', waveform_id = WaveformStreamID(seed_string=this_st[i].id) )
                amplitude_objects.append(amp_obj)
        else:
            this_st = None
        info = CreationInfo(author="coincidence_trigger", creation_time=UTCDateTime())
        this_event = Event(EventType="not reported", creation_info=info, origins=[origin_object], \
                           amplitudes=amplitude_objects)
        cat.addEvent(thistrig, this_st, this_event)
        
    return cat
        
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
            
if __name__ == "__main__":
    print('Tree listing of current directory')
    for line in tree(Path.cwd().joinpath('.')):
        print(line)