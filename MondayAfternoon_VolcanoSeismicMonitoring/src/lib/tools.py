import pickle
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
    def __init__(self, events=None, streams=None, triggers=None, triggerMethod=None, \
                 threshON=None, threshOFF=None, sta=None, lta=None, max_secs=None, \
                 starttime=None, endtime=None,
                 pretrig=None, posttrig=None, **kwargs):
        self.streams = []
        self.triggers = []
        self.miniseedfiles = []
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
            
    def plot_eventrate(self, binsize=pd.Timedelta(days=1)):
        times = self.get_times()
        counts = np.cumsum(np.ones(len(times)))
        times.insert(0, self.starttime) 
        times.append(self.endtime)
        counts = np.insert(counts, 0, 0)
        counts = np.append(counts, counts[-1])
        plt.figure()
        plt.plot([t.datetime for t in times], counts)

        df = self.to_catalog_dataframe()
        dfcounts = df.set_index('time').resample(binsize).sum() 
        print(dfcounts)
        dfcounts['cumsum'] = dfcounts['sum'].cumsum()
        print(dfcounts)
        dfenergy = df.set_index('time').resample(binsize)['energy'].sum()
        print(dfenergy)
        dfenergy['cumsum'] = dfenergy['energy'].cumsum()
        print(dfenergy)
        numevents = len(df)
        if numevents > 0:
            fig1 = plt.figure()
            
            # add subplot - counts versus time
            ax1 = fig1.add_subplot(211)
            plot_counts(ax1, dfcounts, stime, etime)
            
            # add subplot - energy versus time
            ax2 = fig1.add_subplot(212)
            plot_energy(ax2, dfenergy, stime, etime)


    
    def concat(self, other):
        self.events.extend(other.events)
        self.triggers.extend(other.triggers)
        self.streams.extend(other.streams)

    def write_events(self, outdir, xmlfile=None):
        if xmlfile:
            self.write(os.path.join(outdir, xmlfile), format="QUAKEML")  
        times = self.get_times()
        for i, st in enumerate(self.streams):
            mseedfile = os.path.join(outdir, times[i].strftime('%Y%m%dT%H%M%S.mseed'))
            self.miniseedfiles.append(mseedfile)
            if not xmlfile:
                qmlfile = mseedfile.replace('.mseed','.xml')
                self.events[i].write(qmlfile, format='QUAKEML')
            print(f'Writing {mseedfile}')
            st.write(mseedfile, format='mseed')

    def to_catalog_dataframe(self):
        times = [t.datetime for t in self.get_times()]
        magnitudes = []
        lats = []
        longs = []
        depths = []
        for eventObj in self.events:
            magnitudes.append(ev.magnitudes[0]['mag'])
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
        df['time'] = times
        df['mag'] = pd.Series(magnitudes)
        df['energy'] = pd.Series([mag2eng(m) for m in magnitudes])
        df['latitude'] = pd.Series(lats)
        df['longitude'] = pd.Series(longs)
        df['depth'] = pd.Series(depths)
        return df
  
    def save(self, pklfile):
        with open(pklfile, 'wb') as fileptr: 
            print(f'Writing {pklfile}')
            # A new file will be created 
            pickle.dump(ergrid, fileptr)

def load_catalog(pklfile):
    if os.path.exists(pklfile):
        print(f'Loading {pklfile}')
        with open(pklfile, 'rb') as fileptr:
            cat = pickle.load(fileptr)  
        return cat
    else:
        print(pklfile, ' not found')
        return None

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
        if stream:
            this_st = stream.copy().trim(starttime=thistrig['time']-pretrig, endtime=thistrig['time']+thistrig['duration']+posttrig)
            for i, seed_id in enumerate(thistrig['trace_ids']):
                sta_amp = np.nanmax(np.absolute(this_st[i].data))
                amp_obj = Amplitude(snr=thistrig['cft_peaks'][i], generic_amplitude=sta_amp, \
                                    unit='dimensionless', waveform_id = WaveformStreamID(seed_string=this_st[i].id) )
                amplitude_objects.append(amp_obj)
                sta_mag = np.log10(sta_amp)
                sta_mags.append(sta_mag)
                stationmag_objects.append(StationMagnitude(mag=sta_mag, mag_type='M'))
            avg_mag = np.nanmean(sta_mags)
            networkmag_object = Magnitude(mag=avg_map, mag_type='M')
            magnitude_objects.append(networkmag_objects)
        else:
            this_st = None
        info = CreationInfo(author="coincidence_trigger", creation_time=UTCDateTime())
        this_event = Event(EventType="not reported", creation_info=info, origins=[origin_object], \
                           amplitudes=amplitude_objects, magnitudes=magnitude_objects, station_magnitudes=stationmag_objects)
        cat.addEvent(thistrig, this_st, this_event)
        
    return cat

def plot_time_mag(df, ax=None):
    if not ax:
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)    
    df.plot(ax=ax, x='time', y='mag', kind='scatter', style='o', xlabel='Time', ylabel='magnitude', rot=90)
    ax.grid(True)


def plot_counts(ax, df, stime, etime):
	#binsize = df['time'][1] - df['time'][0]
	#binsize_str = binsizelabel(binsize / SECS_PER_DAY)

  	# plot 
    df.plot.bar()
    '''
	counts, bin_edges_out, patches = ax.hist(time, bin_edges_in, cumulative=False, histtype='bar', color='black', edgecolor=None)
    ax.grid(True)
    ax.xaxis_date()
    plt.setp( ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=7 )
    ax.set_ylabel("# Earthquakes\n%s" % binsize_str, fontsize=8)
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
	if snum and enum:
        ax.set_xlim(snum, enum)
        ax2 = ax.twinx()
        p2, = ax2.plot(time,cumcounts,'g', lw=2.5)
        ax2.yaxis.get_label().set_color(p2.get_color())
	ytl_obj = plt.getp(ax2, 'yticklabels')  # get the properties for yticklabels
	#plt.getp(ytl_obj)                       # print out a list of properties
	plt.setp(ytl_obj, color="g")            # set the color of yticks to red
	plt.setp(plt.getp(ax2, 'yticklabels'), color='g') #xticklabels: same
    ax2.set_ylabel("Cumulative\n# Earthquakes", fontsize=8)
    ax2.xaxis.set_major_locator(x_locator)
    ax2.xaxis.set_major_formatter(x_formatter)
	if snum and enum:
        	ax2.set_xlim(snum, enum)
        return
    '''

def plot_energy(ax, dictorigin, x_locator, x_formatter, bin_edges, snum, enum):
	# compute all data needed
    '''
    time = dictorigin['time']
	energy = ml2energy(dictorigin['ml'])
    cumenergy = np.cumsum(energy)
	binned_energy = bin_irregular(time, energy, bin_edges)
	if len(bin_edges) < 2:
		return
        barwidth = bin_edges[1:] - bin_edges[0:-1]
	binsize = bin_edges[1]-bin_edges[0]
	binsize_str = binsizelabel(binsize)
    '''

	# plot
    df.plot.bar()
    '''
    ax.bar(bin_edges[:-1], binned_energy, width=barwidth, color='black', edgecolor=None)

    # re-label the y-axis in terms of equivalent Ml rather than energy
    yticklocs1 = ax.get_yticks()
    ytickvalues1 = np.log10(yticklocs1) / 1.5
    yticklabels1 = list()
    for count in range(len(ytickvalues1)):
            yticklabels1.append("%.2f" % ytickvalues1[count])
    ax.set_yticks(yticklocs1)
    ax.set_yticklabels(yticklabels1)

    ax.grid(True)
    ax.xaxis_date()
    plt.setp( ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=7 )
    ax.set_ylabel("Energy %s\n(unit: Ml)" % binsize_str, fontsize=8)
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
	if snum and enum:
        	ax.set_xlim(snum, enum)

	# Now add the cumulative energy plot - again with yticklabels as magnitudes
    ax2 = ax.twinx()
    p2, = ax2.plot(time,cumenergy,'g',lw=2.5)

    # use the same ytick locations as for the left-hand axis, but label them in terms of equivalent cumulative magnitude
    yticklocs1 = ax.get_yticks()
    yticklocs2 = (yticklocs1 / max(ax.get_ylim())) * max(ax2.get_ylim() )
    ytickvalues2 = np.log10(yticklocs2) / 1.5
    yticklabels2 = list()
    for count in range(len(ytickvalues2)):
            yticklabels2.append("%.2f" % ytickvalues2[count])
    ax2.set_yticks(yticklocs2)
    ax2.set_yticklabels(yticklabels2)

    ax2.yaxis.get_label().set_color(p2.get_color())
	ytl_obj = plt.getp(ax2, 'yticklabels')  # get the properties for yticklabels
	#plt.getp(ytl_obj)                       # print out a list of properties
	plt.setp(ytl_obj, color="g")            # set the color of yticks to red
	plt.setp(plt.getp(ax2, 'yticklabels'), color='g') #xticklabels: same
    ax2.set_ylabel("Cumulative Energy\n(unit: Ml)",fontsize=8)
    ax2.xaxis.set_major_locator(x_locator)
    ax2.xaxis.set_major_formatter(x_formatter)
	if snum and enum:
        ax2.set_xlim(snum, enum)
    '''
        
def mag2energy(mag):
	energy = np.power(10, 1.5 * ml)
	return energy

def energy2mag(energy):
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
'''
def bin_counts(time, bin_edges_in):
	# count the number of "events" in each bin 
	# use this if want to produce counts (per bin) but not actually plot them!
        counts, bin_edges_out = np.histogram(time, bin_edges_in)
	return counts 

def bin_irregular(time, y, bin_edges):
	# bin y against time according to bin_edges (not for binning counts, since they don't have a y value)

    # bin the data as for counts
    counts_per_bin, bin_edges_out = np.histogram(time, bin_edges)
    i_start = 0
    i_end = -1
    binned_y = np.empty(np.alen(counts_per_bin))

    for binnum in range(np.alen(counts_per_bin)):
        i_end += counts_per_bin[binnum]
        if i_start <= i_end:
            binned_y[binnum] = np.sum(y[i_start:i_end+1])
        else:
            binned_y[binnum] = 0
        i_start = i_end + 1
    return binned_y
def autobinsize(daysdiff):
    # Try and keep to around 100 bins or less
    if daysdiff <= 2.0/24:  # less than 2 hours of data, use a binsize of 1 minute
        binsize = 1.0/1440
    elif daysdiff <= 4.0:  # less than 4 days of data, use a binsize of 1 hour
        binsize = 1.0/24
    elif daysdiff <= 100.0:  # less than 100 days of data, use a binsize of 1 day
        binsize = 1.0
    elif daysdiff <= 700.0: # less than 700 days of data, use a binsize of 1 week
        binsize = 7.0
    elif daysdiff <= 365.26 * 23: # less than 23 years of data, use a binsize of (approx) 1 month
        binsize = 365.26/12
    else:
        binsize = 365.26 # otherwise use a binsize of 1 year
    return binsize

def compute_bins(df, stime=None, etime=None, binsize=None):
    # If stime and etime are provided, enum will be end of last bin UNLESS you ask for binsize=365.26, or 365.26/12
    # in which case it will be end of year or month boundary
    # If stime and etime not given, they will end at next boundary - and weeks end on Sat midnight/Sunday 00:00
    # binsize is in days, not seconds
    # First lets calculate the difference in time between the first and last events
    if not stime:
        stime = df['time'].min()
        etime = df['time'].max()
        daysdiff = (etime - stime)/SECS_PER_DAY

    if not binsize:
        binsize = autobinsize(daysdiff) 
#
        
    # special cases
    if binsize == 365.26/12:
        # because a month isn't exactly 365.26/12 days, this is not going to be the month boundary
        # so let us get the year and the month for snum, but throw away the day, hour, minute, second etc
        thisyear = sdate.year
        thismonth = sdate.month
        sdate = obspy.UTCDateTime(thisyear, thismonth, 1)
        bins = list()
        bins.append(sdate)
        count = 0
        while bins[count] < etime + binsize * SECS_PER_DAY:
            count += 1
            thismonth += 1
            if thismonth > 12: # datetime.datetime dies if sdate.month > 12
                thisyear += 1
                thismonth -= 12
                monthdate = obspy.UTCDateTime(thisyear, thismonth, 1)
                bins.append(monthdate)
        bins = np.array(bins)
        etime = np.max(bins)

	elif binsize == 365.26: # binsize of 1 year
                # because a year isn't exactly 365.26 days, this is not going to be the year boundary
                # so let us get the year for snum, but throw away the month, day, hour, minute, second etc
                sdate = mpl.dates.num2date(snum)
                sdate = datetime.datetime(sdate.year, 1, 1, 0, 0, 0)
                snum = mpl.dates.date2num(sdate)
                bins = list()
                bins.append(snum)
                count = 0
                while bins[count] < enum + binsize:
                        count += 1
                        yeardate = datetime.datetime(sdate.year + count, 1, 1, 0, 0, 0)
                        bins.append(mpl.dates.date2num(yeardate))
                bins = np.array(bins)
                enum = np.max(bins)

	else: # the usual case
        	# roundoff the start and end times based on the binsize
		if snum==None and enum==None:
			print "snum and enum undefined - calculating"
        		snum = floor(snum, binsize) # start time
        		enum = ceil(enum, binsize) # end time
        	#bins = np.arange(snum, enum+binsize, binsize)
        	bins = np.arange(enum, snum-binsize, -binsize)
		bins = bins[::-1]

	print 'snum: %s' % datenum2datestr(snum)
	print 'enum: %s' % datenum2datestr(enum)
        return bins, snum, enum
'''            
if __name__ == "__main__":
    print('Tree listing of current directory')
    for line in tree(Path.cwd().joinpath('.')):
        print(line)