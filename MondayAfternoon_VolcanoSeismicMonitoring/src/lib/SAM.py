import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import struct
import fnmatch
import obspy
#from obspy.core.inventory.inventory import Inventory
from obspy.geodetics.base import gps2dist_azimuth, degrees2kilometers
import math

class SAM:

    def __init__(self, dataframes=None, stream=None, sampling_interval=60.0, filter=[0.5, 18.0], bands = {'VLP': [0.02, 0.2], 'LP':[0.5, 4.0], 'VT':[4.0, 18.0]}, corners=4, clip=None, verbose=False):
        ''' Create an SAM object 
        
            Optional name-value pairs:
                dataframes: Create an SAM object using these dataframes. Used by downsample() method, for example. Default: None.
                stream: Create an SAM object from this ObsPy.Stream object.
                sampling_interval: Compute SAM data using this sampling interval (in seconds). Default: 60
                filter: list of two floats, representing fmin and fmax. Default: [0.5, 18.0]. Set to None if no filter wanted.
                bands: a dictionary of filter bands and corresponding column names. Default: {'VLP': [0.02, 0.2], 'LP':[0.5, 4.0], 
                    'VT':[4.0, 18.0]}. For example, the default setting creates 3 additional columns for each DataFrame called 
                    'VLP', 'LP', and 'VT', which contain the mean value for each sampling_interval within the specified filter band
                    (e.g. 0.02-0.2 Hz for VLP). If 'LP' and 'VT' are in this dictionary, an extra column called 'fratio' will also 
                    be computed, which is the log2 of the ratio of the 'VT' column to the 'LP' column, following the definition of
                    frequency ratio by Rodgers et al. (2015).
        '''
        self.dataframes = {} 

        if isinstance(dataframes, dict):
            good_dataframes = {}
            for id, df in dataframes.items():
                if isinstance(df, pd.DataFrame):
                    good_dataframes[id]=df
            if len(good_dataframes)>0:
                self.dataframes = good_dataframes
                #print('dataframes found. ignoring other arguments.')
                return
            else:
                #print('no valid dataframes found')
                pass

        if not isinstance(stream, obspy.core.Stream):
            # empty SAM object
            print('creating blank SAM object')
            return
        
        good_stream = self.check_units(stream)
        if verbose:
            print('good_stream:\n',good_stream)

        if len(good_stream)>0:
            if good_stream[0].stats.sampling_rate == 1/sampling_interval:
                # no downsampling to do
                for tr in good_stream:
                    df = pd.DataFrame()
                    df['time'] = pd.Series(tr.times('timestamp'))
                    df['mean'] = pd.Series(tr.data) 
                    self.dataframes[tr.id] = df
                return 
            elif good_stream[0].stats.sampling_rate < 1/sampling_interval:
                print('error: cannot compute SAM for a Stream with a tr.stats.delta bigger than requested sampling interval')
                return
            
        for tr in good_stream:
            if tr.stats.npts < tr.stats.sampling_rate * sampling_interval:
                print('Not enough samples for ',tr.id,'. Skipping.')
                continue
            #print(tr.id, 'absolute=',absolute)
            df = pd.DataFrame()
            
            t = tr.times('timestamp') # Unix epoch time
            sampling_rate = tr.stats.sampling_rate
            t = self.reshape_trace_data(t, sampling_rate, sampling_interval)
            df['time'] = pd.Series(np.nanmin(t,axis=1))

            if filter:
                if tr.stats.sampling_rate<filter[1]*2.2:
                    print(f"{tr}: bad sampling rate. Skipping.")
                    continue
                tr2 = tr.copy()
                try:
                    tr2.detrend('demean')
                except Exception as e: # sometimes crashes here because tr2.data is a masked array
                    print(e)
                    if isinstance(tr2.data, np.ma.MaskedArray):
                        try:
                            m = np.ma.getmask(tr2.data)
                            tr2.data = tr2.data.filled(fill_value=0)
                            tr2.detrend('demean')
                            tr2.data = tr2.data.filled(fill_value=0)
                        except Exception as e2:
                            print(e2)
                            continue
                    else: # not a masked array
                        continue
                        
                if clip:
                    tr2.data = np.clip(tr2.data, a_max=clip, a_min=-clip)    
                tr2.filter('bandpass', freqmin=filter[0], freqmax=filter[1], corners=corners)
                y = self.reshape_trace_data(np.absolute(tr2.data), sampling_rate, sampling_interval)
            else:
                y = self.reshape_trace_data(np.absolute(tr.data), sampling_rate, sampling_interval)

            df['min'] = pd.Series(np.nanmin(y,axis=1))   
            df['mean'] = pd.Series(np.nanmean(y,axis=1)) 
            df['max'] = pd.Series(np.nanmax(y,axis=1))
            df['median'] = pd.Series(np.nanmedian(y,axis=1))
            df['rms'] = pd.Series(np.nanstd(y,axis=1))

            if bands:
                for key in bands:
                    tr2 = tr.copy()
                    [flow, fhigh] = bands[key]
                    tr2.filter('bandpass', freqmin=flow, freqmax=fhigh, corners=corners)
                    y = self.reshape_trace_data(abs(tr2.data), sampling_rate, sampling_interval)
                    df[key] = pd.Series(np.nanmean(y,axis=1))
                if 'LP' in bands and 'VT' in bands:
                    df['fratio'] = np.log2(df['VT']/df['LP'])
  
            self.dataframes[tr.id] = df

    
    def copy(self):
        ''' make a full copy of an SAM object and return it '''
        selfcopy = self.__class__(stream=obspy.core.Stream())
        selfcopy.dataframes = self.dataframes.copy()
        return selfcopy
    
    def despike(self, metrics=['mean'], thresh=1.5, reps=1, verbose=False):
        if not isinstance(metrics, list):
            metrics = [metrics]
        if metrics=='all':
            metrics = self.get_metrics()
        for metric in metrics:
            st = self.to_stream(metric=metric)
            for tr in st:
                x = tr.data
                count1 = 0
                count2 = 0
                for i in range(len(x)-3): # remove spikes on length 2
                    if x[i+1]>x[i]*thresh and x[i+2]>x[i]*thresh and x[i+1]>x[i+3]*thresh and x[i+2]>x[i+3]*thresh:
                        count2 += 1
                        x[i+1] = (x[i] + x[i+3])/2
                        x[i+2] = x[i+1]
                for i in range(len(x)-2): # remove spikes of length 1
                    if x[i+1]>x[i]*thresh and x[i+1]>x[i+2]*thresh:
                        x[i+1] = (x[i] + x[i+2])/2  
                        count1 += 1  
                if verbose:
                    print(f'{tr.id}: removed {count2} length-2 spikes and {count1} length-1 spikes')           
                self.dataframes[tr.id][metric]=x
        if reps>1:
            self.despike(metrics=metrics, thresh=thresh, reps=reps-1)        

    def downsample(self, new_sampling_interval=3600):
        ''' downsample an SAM object to a larger sampling interval(e.g. from 1 minute to 1 hour). Returns a new SAM object.
         
            Optional name-value pair:
                new_sampling_interval: the new sampling interval (in seconds) to downsample to. Default: 3600
        '''

        dataframes = {}
        for id in self.dataframes:
            df = self.dataframes[id]
            df['date'] = pd.to_datetime(df['time'], unit='s')
            old_sampling_interval = self.get_sampling_interval(df)
            if new_sampling_interval > old_sampling_interval:
                freq = '%.0fmin' % (new_sampling_interval/60)
                new_df = df.groupby(pd.Grouper(key='date', freq=freq)).mean()
                new_df.reset_index(drop=True)
                dataframes[id] = new_df
            else:
                print('Cannot downsample to a smaller sampling interval')
        return self.__class__(dataframes=dataframes) 
        
    def drop(self, id):
        if id in self.__get_trace_ids():
            del self.dataframes[id]

    def get_distance_km(self, inventory, source):
        distance_km = {}
        coordinates = {}
        for seed_id in self.dataframes:
            coordinates[seed_id] = inventory.get_coordinates(seed_id)
            if seed_id[0:2]=='MV':
                if coordinates[seed_id]['longitude'] > 0:
                    coordinates[seed_id]['longitude']  *= -1
            #print(coordinates[seed_id])
            #print(source)
            distance_m, az_source2station, az_station2source = gps2dist_azimuth(source['lat'], source['lon'], coordinates[seed_id]['latitude'], coordinates[seed_id]['longitude'])
            #print(distance_m/1000)
            #distance_km[seed_id] = degrees2kilometers(distance_deg)
            distance_km[seed_id] = distance_m/1000
        return distance_km, coordinates

    def __len__(self):
        return len(self.dataframes)

    def plot(self, metrics=['mean'], kind='stream', logy=False, equal_scale=False, outfile=None):
        ''' plot a SAM object 

            Optional name-value pairs:
                metrics: The columns of each SAM DataFrame to plot. Can be one (scalar), or many (a list)
                         If metrics='bands', this is shorthand for metrics=['VLP', 'LP', 'VT', 'specratio']
                         Default: metrics='mean'
                kind:    The kind of plot to make. kind='stream' (default) will convert each of the request 
                         DataFrame columns into an ObsPy.Stream object, and then use the ObsPy.Stream.plot() method.
                         kind='line' will render plots directly using matplotlib.pyplot, with all metrics requested 
                         on a single plot.
                logy:    In combination with kind='line', will make the y-axis logarithmic. No effect if kind='stream'.
                equal_scale: If True, y-axes for each plot will have same limits. Default: False.
        
        '''
        self.__remove_empty()
        if isinstance(metrics, str):
            metrics = [metrics]
        if kind == 'stream':
            if metrics == ['bands']:
                metrics = ['VLP', 'LP', 'VT', 'fratio']
            for m in metrics:
                print('METRIC: ',m)
                st = self.to_stream(metric=m)
                if outfile:
                    if not m in outfile:
                        this_outfile = outfile.replace('.png', f"_{m}.png")
                        st.plot(equal_scale=equal_scale, outfile=this_outfile);
                    else:
                        st.plot(equal_scale=equal_scale, outfile=outfile);
                else:
                    st.plot(equal_scale=equal_scale);
            return
        for key in self.dataframes:
            df = self.dataframes[key]
            this_df = df.copy()
            this_df['time'] = pd.to_datetime(df['time'], unit='s')
            if metrics == ['bands']:
                # plot f-bands only
                if not 'VLP' in this_df.columns:
                    print('no frequency bands data for ',key)
                    continue
                ph2 = this_df.plot(x='time', y=['VLP', 'LP', 'VT'], kind='line', title=f"{key}, f-bands", logy=logy, rot=45)
                if outfile:
                    this_outfile = outfile.replace('.png', "_bands.png")
                    plt.savefig(this_outfile)
                else:
                    plt.show()
            else:
                for m in metrics:
                    got_all_metrics = True
                    if not m in this_df.columns:
                        print(f'no {m} column for {key}')
                        got_all_metrics = False
                if not got_all_metrics:
                    continue
                if kind == 'line':
                    ph = this_df.plot(x='time', y=metrics, kind=kind, title=key, logy=logy, rot=45)
                elif kind  == 'scatter':
                    fh, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
                    for i, m in enumerate(metrics):
                        this_df.plot(x='time', y=m, kind=kind, ax=ax[i], title=key, logy=logy, rot=45)
                if outfile:
                    this_outfile = outfile.replace('.png', "_metrics.png")
                    plt.savefig(this_outfile)
                else:
                    plt.show()
            plt.close('all')

    @classmethod
    def read(classref, startt, endt, SAM_DIR, trace_ids=None, sampling_interval=60, ext='pickle'):
        ''' read one or many SAM files from folder specified by SAM_DIR for date/time range specified by startt, endt
            return corresponding SAM object

            startt and endt must be ObsPy.UTCDateTime data types

            Optional name-value pairs:
                trace_ids (list): only load SAM files corresponding to these trace IDs.
                sampling_interval (int): seconds of raw seismic data corresponding to each SAM sample. Default: 60
                ext (str): should be 'csv' or 'pickle' (default). Indicates what type of file format to open.

        '''
        #self = classref() # blank SAM object
        dataframes = {}

        if not trace_ids: # make a list of possible trace_ids, regardless of year
            trace_ids = []
            for year in range(startt.year, endt.year+1):
                samfilepattern = classref.get_filename(SAM_DIR, '*', year, sampling_interval, ext)
                #print(samfilepattern)
                samfiles = glob.glob(samfilepattern)
                #print(samfiles)
                #samfiles = glob.glob(os.path.join(SAM_DIR,'SAM_*_[0-9][0-9][0-9][0-9]_%ds.%s' % (sampling_interval, ext )))
                for samfile in samfiles:
                    parts = samfile.split('_')
                    trace_ids.append(parts[-3])
            trace_ids = list(set(trace_ids)) # makes unique
            #print(trace_ids)
        
        for id in trace_ids:
            df_list = []
            for yyyy in range(startt.year, endt.year+1):
                samfile = classref.get_filename(SAM_DIR, id, yyyy, sampling_interval, ext)
                #print(samfile)
                #samfile = os.path.join(SAM_DIR,'SAM_%s_%4d_%ds.%s' % (id, yyyy, sampling_interval, ext))
                if os.path.isfile(samfile):
                    #print('Reading ',samfile)
                    if ext=='csv':
                        df = pd.read_csv(samfile, index_col=False)
                    elif ext=='pickle':
                        df = pd.read_pickle(samfile)
                    if df.empty:
                        continue
                    if 'std' in df.columns:
                        df.rename(columns={'std':'rms'}, inplace=True)
                    df['pddatetime'] = pd.to_datetime(df['time'], unit='s')
                    # construct Boolean mask
                    mask = df['pddatetime'].between(startt.isoformat(), endt.isoformat())
                    # apply Boolean mask
                    subset_df = df[mask]
                    subset_df = subset_df.drop(columns=['pddatetime'])
                    df_list.append(subset_df)
            if len(df_list)==1:
                dataframes[id] = df_list[0]
            elif len(df_list)>1:
                dataframes[id] = pd.concat(df_list)
                
        samObj = classref(dataframes=dataframes) # create SAM object         
        return samObj


    def select(self, network=None, station=None, location=None, channel=None,
               sampling_interval=None, npts=None, component=None, id=None,
               inventory=None):
        """
        Return new SAM object only with DataFrames that match the given
        criteria (e.g. all DataFrames with ``channel="BHZ"``).

        Alternatively, DataFrames can be selected based on the content of an
        :class:`~obspy.core.inventory.inventory.Inventory` object: DataFrame will
        be selected if the inventory contains a matching channel active at the
        DataFrame start time.

        based on obspy.Stream.select()

        .. rubric:: Examples

        >>> samObj2 = samObj.select(station="R*")
        >>> samObj2 = samObj.select(id="BW.RJOB..EHZ")
        >>> samObj2 = samObj.select(component="Z")
        >>> samObj2 = samObj.select(network="CZ")
        >>> samObj2 = samObj.select(inventory=inv)
    
        All keyword arguments except for ``component`` are tested directly
        against the respective entry in the :class:`~obspy.core.trace.Stats`
        dictionary.

        If a string for ``component`` is given (should be a single letter) it
        is tested against the last letter of the ``Trace.stats.channel`` entry.

        Alternatively, ``channel`` may have the last one or two letters
        wildcarded (e.g. ``channel="EH*"``) to select all components with a
        common band/instrument code.

        All other selection criteria that accept strings (network, station,
        location) may also contain Unix style wildcards (``*``, ``?``, ...).
        """
        if inventory is None:
            dataframes = self.dataframes
        else:
            trace_ids = []
            start_dates = []
            end_dates = []
            for net in inventory.networks:
                for sta in net.stations:
                    for chan in sta.channels:
                        id = '.'.join((net.code, sta.code,
                                       chan.location_code, chan.code))
                        trace_ids.append(id)
                        start_dates.append(chan.start_date)
                        end_dates.append(chan.end_date)
            dataframes = {}
            for thisid, thisdf in self.dataframes.items():
                idx = 0
                while True:
                    try:
                        idx = trace_ids.index(thisid, idx)
                        start_date = start_dates[idx]
                        end_date = end_dates[idx]
                        idx += 1
                        if start_date is not None and\
                                self.__get_starttime(thisdf) < start_date:
                            continue
                        if end_date is not None and\
                                self.__get_endtime(thisdf) > end_date:
                            continue
                        dataframes[thisid]=thisdf
                    except ValueError:
                        break
        dataframes_after_inventory_filter = dataframes

        # make given component letter uppercase (if e.g. "z" is given)
        if component is not None and channel is not None:
            component = component.upper()
            channel = channel.upper()
            if (channel[-1:] not in "?*" and component not in "?*" and
                    component != channel[-1:]):
                msg = "Selection criteria for channel and component are " + \
                      "mutually exclusive!"
                raise ValueError(msg)

        # For st.select(id=) without wildcards, use a quicker comparison mode:
        quick_check = False
        quick_check_possible = (id is not None
                                and sampling_rate is None and npts is None
                                and network is None and station is None
                                and location is None and channel is None
                                and component is None)
        if quick_check_possible:
            no_wildcards = not any(['?' in id or '*' in id or '[' in id])
            if no_wildcards:
                quick_check = True
                [net, sta, loc, chan] = id.upper().split('.')

        dataframes = {}
        for thisid, thisdf in dataframes_after_inventory_filter.items():
            [thisnet, thissta, thisloc, thischan] = thisid.upper().split('.')
            if quick_check:
                if (thisnet.upper() == net
                        and thissta.upper() == sta
                        and thisloc.upper() == loc
                        and thischan.upper() == chan):
                    dataframes.append(thisdf)
                continue
            # skip trace if any given criterion is not matched
            if id and not fnmatch.fnmatch(thisid.upper(), id.upper()):
                continue
            if network is not None:
                if not fnmatch.fnmatch(thisnet.upper(),
                                       network.upper()):
                    continue
            if station is not None:
                if not fnmatch.fnmatch(thissta.upper(),
                                       station.upper()):
                    continue
            if location is not None:
                if not fnmatch.fnmatch(thisloc.upper(),
                                       location.upper()):
                    continue
            if channel is not None:
                if not fnmatch.fnmatch(thischan.upper(),
                                       channel.upper()):
                    continue
            if sampling_interval is not None:
                if float(sampling_interval) != self.get_sampling_interval(thisdf):
                    continue
            if npts is not None and int(npts) != self.__get_npts(thisdf):
                continue
            if component is not None:
                if not fnmatch.fnmatch(thischan[-1].upper(),
                                       component.upper()):
                    continue
            dataframes[thisid]=thisdf
        return self.__class__(dataframes=dataframes)       
     
    def to_stream(self, metric='mean'):
        ''' Convert one column (specified by metric) of each DataFrame in an SAM object to an obspy.Trace, 
            return an ObsPy.Stream that is the combination of all of these Trace objects'''
        st = obspy.core.Stream()
        for key in self.dataframes:
            #print(key)
            df = self.dataframes[key]
            if metric in df.columns:
                dataSeries = df[metric]
                tr = obspy.core.Trace(data=np.array(dataSeries))
                #timeSeries = pd.to_datetime(df['time'], unit='s')
                tr.stats.delta = self.get_sampling_interval(df)
                tr.stats.starttime = obspy.core.UTCDateTime(df.iloc[0]['time'])
                tr.id = key
                if tr.data.size - np.count_nonzero(np.isnan(tr.data)):
                    st.append(tr)
                
        return st
    
    def trim(self, starttime=None, endtime=None, pad=False, keep_empty=False, fill_value=None):
        ''' trim SAM object based on starttime and endtime. Both must be of type obspy.UTCDateTime 

            based on obspy.Stream.trim()

            keep_empty=True will retain dataframes that are either blank, or full of NaN's or 0's.
                       Default: False
            
            Note:
            - pad option and fill_value option not yet implemented
        '''
        if pad:
            print('pad option not yet supported')
            return
        if fill_value:
            print('fill_value option not yet supported')
            return
        if not starttime or not endtime:
            print('starttime and endtime required as ObsPy.UTCDateTime')
            return
        for id in self.dataframes:
            df = self.dataframes[id]
            mask = (df['time']  >= starttime.timestamp ) & (df['time'] <= endtime.timestamp )
            self.dataframes[id] = df.loc[mask]
        if not keep_empty:
            self.__remove_empty()
        
    def write(self, SAM_DIR, ext='pickle', overwrite=False):
        ''' Write SAM object to CSV or Pickle files (one per net.sta.loc.chan, per year) into folder specified by SAM_DIR

            Optional name-value pairs:
                ext: Should be 'csv' or 'pickle' (default). Specifies what file format to save each DataFrame.   
        '''
        print('write')
        #print(self)
        if not os.path.isdir(SAM_DIR):
            os.makedirs(SAM_DIR)
        for id in self.dataframes:
            df = self.dataframes[id]
            if df.empty:
                continue
            starttime = df.iloc[0]['time']
            yyyy = obspy.core.UTCDateTime(starttime).year
            samfile = self.get_filename(SAM_DIR, id, yyyy, self.get_sampling_interval(df), ext)
            #samfile = os.path.join(SAM_DIR,'SAM_%s_%4d_%ds.%s' % (id, yyyy, self.get_sampling_interval(df), ext))

            if os.path.isfile(samfile) and not overwrite:
                if ext=='csv':
                    original_df = pd.read_csv(samfile)
                elif ext=='pickle':
                    original_df = pd.read_pickle(samfile)
                # SCAFFOLD: should check if SAM data already exist in file for the DataFrame time range.
                # Currently just combining and dropping duplicates without much thought. maybe ok?
                combined_df = pd.concat([original_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['time'], keep='last') # overwrite duplicate data
                print(f'Modifying {samfile}')
                if ext=='csv':
                    combined_df.to_csv(samfile, index=False)
                elif ext=='pickle':
                    combined_df.to_pickle(samfile)
            else:
                # SCAFFOLD: do i need to create a blank file here for whole year? probably not because smooth() is date-aware
                print(f'Writing {samfile}')
                if ext=='csv':
                    df.to_csv(samfile, index=False)
                elif ext=='pickle':
                    df.to_pickle(samfile)

    @staticmethod
    def check_units(st):
        good_st = st
        print('SAM')
        return good_st
    
    @staticmethod
    def __get_endtime(df):
        ''' return the end time of an SAM dataframe as an ObsPy UTCDateTime'''
        return obspy.core.UTCDateTime(df.iloc[-1]['time'])
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='RSAM'):
	    return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))

    @staticmethod
    def __get_npts(df):
        ''' return the number of rows of an SAM dataframe'''
        return len(df)

    @staticmethod
    def get_sampling_interval(df):
        ''' return the sampling interval of an SAM dataframe in seconds '''
        return df.iloc[1]['time'] - df.iloc[0]['time']       

    @staticmethod
    def __get_starttime(df):
        ''' return the start time of an SAM dataframe as an ObsPy UTCDateTime'''
        return obspy.core.UTCDateTime(df.iloc[0]['time'])
    
    def __get_trace_ids(self):
        return [id for id in self.dataframes]

            
    def __remove_empty(self):
        ''' remove empty dataframes from an SAM object - these are net.sta.loc.chan for which there are no non-zero data '''
        #print('removing empty dataframes')
        dfs_dict = self.dataframes.copy() # store this so we can delete during loop, otherwise complains about deleting during iteration
        for id in self.dataframes:
            df = self.dataframes[id]
            #print(id, self.dataframes[id]['mean'])
            metrics=self.get_metrics(df=df)
            for metric in metrics:
                if (df[metric] == 0).all() or pd.isna(df[metric]).all():
                    #print('dropping ', metric)
                    dfs_dict[id].drop(columns=[metric])
            if len(df)==0:
                del dfs_dict[id]
        self.dataframes = dfs_dict

    @staticmethod
    def reshape_trace_data(x, sampling_rate, sampling_interval):
        ''' reshape data vector from 1-D to 2-D to support vectorized loop for SAM computation '''
        # reshape the data vector into an array, so we can take matplotlib.pyplot.xticks(fontsize= )advantage of np.mean()
        x = np.absolute(x)
        s = np.size(x) # find the size of the data vector
        nc = int(sampling_rate * sampling_interval) # number of columns
        nr = int(s / nc) # number of rows
        x = x[0:nr*nc] # cut off any trailing samples
        y = x.reshape((nr, nc))
        return y
        
    def get_seed_ids(self):
        seed_ids = list(self.dataframes.keys())
        return seed_ids
        
    def get_metrics(self, df=None):
        if isinstance(df, pd.DataFrame):
            metrics = df.columns[1:]
        else:
            seed_ids = self.get_seed_ids()
            metrics = self.dataframes[seed_ids[0]].columns[1:]
        return metrics
        

    def __str__(self):
        contents=""
        for i, trid in enumerate(self.dataframes):
            df = self.dataframes[trid]
            if i==0:
                contents += f"Metrics: {','.join(df.columns[1:])}" + '\n'
                contents += f"Sampling Interval={self.get_sampling_interval(df)} s" + '\n\n'
            startt = self.__get_starttime(df)
            endt = self.__get_endtime(df)        
            contents += f"{trid}: {startt.isoformat()} to {endt.isoformat()}"
            contents += "\n"
        return contents   
    
class RSAM(SAM):
        
    @staticmethod
    def check_units(st):
        print('RSAM')
        good_st = obspy.core.Stream()
        for tr in st:
            if 'units' in tr.stats:
                u = tr.stats['units'].upper()
                if u == 'COUNTS':
                    good_st.append(tr)
            else: # for RSAM, ok if no units
                good_st.append(tr)
        return good_st
        
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='RSAM'):
	    return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))

    @classmethod
    def readRSAMbinary(classref, SAM_DIR, station, stime, etime):
        ''' read one (or many if station is a list) RSAM binary file(s) recorded by the original RSAM system
            return corresponding RSAM object '''
        st = obspy.core.Stream()

        if isinstance(station, list):
            for this_station in station:
                tr = classref.readRSAMbinary(SAM_DIR, this_station, stime, etime)
                if tr.data.size - np.count_nonzero(np.isnan(tr.data)): # throw away Trace objects with only NaNs
                    st.append(tr)
            samObj = classref(stream=st, sampling_interval = 1/st[0].stats.sampling_rate)
            return samObj
        else:

            for year in range(stime.year, etime.year+1):
            
                daysPerYear = 365
                if year % 4 == 0:
                    daysPerYear += 1
                    
                RSAMbinaryFile = os.path.join(SAM_DIR, f"{station}{year}.DAT")
                
                values = []
                if os.path.isfile(RSAMbinaryFile):
                    print('Reading ',RSAMbinaryFile)
        
                    # read the whole file
                    f = open(RSAMbinaryFile, mode="rb")
                    f.seek(4*1440) # 1 day header
                    for day in range(daysPerYear):
                        for minute in range(60 * 24):
                            v = struct.unpack('f', f.read(4))[0]
                            values.append(v)
                            #print(type(v)) 
                            #print(v)
                    f.close()
        
                    # convert to Trace object
                    tr = obspy.Trace(data=np.array(values))
                    tr.stats.starttime = obspy.core.UTCDateTime(year, 1, 1, 0, 0, 0)
                    tr.id=f'MV.{station}..EHZ'
                    tr.stats.sampling_rate=1/60
                    tr.data[tr.data == -998.0] = np.nan
        
                    # Trim based on stime & etime & append to Stream
                    tr.trim(starttime=stime, endtime=etime)
                    st.append(tr)
                else:
                    print(f"{RSAMbinaryFile} not found")
        
            return st.merge(method=0, fill_value=np.nan)[0]


class VSAM(SAM):
    # Before calling, make sure tr.stats.units is fixed to correct units.

    @staticmethod
    def check_units(st):
        print('VSAM')
        good_st = obspy.core.Stream()
        for tr in st:
            if 'units' in tr.stats:
                u = tr.stats['units'].upper()
                if u == 'M/S' or u == 'PA':
                    good_st.append(tr)
        return good_st    

    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='VSAM'):
        return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))

    @staticmethod
    def compute_geometrical_spreading_correction(this_distance_km, chan, surfaceWaves=False, wavespeed_kms=2000, peakf=2.0):
        #print('peakf =',peakf)
        if surfaceWaves and chan[1]=='H': # make sure seismic channel
            wavelength_km = wavespeed_kms/peakf
            gsc = np.sqrt(np.multiply(this_distance_km, wavelength_km))
        else: # body waves - infrasound always here at local distances
            gsc = this_distance_km
        return gsc
    
    
    @staticmethod
    def compute_inelastic_attenuation_correction(this_distance_km, peakf, wavespeed_kms, Q):
        if Q:
            t = np.divide(this_distance_km, wavespeed_kms) # s
            iac = np.exp(math.pi * peakf * t / Q) 
            return iac
        else:
            return 1.0  
      
    def reduce(self, inventory, source, surfaceWaves=False, Q=None, wavespeed_kms=None, fixpeakf=None):
        # if the original Trace objects had coordinates attached, add a method in SAM to save those
        # in self.inventory. And add to SAM __init___ the possibility to pass an inventory object.
        
        #print(self)
        # Otherwise, need to pass an inventory here.
        if not wavespeed_kms:
            if surfaceWaves:
                wavespeed_kms=2 # km/s
            else:
                wavespeed_kms=3 # km/s
        
        # Need to pass a source too, which should be a dict with name, lat, lon, elev.
        distance_km, coordinates = self.get_distance_km(inventory, source)

        corrected_dataframes = {}
        for seed_id, df0 in self.dataframes.items():
            if not seed_id in distance_km:
                continue
            df = df0.copy()
            this_distance_km = distance_km[seed_id]
            ratio = df['VT'].sum()/df['LP'].sum()
            if fixpeakf:
                peakf = fixpeakf
            else:
                peakf = np.sqrt(ratio) * 4
            net, sta, loc, chan = seed_id.split('.')    
            gsc = self.compute_geometrical_spreading_correction(this_distance_km, chan, surfaceWaves=surfaceWaves, \
                                                           wavespeed_kms=wavespeed_kms, peakf=peakf)
            if Q:
                iac = self.compute_inelastic_attenuation_correction(this_distance_km, peakf, wavespeed_kms, Q)
            else:
                iac = 1.0
            for col in df.columns:
                if col in self.get_metrics(): 
                    if col=='VLP':
                        gscvlp = self.compute_geometrical_spreading_correction(this_distance_km, chan, surfaceWaves=surfaceWaves, \
                                                           wavespeed_kms=wavespeed_kms, peakf=0.06)
                        iacvlp = self.compute_inelastic_attenuation_correction(this_distance_km, 0.06, wavespeed_kms, Q)
                        df[col] = df[col] * gscvlp * iacvlp * 1e7 # convert to cm^2/s (or cm^2 for DR)
                    else:
                        df[col] = df[col] * gsc * iac * 1e7                    
            corrected_dataframes[seed_id] = df
        return corrected_dataframes
    
    def compute_reduced_velocity(self, inventory, source, surfaceWaves=False, Q=None, wavespeed_kms=None, peakf=None):
        corrected_dataframes = self.reduce(inventory, source, surfaceWaves=surfaceWaves, Q=Q, wavespeed_kms=wavspeed_kms, fixpeakf=peakf)
        if surfaceWaves:
            return VRS(dataframes=corrected_dataframes)
        else:
            return VR(dataframes=corrected_dataframes)
        

class DSAM(VSAM):
    
    @staticmethod
    def check_units(st):
        print('DSAM')
        good_st = obspy.core.Stream()
        for tr in st:
            if 'units' in tr.stats:
                u = tr.stats['units'].upper()
                if u == 'M' or u == 'PA':
                    good_st.append(tr)
        return good_st

    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='DSAM'):
        return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))

    def compute_reduced_displacement(self, inventory, source, surfaceWaves=False, Q=None, wavespeed_kms=2.0, peakf=None):
        corrected_dataframes = self.reduce(inventory, source, surfaceWaves=surfaceWaves, Q=Q, wavespeed_kms=wavespeed_kms, fixpeakf=peakf)
        if surfaceWaves:
            return DRS(dataframes=corrected_dataframes)
        else:
            return DR(dataframes=corrected_dataframes)


class VSEM(VSAM):

    def __init__(self, dataframes=None, stream=None, sampling_interval=60.0, filter=[0.5, 18.0], bands = {'VLP': [0.02, 0.2], 'LP':[0.5, 4.0], 'VT':[4.0, 18.0]}, corners=4, verbose=False):
        ''' Create a VSEM object 
        
            Optional name-value pairs:
                dataframes: Create an VSEM object using these dataframes. Used by downsample() method, for example. Default: None.
                stream: Create an VSEM object from this ObsPy.Stream object.
                sampling_interval: Compute VSEM data using this sampling interval (in seconds). Default: 60
                filter: list of two floats, representing fmin and fmax. Default: [0.5, 18.0]. Set to None if no filter wanted.
                bands: a dictionary of filter bands and corresponding column names. Default: {'VLP': [0.02, 0.2], 'LP':[0.5, 4.0], 
                    'VT':[4.0, 18.0]}. For example, the default setting creates 3 additional columns for each DataFrame called 
                    'VLP', 'LP', and 'VT', which contain the mean value for each sampling_interval within the specified filter band
                    (e.g. 0.02-0.2 Hz for VLP). If 'LP' and 'VT' are in this dictionary, an extra column called 'fratio' will also 
                    be computed, which is the log2 of the ratio of the 'VT' column to the 'LP' column, following the definition of
                    frequency ratio by Rodgers et al. (2015).
        '''
        self.dataframes = {} 

        if isinstance(dataframes, dict):
            good_dataframes = {}
            for id, df in dataframes.items():
                if isinstance(df, pd.DataFrame):
                    good_dataframes[id]=df
            if len(good_dataframes)>0:
                self.dataframes = good_dataframes
                if verbose:
                    print('dataframes found. ignoring other arguments.')
                return
            else:
                print('no valid dataframes found')
                pass

        if not isinstance(stream, obspy.core.Stream):
            # empty VSEM object
            print('creating blank VSEM object')
            return
        
        good_stream = self.check_units(stream)
        if verbose:
            print('good_stream:\n',good_stream)

        if len(good_stream)>0:
            if good_stream[0].stats.sampling_rate == 1/sampling_interval:
                # no downsampling to do
                for tr in good_stream:
                    df = pd.DataFrame()
                    df['time'] = pd.Series(tr.times('timestamp'))
                    df['mean'] = pd.Series(tr.data) 
                    self.dataframes[tr.id] = df
                return 
            elif good_stream[0].stats.sampling_rate < 1/sampling_interval:
                print('error: cannot compute SAM for a Stream with a tr.stats.delta bigger than requested sampling interval')
                return
            
        for tr in good_stream:
            if tr.stats.npts < tr.stats.sampling_rate * sampling_interval:
                print('Not enough samples for ',tr.id,'. Skipping.')
                continue
            #print(tr.id, 'absolute=',absolute)
            df = pd.DataFrame()
            
            t = tr.times('timestamp') # Unix epoch time
            sampling_rate = tr.stats.sampling_rate
            t = self.reshape_trace_data(t, sampling_rate, sampling_interval)
            df['time'] = pd.Series(np.nanmin(t,axis=1))

            if filter:
                if tr.stats.sampling_rate<filter[1]*2.2:
                    print(f"{tr}: Sampling rate must be at least {filter[1]*2.2:.1f}. Skipping.")
                    continue
                tr2 = tr.copy()
                tr2.detrend('demean')
                tr2.filter('bandpass', freqmin=filter[0], freqmax=filter[1], corners=corners)
                y = self.reshape_trace_data(np.absolute(tr2.data), sampling_rate, sampling_interval)
            else:
                y = self.reshape_trace_data(np.absolute(tr.data), sampling_rate, sampling_interval)
 
            df['energy'] = pd.Series(np.nansum(np.square(y),axis=1)/tr.stats.sampling_rate)

            if bands:
                for key in bands:
                    tr2 = tr.copy()
                    [flow, fhigh] = bands[key]
                    tr2.filter('bandpass', freqmin=flow, freqmax=fhigh, corners=corners)
                    y = self.reshape_trace_data(abs(tr2.data), sampling_rate, sampling_interval)
                    df[key] = pd.Series(np.nansum(np.square(y),axis=1)) 
  
            self.dataframes[tr.id] = df

    @staticmethod
    def check_units(st):
        print('VSEM')
        good_st = obspy.core.Stream()
        for tr in st:
            if 'units' in tr.stats:
                u = tr.stats['units'].upper()
                if u == 'M2/S' or u == 'PA2':
                    good_st.append(tr)
        return good_st  
    
    def reduce(self, inventory, source, Q=None, wavespeed_kms=None, fixpeakf=None):
        # if the original Trace objects had coordinates attached, add a method in SAM to save those
        # in self.inventory. And add to SAM __init___ the possibility to pass an inventory object.
        
        #print(self)
        # Otherwise, need to pass an inventory here.

        if not wavespeed_kms:
            if surfaceWaves:
                wavespeed_kms=2 # km/s
            else:
                wavespeed_kms=3 # km/s
        
        # Need to pass a source too, which should be a dict with name, lat, lon, elev.
        distance_km, coordinates = self.get_distance_km(inventory, source)

        corrected_dataframes = {}
        for seed_id, df0 in self.dataframes.items():
            if not seed_id in distance_km:
                continue
            df = df0.copy()
            this_distance_km = distance_km[seed_id]
            ratio = df['VT'].sum()/df['LP'].sum()
            if fixpeakf:
                peakf = fixpeakf
            else:
                peakf = np.sqrt(ratio) * 4

            net, sta, loc, chan = seed_id.split('.') 
            esc = self.Eseismic_correction(this_distance_km*1000) # equivalent of gsc
            if Q:
                iac = self.compute_inelastic_attenuation_correction(this_distance_km, peakf, wavespeed_kms, Q)
            else:
                iac = 1.0
            for col in df.columns:
                if col in self.get_metrics(): 
                    if col=='VLP':
                        iacvlp = self.compute_inelastic_attenuation_correction(this_distance_km, 0.06, wavespeed_kms, Q)
                        df[col] = df[col] * esc * iacvlp # do i need to multiply by iac**2 since this is energy?
                    else:
                        df[col] = df[col] * esc * iac # do i need to multiply by iac**2 since this is energy?
            corrected_dataframes[seed_id] = df
        return corrected_dataframes
       
    def compute_reduced_energy(self, inventory, source, surfaceWaves=False, Q=None):
        corrected_dataframes = self.reduce(inventory, source, surfaceWaves=surfaceWaves, Q=Q)
        return ER(dataframes=corrected_dataframes)

    @staticmethod
    def Eacoustic_correction(r, c=340, rho=1.2): 
        Eac = (2 * math.pi * r**2) / (rho * c)
        return Eac

    @staticmethod
    def Eseismic_correction(r, c=3000, rho=2500, S=1, A=1): # a body wave formula
        Esc = (2 * math.pi * r**2) * rho * c * S**2/A
        return Esc

    def downsample(self, new_sampling_interval=3600):
        ''' downsample a VSEM object to a larger sampling interval(e.g. from 1 minute to 1 hour). Returns a new VSEM object.
         
            Optional name-value pair:
                new_sampling_interval: the new sampling interval (in seconds) to downsample to. Default: 3600
        '''

        dataframes = {}
        for id in self.dataframes:
            df = self.dataframes[id]
            df['date'] = pd.to_datetime(df['time'], unit='s')
            old_sampling_interval = self.get_sampling_interval(df)
            if new_sampling_interval > old_sampling_interval:
                freq = '%.0fmin' % (new_sampling_interval/60)
                new_df = df.groupby(pd.Grouper(key='date', freq=freq)).sum()
                new_df.reset_index(drop=True)
                dataframes[id] = new_df
            else:
                print('Cannot downsample to a smaller sampling interval')
        return self.__class__(dataframes=dataframes) 
            
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='VSEM'):
        return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))
	    
	    

class DR(SAM):
    def __init__(self, dataframes=None):
 
        self.dataframes = {} 

        if isinstance(dataframes, dict):
            good_dataframes = {}
            for id, df in dataframes.items():
                if isinstance(df, pd.DataFrame):
                    good_dataframes[id]=df
            if len(good_dataframes)>0:
                self.dataframes = good_dataframes
                #print('dataframes found. ignoring other arguments.')
                return
            else:
                pass
                #print('no valid dataframes found')
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='DR'):
	    return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))
	    
    def linearplot(st, equal_scale=False, percentile=None, linestyle='-'):
    	hf = st.plot(handle=True, equal_scale=equal_scale, linestyle=linestyle) #, method='full'); # standard ObsPy plot
    	# change the y-axis so it starts at 0
    	allAxes = hf.get_axes()
    	ylimupper = [ax.get_ylim()[1] for ax in allAxes]
    	print(ylimupper)
    	if percentile:
        	ylimupper = np.array([np.percentile(tr.data, percentile) for tr in st])*1.1
    	# if equal_scale True, we set to maximum scale
    	print(ylimupper)
    	ymax=max(ylimupper)
    	for i, ax in enumerate(allAxes):
            if equal_scale==True:
            	ax.set_ylim([0, ymax])
            else:
            	ax.set_ylim([0, ylimupper[i]])  

    def iceweb_plot(self, metric='median', equal_scale=False, type='log', percentile=None, linestyle='-', outfile=None):
        measurement = self.__class__.__name__
        if measurement[1]=='R':
            if measurement[0]=='D':
                units = f"(cm\N{SUPERSCRIPT TWO})"
            elif measurement[0]=='V':
                units = f"(cm\N{SUPERSCRIPT TWO}/s)"
            subscript = "{%s}" % measurement[1:]
            
            measurement = f"${measurement[0]}_{subscript}$"
        st = self.to_stream(metric=metric)
        for tr in st:
            tr.data = np.where(tr.data==0, np.nan, tr.data)
        if type=='linear':
            linearplot(st, equal_scale=equal_scale, percentile=percentile, linestyle=linestyle)
        elif type=='log':

            plt.rcParams["figure.figsize"] = (10,6)
            fig, ax = plt.subplots()
            for tr in st:  
                t = [this_t.datetime for this_t in tr.times("utcdatetime") ]     
                ax.semilogy(t, tr.data, linestyle, label='%s' % tr.id) #, alpha=0.03) 1e7 is conversion from amplitude in m at 1000 m to cm^2
            ax.format_xdata = mdates.DateFormatter('%H')
            ax.legend()
            plt.xticks(rotation=90)
            plt.ylim((0.2, 100)) # IceWeb plots went from 0.05-30
            plt.yticks([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0], \
                ['0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'])
            #plt.ylabel(r'$D_{RS}$ ($cm^{2}$)')
            #plt.xlabel(r'UTC / each point is max $D_{RS}$ in %d minute window' % (st[0].stats.delta/60))
            #plt.title('Reduced Displacement (%s)\n%s to %s' % (r'$D_{RS}$', t[0].strftime('%d-%b-%Y %H:%M:%S UTC'), t[-1].strftime('%d-%b-%Y %H:%M:%S UTC')))
            plt.ylabel(measurement + units)
            plt.xlabel(r'UTC / each point is max %s in %d minute window' % (measurement, st[0].stats.delta/60))
            plt.title('Reduced Displacement (%s)\n%s to %s' % (measurement, t[0].strftime('%d-%b-%Y %H:%M:%S UTC'), t[-1].strftime('%d-%b-%Y %H:%M:%S UTC')))            
            plt.xticks(fontsize=6)
            if outfile:
            	plt.savefig(outfile)
            else:
            	plt.show()
     
    def max(self, metric='rms'):
        lod = []
        #print(type(self))
        if metric=='rms' and not 'rms' in self.get_metrics():
            metric='std'
        allmax = []
        classname = self.__class__.__name__
        for seed_id in self.dataframes:
            df = self.dataframes[seed_id]
            thismax = df[metric].max()
            if thismax == 0 or np.isnan(thismax):
                continue
            #print(f"{seed_id}: {thismax:.1e} m at 1 km" )
            #maxes[seed_id] = thismax
            allmax.append(thismax)
            thisDict = {'seed_id': seed_id, classname:np.round(thismax,2)}
            lod.append(thisDict)
        allmax = np.array(sorted(allmax))
        medianMax = np.median(allmax) 
        #print(f"Network: {medianMax:.1e} m at 1 km" )   
        networkMax = np.round(medianMax,2)
        thisDict = {'seed_id':'Network', classname:networkMax}
        lod.append(thisDict)
        df = pd.DataFrame(lod)
        display(df)
        return networkMax

    def show_percentiles(self, metric):
        st = self.to_stream(metric=metric)
        #fig, ax = plt.subplots(len(st), 1)
        for idx,tr in enumerate(st):
            if tr.id.split('.')[-1][-1]!='Z':
                continue
            y = tr.data #.sort()
            p = [p for p in range(101)]
            h = np.percentile(y, p)
            #ax[idx].plot(p[1:], np.diff(h))
            #ax[idx].set_title(f'percentiles for {tr.id}')
            plt.figure()
            plt.semilogy(p[1:], np.diff(h))
            plt.title(f'percentiles for {tr.id}')
                    
    def examine_spread(self, low_percentile=50, high_percentile=98):
        medians = {}
        station_corrections = {}
        metrics = self.get_metrics()
        for bad_metric in ['min', 'max', 'fratio']:
            if bad_metric in metrics:
                metrics = metrics.drop(bad_metric)       
        seed_ids = self.get_seed_ids()
        for metric in metrics: 
            st = self.to_stream(metric=metric)
            medians[metric] = {}
            station_corrections[metric] = {}
            m_array = []
            for tr in st:
                y = np.where( (tr.data>np.percentile(tr.data, low_percentile)) & (tr.data<np.percentile(tr.data, high_percentile)), tr.data, np.nan)
                m = np.nanmedian(y)
                medians[metric][tr.id] =m 
                m_array.append(y)
            m_array=np.array(m_array)
            medians[metric]['network_median'] = np.nanmedian(m_array)
            s_array = []
            for tr in st:
                s = medians[metric]['network_median']/medians[metric][tr.id]
                station_corrections[metric][tr.id] = s
                s_array.append(s)
            
            s_array = np.array(s_array)
            for idx, element in enumerate(s_array):
                if element < 0.1 or element > 10.0:
                    s_array[idx]=np.nan
                if element < 1.0:
                    s_array[idx]=1.0/element     
            station_corrections[metric]['network_std'] = np.nanstd(s_array)
 
            print('\nmetric: ', metric)
            for seed_id in seed_ids:
                print(f"{seed_id}, median: {medians[metric][seed_id]:.3e}, station correction: {station_corrections[metric][seed_id]:.3f}")
            print(f"network: median: {medians[metric]['network_median']:.03e}, station correction std: {station_corrections[metric]['network_std']:.03e}") 
        return medians, station_corrections
                
    def apply_station_corrections(self, station_corrections):
        for seed_id in self.dataframes:
            df = self.dataframes[seed_id]
            for metric in self.get_metrics():
                if metric in station_corrections:
                    if seed_id in station_corrections[metric]: 
                        df[metric] = df[metric] * station_corrections[metric][seed_id]
                    
                      
    def compute_average_dataframe(self, average='mean'):
        ''' 
        Average a SAM object across the whole network of seed_ids
        This is primarily a tool for then making iceweb_plot's with just one representative trace
        It is particularly designed to be used after running examine_spread and apply_station_corrections
        '''
        df = pd.DataFrame()
        for metric in self.get_metrics():
            st = self.to_stream(metric)
            df['time'] = self.dataframes[st[0].id]['time']
            all_data_arrays = []
            for tr in st:
                all_data_arrays.append(tr.data)
            twoDarray = np.stac
#import pytzk(all_data_arrays)
            if average=='mean':
                df[metric] = pd.Series(np.nanmean(y,axis=1))  
            elif average=='median':
                df[metric] = pd.Series(np.nanmedian(y,axis=1))
            net, sta, loc, chan = st[0].id.split('.')
            average_id = '.'.join(net, 'AVRGE', loc, chan)
            dataframes[average_id] = df
        return self.__class__(dataframes=dataframes)  
            
        
class DRS(DR):
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='DRS'):
	    return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))
	    
class VR(DR):
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='VR'):
	    return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))
	    
class VRS(DR):
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='VRS'):
	    return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))	   
	    	    
class ER(DR):

    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='ER'):
	    return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))	   
       
    def sum_energy(self, startt=None, endt=None, metric='energy'): #, inventory, source):
        st = self.to_stream(metric)
        if startt and endt:
            st.trim(starttime=startt, endtime=endt)
        #r_km, coords = self.get_distance_km(inventory, source)
        lod = []
        allE = []
        allM = []
        for tr in st:
            
            #r = r_km[tr.id] * 1000.0
            e = np.nansum(tr.data)
            if e==0:
                continue
            m = np.round(energy2magnitude(e),2)
            allE.append(e)
            allM.append(m)
            print(f"{tr.id}: Joules: {e:.2e}, Magnitude: {m:.1f}")
            thisDict = {'seed_id':tr.id, 'Energy':e, 'EMag':m}
            lod.append(thisDict)

        medianE = np.median(allE)
        medianM = np.round(np.median(allM),2)
        print(f"Network: Joules: {medianE:.2e}, Magnitude: {medianM:.1f}")
        thisDict = {'seed_id':'Network', 'Energy':medianE, 'EMag':medianM}
        lod.append(thisDict)
        df = pd.DataFrame(lod)  
        return medianE, medianM

    #def plot():
    #    pass
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='ER'):
	    return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))	   

def magnitude2energy(mag, a=-4.7, b=2/3):
    ''' 
    Convert (a vector of) magnitude into (a vector of) equivalent energy(ies).
   
    Conversion is based on the equation 7 Hanks and Kanamori (1979):
 
       mag = 2/3 * log10(energy) - 4.7
 
    That is, energy (Joules) is roughly proportional to the peak amplitude to the power of 1.5.
    This obviously is based on earthquake waveforms following a characteristic shape.
    For a waveform of constant amplitude, energy would be proportional to peak amplitude
    to the power of 2.
 
    For Montserrat data, when calibrating against events in the SRU catalog, a factor of
    a=-3.7 was preferred to a=-4.7.
    
    based on https://github.com/geoscience-community-codes/GISMO/blob/master/core/%2Bmagnitude/mag2eng.m
 
    '''

    eng = np.power(10, (1/b * mag - a ))
    return eng 
    
    
def energy2magnitude(eng, a=-3.2, b=2/3):
    '''
    Convert a (vector of) magnitude(s) into a (vector of) equivalent energy(/ies).
    
    Conversion is based on the the following formula from Hanks and Kanamori (1979):
 
       mag = 2/3 * log10(energy) - 4.7
 
    That is, energy (Joules) is roughly proportional to the peak amplitude to the power of 1.5.
    This obviously is based on earthquake waveforms following a characteristic shape.
    For a waveform of constant amplitude, energy would be proportional to peak amplitude
    to the power of 2.
 
    For Montserrat data, when calibrating against events in the SRU catalog, a factor of
    3.7 was preferred to 4.7.

    based on https://github.com/geoscience-community-codes/GISMO/blob/master/core/%2Bmagnitude/eng2mag.m
 
    '''
    
    mag = b * (np.log10(eng) + a) 	
    return mag
    
if __name__ == "__main__":
    pass
