#!/usr/bin/env python

import sys
import numpy as np
from obspy import read_inventory
from obspy.core import UTCDateTime, read, Trace, AttribDict
from obspy.signal.rotate import rotate2zne
from obspy.signal.trigger import classic_sta_lta, plot_trigger
#from pyquaternion import Quaternion




def get_data(stream1, stream2, utctime, duration, seis_channel, rot_channel, inventory, ch_r, ch_s, sim=False, corr_up=False):
    """
    This method reads in data from two file names and does basic pre-processing:
        1. sort the channels
        2. remove the response
        3. rotate to zne-system if required
        4. cut out reqired time span
        5. select source and reciever channles
    
    .. type stream1: string
    .. param stream1: full path to data recorded on 'seis_channel'
    .. type stream2: string
    .. param stream2: full path to data recorded on 'rot_channel'
    .. type utctime: string
    .. param utctime: start time of record to be analysed,
                      format: YYYY-MM-DDThh:mm:ss
    .. type duration: float
    .. param duration: length of time span to by analysed in sec
    .. type seis_channel: string
    .. param seis_channel: channel(s) containing seismometer recordings
    .. type rot_channel: string
    .. param rot_channel: channel(s) containing rotation rate recordings
    .. type inventory1: string
    .. param inventory1: path to *.xml file containing response information
    .. type ch_r: string
    .. param ch_r: reciever channel (data to be corrected)
    .. type ch_s: string
    .. param ch_s: source channel (data to correct for)
    .. type sim: bolean
    .. param sim: set to True if data from simulation is used
    .. type corr_up: bolean
    .. param corr_up: set to True if you want to correct for up-down component, too
    
    return:
    .. type r: obspy.Stream
    .. param r: resciever channel
    .. type s: obspy.Stream
    .. param s: source channel
    .. type r_u: obspy.Stream
    .. param r_u: resciever up-down channel
    """
# define some parameters
    
    p = 0.1
    dt = p * duration

    p2 = 0.03
    dt2 = p2 * duration    

    inv = read_inventory(inventory)
    
    t = UTCDateTime(utctime)

    l = -1

# process the classic seismometer records
# 1. read in the records and sort the channels
# 2. remove response and out put velocity
# 3. rotate the components, compensate for missaligment
    chan1 = seis_channel
    sz1 = read(stream1, starttime=t-dt, endtime=t+duration+dt)

    ##################################################################
    #sz1.write('TC120_HH.mseed', format='MSEED')
    
    sz1.sort()
    if not sim:
        sz1.attach_response(inv)
        sz1.remove_response(water_level=10, output='VEL')
    if "*" in chan1:
        for tr in sz1:
            for channel in inv[0].stations[0].channels:
                if tr.stats.channel == channel.code:
                    tr.stats.align = AttribDict(dict(azimuth=channel.azimuth,
                                      dip=channel.dip))
                    break
    
        z,n,e = rotate2zne(sz1[0].data,sz1[0].stats.align.azimuth,sz1[0].stats.align.dip,\
                        sz1[1].data,sz1[1].stats.align.azimuth,sz1[1].stats.align.dip,\
                        sz1[2].data,sz1[2].stats.align.azimuth,sz1[2].stats.align.dip)

        sz1[0].data = z
        data = (sz1[0].data)
        offset_Z = 0
        if not sim:
            offset_Z = np.mean(data[:l])
        sz1[0].data = data - offset_Z
        sz1[0].stats.channel = "HHZ"
        sz1[1].data = n
        data = (sz1[1].data)
        offset_N = 0
        if not sim:
            offset_N = np.mean(data[:l])
        sz1[1].data = data - offset_N
        sz1[1].stats.channel = "HHN"
        sz1[2].data = e
        data = (sz1[2].data)
        offset_E = 0
        if not sim:
            offset_E = np.mean(data[:l])
        sz1[2].data = data - offset_E
        sz1[2].stats.channel = "HHE"
    else:
        data = (sz1[0].data)
        offset = 0
        if not sim:
            offset = np.mean(data[:l])
        sz1[0].data = data - offset
    
    sz1.taper(0.1, side='left')
    df1 = sz1[0].stats.sampling_rate
    npts1 = sz1[0].stats.npts

    
    
# process the rotation rate records
# 1. read in the records and sort the channels
# 2. remove response (scale by sensitivity) and out put rotation rate (VEL)
# 3. rotate the components, compensate for missaligment
    chan2 = rot_channel
    sz2 = read(stream2, starttime=t-dt, endtime=t+duration+dt)

    ###############################################################
    #sz2.write('BS1_HJ.mseed', format='MSEED')
    
    sz2.sort()
    if not sim:
        sz2.attach_response(inv)
        sz2.remove_response(output='VEL')
    
    if "*" in chan2:
        for tr in sz2:
            for channel in inv[0].stations[1].channels:
                if tr.stats.channel == channel.code:
                    tr.stats.align = AttribDict(dict(azimuth=channel.azimuth,
                                      dip=channel.dip))
                    break
    
        z,n,e = rotate2zne(sz2[0].data,sz2[0].stats.align.azimuth,sz2[0].stats.align.dip,\
                        sz2[1].data,sz2[1].stats.align.azimuth,sz2[1].stats.align.dip,\
                        sz2[2].data,sz2[2].stats.align.azimuth,sz2[2].stats.align.dip)


        sz2[0].data = z
        data = (sz2[0].data)
        offset_Z = 0
        if not sim:
            offset_Z = np.mean(data[:l])
        sz2[0].data = data - offset_Z
        sz2[0].stats.channel = "HJZ"
        sz2[1].data = n
        data = (sz2[1].data)
        offset_N = 0
        if not sim:
            offset_N = np.mean(data[:l])
        sz2[1].data = data - offset_N
        sz2[1].stats.channel = "HJN"
        sz2[2].data = e
        data = (sz2[2].data)
        offset_E = 0
        if not sim:
            offset_E = np.mean(data[:l])
        sz2[2].data = data - offset_E
        sz2[2].stats.channel = "HJE"
    else:
        data = (sz2[0].data)
        offset = 0
        if not sim:
            offset = np.mean(data[:l])
        sz2[0].data = data - offset
    
    sz2.taper(0.1, side='left')
    df2 = sz2[0].stats.sampling_rate
    npts2 = sz2[0].stats.npts

    
    sz1.trim(t+dt2, t+duration-dt2)
    sz2.trim(t+dt2, t+duration-dt2)

# do sanity checks
# 1. check for sampling rate
# 2. check for number of samples
    if df1 != df2:
        print("Sampling rates not the same, exit!!")
        sys.exit(1)
    
    if npts1 != npts2:
        print("Number of data points not the same, exit!!")
        sys.exit(1)

# return the reciever and the source channel
    r = sz1.select(channel=ch_r)
    s = sz2.select(channel=ch_s)
    
    if not corr_up:
        return r, s
    if corr_up:
        r_u = sz1.select(channel='HHZ')
        return r, s, r_u

def get_data_stromboli(stream1, stream2, utctime, duration, seis_channel, rot_channel, inventory1, inventory2, ch_r, ch_s, sim=False, corr_up=False):
    """
    This method reads in data from two file names and does basic pre-processing:
        1. sort the channels
        2. remove the response
        3. rotate to zne-system if required
        4. cut out reqired time span
        5. select source and reciever channles
    
    .. type stream1: string
    .. param stream1: full path to data recorded on 'seis_channel'
    .. type stream2: string
    .. param stream2: full path to data recorded on 'rot_channel'
    .. type utctime: string
    .. param utctime: start time of record to be analysed,
                      format: YYYY-MM-DDThh:mm:ss
    .. type duration: float
    .. param duration: length of time span to by analysed in sec
    .. type seis_channel: string
    .. param seis_channel: channel(s) containing seismometer recordings
    .. type rot_channel: string
    .. param rot_channel: channel(s) containing rotation rate recordings
    .. type inventory1: string
    .. param inventory1: path to *.xml file containing response information
    .. type inventory2: string
    .. param inventory2: path to *.xml file containing response information
    .. type ch_r: string
    .. param ch_r: reciever channel (data to be corrected)
    .. type ch_s: string
    .. param ch_s: source channel (data to correct for)
    .. type sim: bolean
    .. param sim: set to True if data from simulation is used
    .. type corr_up: bolean
    .. param corr_up: set to True if you want to correct for up-down component, too
    
    return:
    .. type r: obspy.Stream
    .. param r: resciever channel
    .. type s: obspy.Stream
    .. param s: source channel
    .. type r_u: obspy.Stream
    .. param r_u: resciever up-down channel
    """

    df = 50.
    p = 0.1
    #dt = p * duration
    dt = 12

    p2 = 0.03
    dt2 = p2 * duration    

    t = UTCDateTime(utctime)

    inv1 = read_inventory(inventory1)
    sz1 = read(stream1, starttime=t-dt, endtime=t+duration+dt)
    if not sim:
        sz1.attach_response(inv1)
    sz1.merge()
    sz1.sort()
    sz1.reverse()
    #sz1.detrend("simple")
    sz1.detrend("linear")
    #sz1.detrend("demean")
    #sz1.taper(0.1)
    sz1.remove_response(water_level=60,output="ACC")
    sz1.rotate(method="->ZNE",inventory=inv1,components=["ZNE"])
    sz1.filter('lowpass', freq=45.0, corners=8, zerophase=True)
    sz1.resample(sampling_rate=df)

    df1 = sz1[0].stats.sampling_rate
    npts1 = sz1[0].stats.npts
    start1 = sz1[0].stats.starttime
    end1 = sz1[0].stats.endtime


    inv2 = read_inventory(inventory2)
    sz2 = read(stream2, starttime=t-dt, endtime=t+duration+dt)
    if not sim:
        sz2.attach_response(inv2)
    sz2.merge()
    sz2.sort()
    sz2.remove_sensitivity()
    #sz2.detrend("simple")
    #sz2.detrend("linear")
    sz2.detrend("demean")
    #sz2.taper(0.1)
    sz2.filter('lowpass', freq=45.0, corners=8, zerophase=True)
    sz2.resample(sampling_rate=df)
    #sz2.interpolate(sampling_rate=df,starttime=sz1[0].stats.starttime+1)
    sz2.rotate(method="->ZNE",inventory=inv2,components=["321"])
    start2 = sz2[0].stats.starttime
    end2 = sz2[0].stats.endtime
####### attitude corr ################
    #sz2.integrate()
    #z = sz2[0].data
    #n = sz2[1].data
    #e = sz2[2].data

    #zc, nc, ec = corr_attitude(z, n, e)

    #sz2[0].data = zc
    #sz2[1].data = nc
    #sz2[2].data = ec
    #sz2.differentiate()
########################################
    #print(inv2.get_coordinates("XS.TOR2..HJ1", t))

    if start2 < start1:
        start2 = start1;
    else:
        start1 = start2;

    if end2 > end1:
        end2 = end1
    else:
        end1 = end2

    sz1.trim(start1,end1)
    sz2.trim(start2,end2)

    df1 = sz1[0].stats.sampling_rate
    npts1 = sz1[0].stats.npts
    df2 = sz2[0].stats.sampling_rate
    npts2 = sz2[0].stats.npts
   

    if df1 != df2:
        print("Sampling rates not the same, exit!!"); sys.exit(1)

    if npts1 != npts2:
        print("Number of data points not the same, exit!!"); sys.exit(1)


    # return the reciever and the source channel
    r = sz1.select(channel=ch_r)
    s = sz2.select(channel=ch_s)

    return r, s
 

def ypr2quat1D(yaw, pitch, roll):
# navigation uses yaw (Z), pitch (Y/N), roll (X/E)
    cy = np.cos(yaw * 0.5);
    sy = np.sin(yaw * 0.5);
    cp = np.cos(pitch * 0.5);
    sp = np.sin(pitch * 0.5);
    cr = np.cos(roll * 0.5);
    sr = np.sin(roll * 0.5);

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    q = np.asarray((w, x, y, z))
    return q

        
def ypr2rotmatrix1D(yaw, pitch, roll):
    theta = roll
    phi = pitch
    psi = yaw
    M = np.array([
    [np.cos(theta)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi)],
    [np.cos(theta)*np.sin(phi), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi)],
    [-np.sin(theta), np.sin(psi)*np.cos(theta), np.cos(psi)*np.cos(theta)]])
    return M
    
def ypr2UNE1D(yaw, pitch, roll, M):
    v = np.array([yaw, roll, pitch])
    _v = np.matmul(M, v)
    new = np.array(_v)
    return new[0], new[2], new[1]

def corr_attitude(z, n, e):
    zref = np.mean(z[:100])
    nref = np.mean(n[:100])
    eref = np.mean(e[:100])
    z_new = [zref]
    n_new = [nref]
    e_new = [eref]
    M0 = ypr2rotmatrix1D(zref, nref, eref)
    M = [M0]
    for i in range(len(z)-1):
        _M = ypr2rotmatrix1D(z[i+1], n[i+1], e[i+1])
        M.append(np.matmul(M[i], _M))
        _z, _n, _e = ypr2UNE1D(z[i+1], n[i+1], e[i+1], np.transpose(M[i]))
        z_new.append(_z)
        n_new.append(_n)
        e_new.append(_e)

    return np.array(z_new), np.array(n_new), np.array(e_new)



def trigger(tr1, a, b, d0, d1, c_on, c_off, start, stop, plot_flagg=False):
    """
    This method searches for time spans when the steps are performed.
    STA/LTA- trigger is used to calculate the characteristic function.
    A constant offset can be applied bacause the steps are uniform.
    
    .. type tr1: obspy.Trace
    .. param tr1: rotation rate recording containing steps
    .. type a: int
    .. param a: number of samples for short term average
    .. type b: int
    .. param b: number of samples for long term average
    .. type d0: float
    .. param d0: threshold for trigger-on
    .. type d1: float
    .. param d1: threshold for trigger-off
    .. type c_on: float
    .. param c_on: constant correction for trigger at the start of each step in sec
    .. type c_off: float
    .. param c_off: constant correction for trigger at the end of each step in sec
    .. type start: float
    .. param start: offset in sec to start searching for steps
    .. type stop: float
    .. param stop: offset in sec to stop searching for steps
    
    return:
    .. type on: list
    .. param on: start time of each step
    .. type off: list
    .. param off: end time of each step
    """
# define some parameters
    data1 = tr1.data
    df1 = tr1.stats.sampling_rate

# get the characteristic function
    cft1 = classic_sta_lta(data1, int(a), int(b))

# you can plot it if you want
    if plot_flagg:
        plot_trigger(tr1, cft1, d0, d1)

# find the on/off time stamps of each step
    _on = np.where(cft1 < d0)[0]
    _off = np.where(cft1 > d1)[0]
    
    on = []
    on0 = 0
    for i in range(len(_on)-1):
        if _on[i+1] - _on[i] > 1:
            trigg = _on[i]*tr1.stats.delta
            if trigg >= start and trigg <= stop:
                if np.abs((trigg + c_on)-on0) > 1.0:
                    on.append(trigg + c_on)
                    on0 = trigg + c_on
    off = []
    off0 = 0
    for i in range(len(_off)-1):
        if _off[i+1] - _off[i] > 1:
            trigg = _off[i]*tr1.stats.delta
            if trigg >= start and trigg <= stop:
                if np.abs((trigg + c_off)-off0) > 1.0:
                    off.append(trigg + c_off)
                    off0 = trigg + c_off

    return on, off


def find_nearest(t, data, on, off):
    """
    This method finds the nearest sample in 'data' to 'on' and 'off'
    .. type t: numpy.array
    .. param t: array containing timestamps of samples in data
    .. type data: numpy.array
    .. param data: data array where the nearest samples should be found
    .. type on: float
    .. param on: time stamp found with method 'trigger()'
    .. type off: float
    .. param off: time stamp found with method 'trigger()'
    
    return:
    .. type idx_on: int
    .. param idx_on: index of first sample in step
    .. type idx_off: int
    .. param idx_off: index of last sample in step
    .. type data[idx_on]: float
    .. param data[idx_on]: corresponding data point
    .. type data[idx_off]: float
    .. param data[idx_off]: corresponding data point
    """
    idx_on = (np.abs(t-on)).argmin()
    idx_off = (np.abs(t-off)).argmin()
    return idx_on, idx_off, data[idx_on], data[idx_off]



def calc_residual_disp(tr1, on, off, r, theo=False):
    """
    This method calculates the residual displacement (lateral displacement
    introduced by the tilt motion) which is left over after tilt correction.

    .. type tr1: obspy.Trace
    .. param tr1: trace containing tilt corrected velocity recording
    .. type on: list
    .. param on: list of time stamps found with method 'trigger()'
    .. type off: list
    .. param off: list of time stamps found with method 'trigger()'
    .. type r: numpy.array
    .. param r: array containing therotetical residual displacement. This is only used to shift the traces
                to make a nicer plot.
    .. type theo: bolean
    .. param theo: set True if theoretical displacement is calculated

    return:
    .. type time: list
    .. param time: list containing time stamps of each step
    .. type disp: list
    .. param disp: list containing residual displacement for each step
    .. type mean_tr: float
    .. param mean_tr: geometric mean value of 'disp'
    .. type sigma_tr: float
    .. param sigma_tr: standard deviation of 'disp'
    """
    disp_tr = []
    disp = []
    time = []

    t = np.arange(len(tr1[0].data))/(tr1[0].stats.sampling_rate)


    for i in range(len(on)):
        idx_on, idx_off, d_0, d_1 = find_nearest(t, tr1[0].data, on[i], off[i])

        data = tr1[0].data[idx_on:idx_off]
        stats = tr1[0].stats
        stats.starttime = tr1[0].stats.starttime+idx_on*tr1[0].stats.delta
        tr = Trace(data=data, header=stats)

        # suppose that velocity is zero at the beginning and at the end of a step
        if not theo:
            tr.detrend('linear')
        y0 = tr.data[0]
        tr.data = tr.data - y0
        
        # integrate to displacement
        tr.integrate()
        
        # shift the whole trace to make it comparable to theoretical displacement
        y0 = tr.data[0]
        diff = (y0 - r[idx_on])
        tr.data = tr.data - diff

        disp.append(tr.data)
        time.append(t[idx_on:idx_off])
        
        disp_tr.append(np.abs(max(tr.data)-min(tr.data)))

    mean_tr = np.mean(disp_tr)
    sigma_tr = np.std(disp_tr)

        
    return time, disp, mean_tr, sigma_tr

def get_angle(st, on, off):
    """
    This method calculates the absolute angle for each step
    
    .. type st: obspy.Stream
    .. param st: stream containing integrated rotation rate data (angle)
    .. type on: list
    .. param on: list of time stamps found with method 'trigger()'
    .. type off: list
    .. param off: list of time stamps found with method 'trigger()'
    
    return:
    .. type : numpy.array
    .. param : array containing absolute angle for each step 
    """
    t = np.arange(len(st[0].data))/(st[0].stats.sampling_rate)
    alpha = []
    for i in range(len(on)):
        idx_on, idx_off, d_0, d_1 = find_nearest(t, st[0].data, on[i], off[i])
        alpha.append(np.abs(d_0 - d_1))
    return np.asarray(alpha)


def theo_resid_disp(alpha0, l, h, dh, rr):
    """
    This method calculates the theoretical residual displacement
    induced by a tilt movement of the angle alpha0
    
    .. type alpha0: numpy.array
    .. param alpha0: integrated rotation rate recording (angle)
    .. type l: float
    .. param l: horizontal distance between axis of rotation and center of seismometer [m]
    .. type h: float
    .. param h: vertical distance between bottom of seismometer and seismometer mass [m]
    .. type dh: float
    .. param dh: vertical distance between bottom of seismometer and axis of rotation [m]

    return:
    .. type r: numpy.array
    .. param r: array containing theoretical residual displacement
    """
    x = l * (1. - np.cos(alpha0))
    y = (dh + h) * np.cos((np.pi/2.) - alpha0)
    r = -1*(x + y)
    c = np.sqrt(l**2 + (dh+h)**2) * rr**2
    return r, c


def calc_height_of_mass(disp, l, dh, alpha):
    """
    This method calculates the vertical distance between the bottom
    of the seismometer and the seismometer mass from the residual displacement. 
    
    .. type disp: list
    .. param disp: list containing residual displacements for each step from 'calc_residual_disp()'
    .. type l: float
    .. param l: horizontal distance between axis of rotation and center of seismometer [m]
    .. type dh: float
    .. param dh: vertical distance between bottom of seismometer and axis of rotation [m]
    .. type alpha: numpy.array
    .. param alpha: rotation angles for each step from 'get_angle()'

    return:
    .. type : float, float
    .. param : mean and standard deviation of vertical distance between the bottom
                of the seismometer and the seismometer mass
    """
    alpha0 = alpha
    X = l * (1. - np.cos(alpha0))
    A = disp - X
    B = A / np.cos(alpha0)
    h = np.tan((np.pi/2.) - alpha0) * B - dh
    
    return np.mean(h), np.std(h)
