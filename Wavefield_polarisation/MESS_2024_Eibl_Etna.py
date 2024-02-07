# Based on Eibl, E. P. S., Rosskopf*, M., Sciotto, M., Currenti, G., Di Grazia, G., Jousset, P., Krüger, F., Weber, M. (2022) Performance of a Rotational Sensor to Decipher Volcano Seismic Signals on Etna, Italy, Journal of Geophysical Research: Solid Earth, 127, e2021JB023617. DOI: 10.1029/2021JB023617
# author: Eva Eibl
# 11/1/2024
# ------------------------------------------------------------------------------
# we import packages
from obspy.core import read, UTCDateTime
import matplotlib.dates as mdates
from obspy import read_inventory
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
import numpy as np

# we set paths
localpath = "/data/data/13_Strokkur/"
datapath = "/data/data/code_python/skience2024/Wavefield_polarisation/2019_Etna/"

# we define filter and axis limits
prefilt = [0.008, 0.01, 95, 99]
fmin = 0.5
fmax = 20
ticks = 30
wlen = 128
overlap = wlen / 2
vmin1 = 2 * 10**-15
vmax1 = 1.0 * 10**-10
vmin = 2 * 10**-17
vmax = 3.0 * 10**-16
log = False

for ii in range(6):
    if ii == 0:  # VT
        # Please use 'UTCDateTime' to set tstart to 15:52:0 on 4 September 2019.
        tstart = UTCDateTime(2019, 9, 4, 15, 52, 0)
        tend = UTCDateTime(2019, 9, 4, 15, 54, 0)
        julday = 247
    elif ii == 1:  # VT
        # 2019-09-17T18:40:52.400000Z  17-09 18:40:52  3.1 ML  37.735  14.873  4.1  0.2 km SE from Monte Minardo (CT)  260
        tstart = UTCDateTime(2019, 9, 17, 18, 40, 30)
        tend = UTCDateTime(2019, 9, 17, 18, 42, 30)
        julday = 260
    elif ii == 2:  # LP
        # Please use 'UTCDateTime' to set tstart to 14:21:0 on 27 August 2019.
        tstart = UTCDateTime(2019, 8, 27, 14, 21, 0)
        tend = UTCDateTime(2019, 8, 27, 14, 22, 30)
        julday = 239
    elif ii == 3:  # LP
        tstart = UTCDateTime(2019, 8, 27, 12, 18, 0)
        tend = UTCDateTime(2019, 8, 27, 12, 19, 30)
        julday = 239
    elif ii == 4:  # tremor
        # Please use 'UTCDateTime' to set tstart to 12:18:0 on 8 September 2019.
        tstart = UTCDateTime(2019, 9, 8, 12, 18, 0)
        tend = UTCDateTime(2019, 9, 8, 12, 20, 0)
        julday = 251
    elif ii == 5:  # tremor
        tstart = UTCDateTime(2019, 9, 9, 12, 18, 0)
        tend = UTCDateTime(2019, 9, 9, 12, 20, 0)
        julday = 252

    # we define a wider time window for the plotting
    tstart_early = tstart - 1 * 30
    tend_late = tend + 1 * 30

    # we read in the seismometer data
    fullpath = datapath + "ZR.RS1..HH*"
    st_trans = read(fullpath, starttime=tstart_early, endtime=tend_late)

    # we do further preprocessing of the seismometer data
    print(st_trans)
    st_trans.detrend("demean")
    st_trans.detrend("linear")
    st_trans.taper(max_percentage=0.01, type="cosine")
    st_trans.taper(max_percentage=0.01, type="cosine")

    # we remove the instrument response using a stationxml file
    for tr in st_trans:
        inv = read_inventory(datapath + "Stations_Etna_2019_seis.xml")
        pre_filt = [prefilt[0], prefilt[1], prefilt[2], prefilt[3]]
        tr.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL")

    # we do further preprocessing of the seismometer data
    st_trans.detrend("demean")
    st_trans.detrend("linear")
    st_trans.taper(max_percentage=0.01, type="cosine")
    st_trans.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    st_trans.trim(tstart, tend)
    st_trans.sort()

    # we read in the rotational sensor data
    fullpath = datapath + "ZR.RS1..HJ*"
    st_rot = read(fullpath, starttime=tstart, endtime=tend)

    # we do further preprocessing 
    print(st_rot)
    st_rot.detrend("demean")
    st_rot.detrend("linear")
    st_rot.taper(max_percentage=0.01, type="cosine")
    st_rot.taper(max_percentage=0.01, type="cosine")
    st_rot.detrend("demean")
    st_rot.detrend("linear")
    st_rot.taper(max_percentage=0.01, type="cosine")
    st_rot.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    st_rot.trim(tstart, tend)
    st_rot.sort()

    # we convert the rotational sensor rotation rate data from nanorad/s to rad/s, 
    # the response is flat and no further instrument response removal is needed 
    st_rot[0].data = st_rot[0].data * 1e-9
    st_rot[1].data = st_rot[1].data * 1e-9
    st_rot[2].data = st_rot[2].data * 1e-9

    # we integrate to convert the rotational sensor data to rotation
    st_rot.integrate()

    # we merge seismometer and rotational sensor data into one stream object
    st = st_trans + st_rot
    siglen = int(np.floor(st_trans[0].stats.endtime - st_trans[0].stats.starttime))

    # we initialise the figure
    fig = plt.figure(figsize=(7.48, 8.48))
    mpl.rcParams.update({"font.size": 8})
    mpl.rcParams["pcolor.shading"]

    # we derive the y axes limits 
    trans_max = np.abs(st_trans[0].max())
    rot_max = np.abs(st_rot[0].max())

    for i in range(6):
        ax0 = plt.subplot(6, 2, 1 + 2 * i)
        # please plot the seismograms. 
        # Column 1 is from top to bottom: HHE, HHN, HHZ, HJE, HJN, HJZ
        # We want to use the 'matplotlib' date format from 
        # 'obspy.core.trace.Trace.times' for the time axes. 
        ax0.plot(st[i].times("matplotlib"), st[i].data, "k", lw=0.4)

        # we label the y axes
        if st[i].stats.channel == "HHE":
            chan = st[i].stats.channel + " (m/s)"
        if st[i].stats.channel == "HHN":
            chan = st[i].stats.channel + " (m/s)"
        if st[i].stats.channel == "HHZ":
            chan = st[i].stats.channel + " (m/s)"
        if st[i].stats.channel == "HJ1" or st[i].stats.channel == "HJE":
            chan = " HJE (rad)"
        if st[i].stats.channel == "HJ2" or st[i].stats.channel == "HJN":
            chan = " HJN (rad)"
        if st[i].stats.channel == "HJ3" or st[i].stats.channel == "HJZ":
            chan = " HJZ (rad)"

        # we set the y axes limits
        if i < 3:
            ax0.set_ylim(-trans_max, trans_max)
        else:
            ax0.set_ylim(-rot_max, rot_max)
        ax0.set_ylabel(f"{st[i].stats.station}: {chan}")

        # we format the seismometer time axis 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=ticks))

        # we plot the spectrograms
        ax1 = plt.subplot(6, 2, 2 + 2 * i)
        cmap = plt.cm.viridis
        f, t, Sxx = signal.spectrogram(
            st[i].data, st[i].stats.sampling_rate, nperseg=wlen, noverlap=overlap
        )
        tshift = (st[i].stats.starttime - tstart) * st[i].stats.sampling_rate
        if i < 3:
            img = plt.pcolormesh(
                t + tshift / st[i].stats.sampling_rate,
                f,
                Sxx,
                cmap=cmap,
                vmin=vmin1,
                vmax=vmax1,
                shading="gouraud",
            )  # flat is sharper
        else:
            img = plt.pcolormesh(
                t + tshift / st[i].stats.sampling_rate,
                f,
                Sxx,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="gouraud",
            )  # flat is sharper
        ax1.set_ylim(fmin, fmax)
        ax1.set_ylabel("Frequency (Hz)")
        if log == True:
            ax1.set_yscale("symlog")

        # we label seismogram and spectrogram time axes
        if i == 5:
            ax0.set_xlabel(
                f"Time on {(st[i].stats.starttime+2).day}/{st[i].stats.starttime.month}/{st[i].stats.starttime.year} (hh:mm:ss)"
            )
            ax1.set_xlabel(
                f"Time from {(st[i].stats.starttime+2).day}/{st[i].stats.starttime.month}/{st[i].stats.starttime.year} {st[i].stats.starttime.hour}:{st[i].stats.starttime.minute}:{st[i].stats.starttime.second}(s)"
            )

        # we remove some x axis labels
        if i != len(st) - 1:
            plt.setp(ax0.get_xticklabels(), visible=False)
            plt.setp(ax1.get_xticklabels(), visible=False)

        # we add a colorbar
        if i < 3:
            cbaxes = fig.add_axes([0.927, 0.505, 0.01, 0.37])
            cb = plt.colorbar(img, cax=cbaxes, label="Spectral density ((m/s)$^2$/Hz)")
        else:
            cbaxes = fig.add_axes([0.927, 0.11, 0.01, 0.37])
            cb = plt.colorbar(img, cax=cbaxes, label="Spectral density ((rad)$^2$/Hz)")

    # we save the figure
    savefile = (
        localpath
        + "teaching_MESS/seismogram_"
        + str(tstart.year)
        + "_"
        + str(tstart.month)
        + "_"
        + str(tstart.day)
        + "_h"
        + str(tstart.hour)
        + "-"
        + str(tstart.minute)
        + "-"
        + str(tstart.second)
        + "_f"
        + str(fmin)
        + "-"
        + str(fmax)
    )
    plt.savefig(savefile + ".png", format="png", dpi=500)
    plt.show()
