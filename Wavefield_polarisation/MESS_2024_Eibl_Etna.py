# Based on Eibl, E. P. S., Rosskopf*, M., Sciotto, M., Currenti, G., Di Grazia, G., Jousset, P., Krüger, F., Weber, M. (2022) Performance of a Rotational Sensor to Decipher Volcano Seismic Signals on Etna, Italy, Journal of Geophysical Research: Solid Earth, 127, e2021JB023617. DOI: 10.1029/2021JB023617
# author: Eva Eibl
# 11/1/2024
# ------------------------------------------------------------------------------

from obspy.core import read, UTCDateTime
import matplotlib.dates as mdates
from obspy import read_inventory
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
import numpy as np

localpath = "/data/data/13_Strokkur/"
datapath = "2019_Etna/"
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
stat = "RS1"

for ii in range(6):
    if ii == 0:  # VT
        tstart = UTCDateTime(2019, 9, 4, 15, 52, 0)
        tend = UTCDateTime(2019, 9, 4, 15, 54, 0)
        julday = 247
    elif ii == 1:  # VT
        # 2019-09-17T18:40:52.400000Z  17-09 18:40:52  3.1 ML  37.735  14.873  4.1  0.2 km SE from Monte Minardo (CT)  260
        tstart = UTCDateTime(2019, 9, 17, 18, 40, 30)
        tend = UTCDateTime(2019, 9, 17, 18, 42, 30)
        julday = 260
    elif ii == 2:  # LP
        tstart = UTCDateTime(2019, 8, 27, 14, 21, 0)
        tend = UTCDateTime(2019, 8, 27, 14, 22, 30)
        julday = 239
    elif ii == 3:  # LP
        tstart = UTCDateTime(2019, 8, 27, 12, 18, 0)
        tend = UTCDateTime(2019, 8, 27, 12, 19, 30)
        julday = 239
    elif ii == 4:  # tremor
        tstart = UTCDateTime(2019, 9, 8, 12, 18, 0)
        tend = UTCDateTime(2019, 9, 8, 12, 20, 0)
        julday = 251
    elif ii == 5:  # tremor
        tstart = UTCDateTime(2019, 9, 9, 12, 18, 0)
        tend = UTCDateTime(2019, 9, 9, 12, 20, 0)
        julday = 252

    t2 = tend + 1 * 30
    t3 = tstart - 1 * 30

    # read in seismic data
    fullpath = datapath + "ZR.RS1..HH*"
    st_trans = read(fullpath, starttime=t3, endtime=t2)
    st_trans.merge(method=1, fill_value="interpolate")
    st_trans.detrend("demean")
    st_trans.detrend("linear")
    st_trans.taper(max_percentage=0.01, type="cosine")
    print(st_trans)
    st_trans.taper(max_percentage=0.01, type="cosine")
    print("remove instrument correction using stxml")
    for tr in st_trans:
        inv = read_inventory(datapath + "Stations_Etna_2019_seis.xml")
        pre_filt = [prefilt[0], prefilt[1], prefilt[2], prefilt[3]]
        tr.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL")
    st_trans.detrend("demean")
    st_trans.detrend("linear")
    print("filter & co")
    st_trans.taper(max_percentage=0.01, type="cosine")
    st_trans.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    st_trans.trim(tstart, tend)
    st_trans.sort()

    # read in rotational sensor data
    fullpath = datapath + "ZR.RS1..HJ*"
    st_rot = read(fullpath, starttime=tstart, endtime=tend)
    st_rot.merge(method=1, fill_value="interpolate")
    st_rot.detrend("demean")
    st_rot.detrend("linear")
    st_rot.taper(max_percentage=0.01, type="cosine")
    print(st_rot)
    st_rot.taper(max_percentage=0.01, type="cosine")
    st_rot.detrend("demean")
    st_rot.detrend("linear")
    print("filter & co")
    st_rot.taper(max_percentage=0.01, type="cosine")
    st_rot.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    st_rot.trim(tstart, tend)
    st_rot[0].data = st_rot[0].data * 1e-9  # unit: rad/s
    st_rot[1].data = st_rot[1].data * 1e-9  # unit: rad/s
    st_rot[2].data = st_rot[2].data * 1e-9  # unit: rad/s
    st_rot.integrate()  # unit: rad
    st_rot.sort()
    st = st_trans + st_rot
    siglen = int(np.floor(st_trans[0].stats.endtime - st_trans[0].stats.starttime))

    # initialise figure
    fig = plt.figure(figsize=(7.48, 8.48))
    mpl.rcParams.update({"font.size": 8})
    mpl.rcParams["pcolor.shading"]
    trans_max = np.abs(st_trans[0].max())
    rot_max = np.abs(st_rot[0].max())
    for i in range(6):
        ## -- plot seismogram --
        ax0 = plt.subplot(6, 2, 1 + 2 * i)
        ax0.plot(st[i].times("matplotlib"), st[i].data, "k", lw=0.4)
        if i < 3:
            ax0.set_ylim(-trans_max, trans_max)
        else:
            ax0.set_ylim(-rot_max, rot_max)

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
        ax0.set_ylabel(f"{st[i].stats.station}: {chan}")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=ticks))

        ## -- plot spectrogram --
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

        if log == True:
            ax1.set_yscale("symlog")

        ax1.set_ylim(fmin, fmax)
        ax1.set_ylabel("Frequency (Hz)")
        if i == 5:
            ax0.set_xlabel(
                f"Time on {(st[i].stats.starttime+2).day}/{st[i].stats.starttime.month}/{st[i].stats.starttime.year} (hh:mm:ss)"
            )
            ax1.set_xlabel(
                f"Time from {(st[i].stats.starttime+2).day}/{st[i].stats.starttime.month}/{st[i].stats.starttime.year} {st[i].stats.starttime.hour}:{st[i].stats.starttime.minute}:{st[i].stats.starttime.second}(s)"
            )

        ## -- adapt axes --
        ax0.tick_params(
            "both", length=5, width=1, which="major", direction="in", top=True
        )
        ax1.tick_params(
            "both", length=5, width=1, which="major", direction="in", top=True
        )
        ## -- remove xaxislabels
        if i != len(st) - 1:
            plt.setp(ax0.get_xticklabels(), visible=False)
            plt.setp(ax1.get_xticklabels(), visible=False)

        ## -- add colorbar --
        if i < 3:
            cbaxes = fig.add_axes([0.927, 0.522, 0.01, 0.45])
            cb = plt.colorbar(img, cax=cbaxes, label="Spectral density ((m/s)$^2$/Hz)")
        else:
            cbaxes = fig.add_axes([0.927, 0.045, 0.01, 0.45])
            cb = plt.colorbar(img, cax=cbaxes, label="Spectral density ((rad)$^2$/Hz)")

        ## -- add labels --
        if i == 0:
            ax0.text(
                0.023,
                0.87,
                "a",
                fontweight="bold",
                fontsize=9,
                zorder=10,
                transform=ax0.transAxes,
            )
            ax1.text(
                0.023,
                0.87,
                "g",
                fontweight="bold",
                fontsize=9,
                backgroundcolor="white",
                zorder=10,
                transform=ax1.transAxes,
            )
        elif i == 1:
            ax0.text(
                0.023,
                0.87,
                "b",
                fontweight="bold",
                fontsize=9,
                zorder=10,
                transform=ax0.transAxes,
            )
            ax1.text(
                0.023,
                0.87,
                "h",
                fontweight="bold",
                fontsize=9,
                backgroundcolor="white",
                zorder=10,
                transform=ax1.transAxes,
            )
        elif i == 2:
            ax0.text(
                0.023,
                0.87,
                "c",
                fontweight="bold",
                fontsize=9,
                zorder=10,
                transform=ax0.transAxes,
            )
            ax1.text(
                0.023,
                0.87,
                "i",
                fontweight="bold",
                fontsize=9,
                backgroundcolor="white",
                zorder=10,
                transform=ax1.transAxes,
            )
        elif i == 3:
            ax0.text(
                0.023,
                0.87,
                "d",
                fontweight="bold",
                fontsize=9,
                zorder=10,
                transform=ax0.transAxes,
            )
            ax1.text(
                0.023,
                0.87,
                "j",
                fontweight="bold",
                fontsize=9,
                backgroundcolor="white",
                zorder=10,
                transform=ax1.transAxes,
            )
        elif i == 4:
            ax0.text(
                0.023,
                0.87,
                "e",
                fontweight="bold",
                fontsize=9,
                zorder=10,
                transform=ax0.transAxes,
            )
            ax1.text(
                0.023,
                0.87,
                "k",
                fontweight="bold",
                fontsize=9,
                backgroundcolor="white",
                zorder=10,
                transform=ax1.transAxes,
            )
        elif i == 5:
            ax0.text(
                0.023,
                0.87,
                "f",
                fontweight="bold",
                fontsize=9,
                zorder=10,
                transform=ax0.transAxes,
            )
            ax1.text(
                0.023,
                0.87,
                "l",
                fontweight="bold",
                fontsize=9,
                backgroundcolor="white",
                zorder=10,
                transform=ax1.transAxes,
            )

    fig.subplots_adjust(
        hspace=0.0, wspace=0.20, bottom=0.042, left=0.08, right=0.890, top=0.985
    )

    ## -- saving --
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
    #plt.savefig(savefile + ".png", format="png", dpi=500)
    plt.show()
