# Based on Eibl, E. P. S., Müller, D., Walter, T. R., Allahbakhshi, M., Jousset, P., Hersir, G. P., Dahm, T., (2021) Eruptive Cycle and Bubble Trap of Strokkur Geyser, Iceland, Journal of Geophysical Research: Solid Earth, 126, DOI: 10.1029/2020JB020769
# author: Eva Eibl
# 11/1/2024
# ------------------------------------------------------------------------------
# we import packages
import matplotlib.animation as animation
from obspy.core import read, UTCDateTime
import matplotlib.dates as mdates
from obspy import read_inventory
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# we set paths
filen = "/data/data/code_python/skience2024/Wavefield_polarisation/2018_Strokkur/VI.*"
respfolder = "/data/data/code_python/skience2024/Wavefield_polarisation/2018_Strokkur/"
localpath = "/data/data/13_Strokkur/"

# we set station coordinates
statlist = ["S1", "S2", "S3", "S4", "S5"]
Easting = [533814.8959, 533822.2077, 533826.0672, 533814.4101, 533826.5201]
Northing = [7132036.3130, 7132037.5280, 7132045.5080, 7132031.8781, 7132031.3270]
depthstat = [182.7, 182.853, 183.083, 182.389, 182.227]
Strokkurlat = [7132048, 7132048]
Strokkurlon = [533820, 533820]
Strokkurdep = [183.2, 172]

# we define filter, projection and axis limits
prefilt = [0.05, 0.06, 170, 175]
fmin = 2
fmax = 9
ampl_lim = 0.0000013
dimension = "2D"

# we initialise the figure
fig = plt.figure(figsize=(7.48, 1.2 * 7.48))
mpl.rcParams.update({"font.size": 9})


def animate(frame):
    plt.clf()
    ii = frame

    # Please use 'UTCDateTime' to set tstart to 0:10:55.2 on 10 June 2018. Note that tstart should increase by 1 s in each iteration.  
    tstart = UTCDateTime(2018, 6, 10, 0, 10, 55.2) + ii * 1

    # Please define the endtime so that the time window is 1 s long.
    tend = tstart + 1

    # we define a wider time window for the plotting
    tstart_early = tstart - 1 * 30
    tend_late = tend + 1 * 30

    # Please read in the seismic data using the 'read' function 
    st = read(filen, starttime=tstart_early, endtime=tend_late)

    # we do further preprocessing 
    print(st)
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.07, type="cosine")
    st.taper(max_percentage=0.07, type="cosine")

    # we remove the instrument response using a stationxml file
    for tr in st:
        inv = read_inventory(respfolder + "Stations_S.xml")
        pre_filt = [prefilt[0], prefilt[1], prefilt[2], prefilt[3]]
        tr.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL")

    # we do further preprocessing 
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, type="cosine")
    st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)

    # we save a copy of the not trimmed data for plotting
    st_notrim = st.copy()
    st.trim(tstart, tend)
    st.sort()

    if dimension == "2D":
        # we plot the waveforms
        ax0 = plt.subplot(4, 1, 1)
        ax0.plot(st_notrim[0].times("matplotlib"), st_notrim[0], "grey", lw=0.7)
        ax0.plot(st[0].times("matplotlib"), st[0], "k", lw=0.8)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=10))

        # we want to plot data from 3 seismometers
        for i in range(3):
            # we initialise the subfigures 
            ax1 = plt.subplot(4, 3, 3 * i + 4)
            ax2 = plt.subplot(4, 3, 3 * i + 5)
            ax3 = plt.subplot(4, 3, 3 * i + 6)

            # please plot the E component (x) against the N component (y)
            ax1.plot(st[3 * i + 0], st[3 * i + 1], "0.3", lw=0.4)

            # we label the axes 
            ax1.set_xlabel("E")
            ax1.set_ylabel("N")

            # please plot the Z component (x) against the N component (y)
            ax2.plot(st[3 * i + 2], st[3 * i + 1], "0.3", lw=0.4)

            # we label the axes 
            ax2.set_xlabel("Z")
            ax2.set_ylabel("N")

            # please plot the E component (x) against the Z component (y)
            ax3.plot(st[3 * i + 0], st[3 * i + 2], "0.3", lw=0.4)

            # we label the axes 
            ax3.set_xlabel("E")
            ax3.set_ylabel("Z")

            # we mark the startpoint of the ground motion
            ax1.plot(st[3 * i + 0][0], st[3 * i + 1][0], "k*", ms=8)
            ax2.plot(st[3 * i + 2][0], st[3 * i + 1][0], "k*", ms=8)
            ax3.plot(st[3 * i + 0][0], st[3 * i + 2][0], "k*", ms=8)

            # we do some axes formatting
            ax1.yaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax1.xaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax1.text(0.0000006, 0.0000006, st[3 * i + 1].stats.station)
            ax1.set_xlim([-ampl_lim, ampl_lim])
            ax1.set_ylim([-ampl_lim, ampl_lim])
            ax2.yaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax2.xaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax2.text(0.0000006, 0.0000006, st[3 * i + 1].stats.station)
            ax2.set_xlim([-ampl_lim, ampl_lim])
            ax2.set_ylim([-ampl_lim, ampl_lim])
            ax3.yaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax3.xaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax3.text(0.0000006, 0.0000006, st[3 * i + 1].stats.station)
            ax3.set_xlim([-ampl_lim, ampl_lim])
            ax3.set_ylim([-ampl_lim, ampl_lim])

        # we save the figures separately, Note if wanted, set 'frame = 1' and uncomment here
        # if i==2:
        #    fig.subplots_adjust(hspace=0.32, wspace = 0.32, bottom=0.07, left=0.10, right=0.99, top=0.96)
        #    savefile = localpath + 'teaching_MESS/'+str(tstart.year) +'_'+ str(tstart.month) +'_'+\
        #                    str(tstart.day) +'_'+str(tstart.hour)+':'+str(tstart.minute)+':'+\
        #                    str(tstart.second)+'.'+str(int(tstart.microsecond/100000))+'-'+\
        #                    str(tend.hour)+':'+str(tend.minute)+':'+str(tend.second)+'.'+\
        #                    str(int(tend.microsecond/100000))+'_'+'Particle_motion_f_' +\
        #                    str(fmin)+ '-' +str(fmax)
        #    plt.savefig(savefile+'.png', format='png', dpi=500)
        #    #plt.show()

    elif dimension == "3D":
        ax4 = plt.gca(projection="3d")
        #ax4.view_init(azim=-90, elev=90) # topview
        fig.suptitle(
            "Start: {}, End: {}".format(st[0].stats.starttime, st[0].stats.endtime)
        )
        for i in range(len(statlist)):
            # please plot the particle motion in 3D at the station locations in space
            # Hint: use e.g. the Easting of a certain station location and 
            #       add the ground motion recorded on the E component to it 
            #       (please multiply the ground motion by 10**7 to make it visible) 
            ax4.plot(
                Easting[i] + np.array(st[3 * i + 0]) * 10**7,
                Northing[i] + np.array(st[3 * i + 1]) * 10**7,
                depthstat[i] + np.array(st[3 * i + 2]) * 10**7,
                "0.3",
                lw=0.4,
            )

            # we mark the startpoint of the ground motion
            ax4.plot(
                [Easting[i] + st[3 * i + 0][0] * 10**7],
                [Northing[i] + st[3 * i + 1][0] * 10**7],
                [depthstat[i] + st[3 * i + 2][0] * 10**7],
                "k*",
                ms=4,
            )

            # please plot the conduit of Strokkur in 3D
            ax4.plot(Strokkurlon, Strokkurlat, Strokkurdep, "b--")

            # we mark and label the stations
            ax4.plot([Easting[i]], [Northing[i]], [depthstat[i]], "k^", markersize=6)
            ax4.text(
                Easting[i] + 1,
                Northing[i] + 1,
                depthstat[i],
                st[3 * i + 1].stats.station,
            )

            # we format and label the axes
            ax4.yaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax4.xaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax4.set_ylabel("Northing (m)")
            ax4.set_xlabel("Easting (m)")
            ax4.set_zlabel("Height (m)")

            # if(i==4):  ## -- saving --
            #    plt.savefig(localpath + 'teaching_MESS/Strokkur_particle_motion.png', format='png', dpi=300) #, transparent=True)
            #    plt.show()
            #    plt.close()

# we create the animation here. If you increase the frames, a longer batch of the
# seismic data will be processed as 1 frame equals a 1 second seismic window.
anim = animation.FuncAnimation(fig, animate, frames=10, interval=1000, blit=False)

# saving to m4 using ffmpeg writer
print("Saving video")
writervideo = animation.FFMpegWriter(fps=10)
anim.save(
    localpath + "teaching_MESS/Geyser_tremor_" + dimension + "...mp4", writer=writervideo
)
plt.close()
