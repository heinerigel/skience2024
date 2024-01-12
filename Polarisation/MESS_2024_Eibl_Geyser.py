# Based on Eibl, E. P. S., Müller, D., Walter, T. R., Allahbakhshi, M., Jousset, P., Hersir, G. P., Dahm, T., (2021) Eruptive Cycle and Bubble Trap of Strokkur Geyser, Iceland, Journal of Geophysical Research: Solid Earth, 126, DOI: 10.1029/2020JB020769
# author: Eva Eibl
# 11/1/2024
# ------------------------------------------------------------------------------
import matplotlib.animation as animation
from obspy.core import read, UTCDateTime
import matplotlib.dates as mdates
from obspy import read_inventory
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

filen = "/data/store/2018_Strokkur/VI.*"
respfolder = "/data/store/2018_Strokkur/"
localpath = "/data/data/13_Strokkur/"
statlist = ["S1", "S2", "S3", "S4", "S5"]
Easting = [533814.8959, 533822.2077, 533826.0672, 533814.4101, 533826.5201]
Northing = [7132036.3130, 7132037.5280, 7132045.5080, 7132031.8781, 7132031.3270]
depthstat = [182.7, 182.853, 183.083, 182.389, 182.227]
Strokkurbaz = [24.68 + 180, 351.03 - 180, 295.88 - 180, 20.37 + 180, 340.17 - 180]
prefilt = [0.05, 0.06, 170, 175]
Strokkurlat = [7132048, 7132048]
Strokkurlon = [533820, 533820]
Strokkurdep = [183.2, 172]
fmin = 2
fmax = 9
array = "S"
pmlim = 0.0000013
dimension = "3D"


fig = plt.figure(figsize=(7.48, 1.2 * 7.48))
mpl.rcParams.update({"font.size": 9})


def animate(frame):
    plt.clf()
    ii = frame
    ## window 5s long => 2 Hz is minimum f of interest
    tstart = UTCDateTime(2018, 6, 10, 0, 10, 55.2) + ii * 1
    tend = UTCDateTime(2018, 6, 10, 0, 10, 56.2) + ii * 1
    t2 = tend + 1 * 30
    t3 = tstart - 1 * 30

    ## -- read in seismic data --
    st = read(filen, starttime=t3, endtime=t2)
    st.merge(method=1, fill_value="interpolate")
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.07, type="cosine")
    print(st)
    st.taper(max_percentage=0.07, type="cosine")
    print("remove instrument correction using stxml")
    for tr in st:
        inv = read_inventory(respfolder + "Stations_S.xml")
        pre_filt = [prefilt[0], prefilt[1], prefilt[2], prefilt[3]]
        tr.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL")
    # end
    st.detrend("demean")
    st.detrend("linear")
    print("filter & co")
    st.taper(max_percentage=0.01, type="cosine")
    st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    st_notrim = st.copy()
    st.trim(tstart, tend)
    st.sort()

    if dimension == "2D":
        # particle motion 2D
        ax0 = plt.subplot(4, 1, 1)
        ax0.plot(st_notrim[0].times("matplotlib"), st_notrim[0], "grey", lw=0.7)
        ax0.plot(st[0].times("matplotlib"), st[0], "k", lw=0.8)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=10))
        for i in range(3):
            ## -- all NE particle motion, all 5 stations --
            ax1 = plt.subplot(4, 3, 3 * i + 4)
            ax1.plot(st[3 * i + 0], st[3 * i + 1], "0.3", lw=0.4)  # N/E
            ax1.plot(st[3 * i + 0][0], st[3 * i + 1][0], "k*", ms=8)  # mark 1
            ax1.plot(st[3 * i + 0][1], st[3 * i + 1][1], "k^", ms=5)  # mark 2
            ## -- format axes --
            ax1.yaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax1.xaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax1.text(0.0000006, 0.0000006, st[3 * i + 1].stats.station)
            ax1.set_xlim([-pmlim, pmlim])
            ax1.set_ylim([-pmlim, pmlim])
            ax1.set_ylabel("N")
            ax1.set_xlabel("E")

            ## -- all NZ particle motion, all 5 stations --
            ax2 = plt.subplot(4, 3, 3 * i + 5)
            ax2.plot(st[3 * i + 2], st[3 * i + 1], "0.3", lw=0.4)  # N/Z
            ax2.plot(st[3 * i + 2][0], st[3 * i + 1][0], "k*", ms=8)  # mark 1
            ax2.plot(st[3 * i + 2][1], st[3 * i + 1][1], "k^", ms=5)  # mark 2
            ## -- format axes --
            ax2.yaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax2.xaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax2.text(0.0000006, 0.0000006, st[3 * i + 1].stats.station)
            ax2.set_xlim([-pmlim, pmlim])
            ax2.set_ylim([-pmlim, pmlim])
            ax2.set_ylabel("N")
            ax2.set_xlabel("Z")

            ## -- all ZE particle motion, all 5 stations --
            ax3 = plt.subplot(4, 3, 3 * i + 6)
            ax3.plot(st[3 * i + 0], st[3 * i + 2], "0.3", lw=0.4)  # Z/E
            ax3.plot(st[3 * i + 0][0], st[3 * i + 2][0], "k*", ms=8)  # mark 1
            ax3.plot(st[3 * i + 0][1], st[3 * i + 2][1], "k^", ms=5)  # mark 2
            ## -- format axes --
            ax3.yaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax3.xaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax3.text(0.0000006, 0.0000006, st[3 * i + 1].stats.station)
            ax3.set_xlim([-pmlim, pmlim])
            ax3.set_ylim([-pmlim, pmlim])
            ax3.set_ylabel("Z")
            ax3.set_xlabel("E")

        ## -- saving --
        # if i==2:
        #    fig.subplots_adjust(hspace=0.32, wspace = 0.32, bottom=0.07, left=0.10, right=0.99, top=0.96)
        #    savefile = localpath + 'teaching_MESS/'+str(tstart.year) +'_'+ str(tstart.month) +'_'+\
        #                    str(tstart.day) +'_'+str(tstart.hour)+':'+str(tstart.minute)+':'+\
        #                    str(tstart.second)+'.'+str(int(tstart.microsecond/100000))+'-'+\
        #                    str(tend.hour)+':'+str(tend.minute)+':'+str(tend.second)+'.'+\
        #                    str(int(tend.microsecond/100000))+'_'+'Particle_motion_f_' +\
        #                    str(fmin)+ '-' +str(fmax)+'_' +array
        #    plt.savefig(savefile+'.png', format='png', dpi=500)
        #    #plt.show()

    elif dimension == "3D":
        ax4 = plt.gca(projection="3d")
        #ax4.view_init(azim=-90, elev=90) # topview
        fig.suptitle(
            "Start: {}, End: {}".format(st[0].stats.starttime, st[0].stats.endtime)
        )
        for i in range(len(statlist)):
            ## -- all 3D particle motion, all 5 stations --
            ax4.plot(
                Easting[i] + np.array(st[3 * i + 0]) * 10**7,
                Northing[i] + np.array(st[3 * i + 1]) * 10**7,
                depthstat[i] + np.array(st[3 * i + 2]) * 10**7,
                "0.3",
                lw=0.4,
            )  # N/E/Z
            ax4.plot(
                [Easting[i] + st[3 * i + 0][0] * 10**7],
                [Northing[i] + st[3 * i + 1][0] * 10**7],
                [depthstat[i] + st[3 * i + 2][0] * 10**7],
                "k*",
                ms=4,
            )  # mark 1
            ax4.plot(
                [Easting[i] + st[3 * i + 0][1] * 10**7],
                [Northing[i] + st[3 * i + 1][1] * 10**7],
                [depthstat[i] + st[3 * i + 2][1] * 10**7],
                "k^",
                ms=4,
            )  # mark 2
            ## -- Strokkur conduit --
            ax4.plot(Strokkurlon, Strokkurlat, Strokkurdep, "b--", label="Strokkur")
            ## -- add stations --
            ax4.plot([Easting[i]], [Northing[i]], [depthstat[i]], "k^", markersize=6)
            ax4.text(
                Easting[i] + 1,
                Northing[i] + 1,
                depthstat[i],
                st[3 * i + 1].stats.station,
            )
            ## -- format axes --
            ax4.set_ylabel("Latitude")
            ax4.set_xlabel("Longitude")
            ax4.set_zlabel("Depth")
            ax4.yaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax4.xaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax4.set_ylabel("Northing (m)")
            ax4.set_xlabel("Easting (m)")
            ax4.set_zlabel("Height (m)")

            # if(i==4):  ## -- saving --
            #    plt.savefig(localpath + 'teaching_MESS/Strokkur_particle_motion.png', format='png', dpi=300) #, transparent=True)
            #    plt.show()
            #    plt.close()


anim = animation.FuncAnimation(fig, animate, frames=10, interval=1000, blit=False)

# saving to m4 using ffmpeg writer
print("Saving video")
writervideo = animation.FFMpegWriter(fps=10)
anim.save(
    localpath + "teaching_MESS/Geyser_tremor_" + dimension + "...mp4", writer=writervideo
)
plt.close()
