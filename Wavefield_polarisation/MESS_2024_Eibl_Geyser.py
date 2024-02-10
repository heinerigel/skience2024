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
filename = "2018_Strokkur/VI.*"
datafolder = "2018_Strokkur/"
localpath = "figure_output/"

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
dimension = "3D"
# End first cell


# Start second cell
# We read in seismic data
def read_data(tstart, tend, tstart_early, tend_late, fmin, fmax):

    # We read in the seismic data 
    st = read(filename, starttime=tstart_early, endtime=tend_late)

    # we do further preprocessing 
    print(st)
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.07, type="cosine")
    st.taper(max_percentage=0.07, type="cosine")

    # we remove the instrument response using a stationxml file
    for tr in st:
        inv = read_inventory(datafolder + "Stations_S.xml")
        pre_filt = [prefilt[0], prefilt[1], prefilt[2], prefilt[3]]
        tr.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL")

    # we do further preprocessing 
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, type="cosine")
    st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)

    # We save a copy of the not trimmed data for plotting
    st_notrim = st.copy()
    st.trim(tstart, tend)
    st.sort()

    return st, st_notrim
# End second cell


# please plot a seismogram of the trimmed and not trimmed data 
# Step 1: Plot the data of one component. We want to use the 'matplotlib' date 
# format from 'obspy.core.trace.Trace.times' for the time axes.
# Start third cell
# Step 2: set the starttime tstart to 0:10:55.2 on 10 June 2018
# Step 3: call the function that reads the data
# Step 4: Call the function that plots the data
def plot_seismogram(axis, trace, trace_notrim):
    axis.plot(trace_notrim.times("matplotlib"), trace_notrim, "grey", lw=0.7)
    axis.plot(trace.times("matplotlib"), trace, "k", lw=0.8)

tstart = UTCDateTime(2018, 6, 10, 0, 10, 55.2)
tend = tstart + 1
tstart_early = tstart - 1 * 30
tend_late = tend + 1 * 30

stream, stream_notrim = read_data(tstart, tend, tstart_early, tend_late, fmin, fmax)

fig, axis = plt.subplots()
plot_seismogram(axis, stream[0], stream_notrim[0])
#plt.show()
plt.close()
# End third cell


# Start fourth cell
# Step 1: please write a function that plots one trace against another (2D).
# Step 2: define the axes of your plot
# Step 3: plot the particle motion of E against N, Z against N and E against Z
def plot_particle_motion_2D(axis, trace1, trace2):
    # we plot one trace against the other
    axis.plot(trace1, trace2, "0.3", lw=0.4)

    # we label the axes 
    axis.set_xlabel(trace1.stats.channel)
    axis.set_ylabel(trace2.stats.channel)

    # we mark the startpoint of the ground motion
    axis.plot(trace1[0], trace2[0], "k*", ms=8)

ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

plot_particle_motion_2D(ax1, stream[0], stream[1])
plot_particle_motion_2D(ax2, stream[2], stream[1])
plot_particle_motion_2D(ax3, stream[0], stream[2])
#plt.show()
plt.close()
# End fourth cell


# Start fifth cell
# Step 1: please write a function that plots three traces against each other (3D)
# at a station location in space.
# Hint: use e.g. the Easting of a certain station location and 
#       add the ground motion recorded on the E component to it 
#       (please multiply the ground motion by 10**7 to make it visible) 
def plot_particle_motion_3D(axis, stream, Easting, Northing, depthstat):
    axis.plot(
            Easting + np.array(stream[0]) * 10**7,
            Northing + np.array(stream[1]) * 10**7,
            depthstat + np.array(stream[2]) * 10**7,
            "0.3",
            lw=0.4,
        )
    # we mark the startpoint of the ground motion
    axis.plot(
        [Easting + stream[0][0] * 10**7],
        [Northing + stream[1][0] * 10**7],
        [depthstat + stream[2][0] * 10**7],
        "k*",
        ms=4,
    )
    # we mark and label the stations
    axis.plot([Easting], [Northing], [depthstat], "k^", markersize=6)
    axis.text(
        Easting + 1,
        Northing + 1,
        depthstat,
        stream[1].stats.station,
    )

    axis.set_ylabel("Northing (m)")
    axis.set_xlabel("Easting (m)")
    axis.set_zlabel("Height (m)")


ax = plt.gca(projection="3d")
plot_particle_motion_3D(ax, stream.select(station = statlist[0]), Easting[0], Northing[0], depthstat[0])
plt.show()
plt.close()
# End fifth cell



# we initialise the full figure
fig = plt.figure(figsize=(7.48, 1.2 * 7.48))
mpl.rcParams.update({"font.size": 9})

def animate(frame):
    plt.clf()
    ii = frame

    # we use 'UTCDateTime' to set tstart to 0:10:55.2 on 10 June 2018. Note that tstart increases by 1 s in each iteration.  
    tstart = UTCDateTime(2018, 6, 10, 0, 10, 55.2) + ii * 1

    # we define the endtime so that the time window is 1 s long.
    tend = tstart + 1

    # we define a wider time window for the plotting
    tstart_early = tstart - 1 * 30
    tend_late = tend + 1 * 30

    st, st_notrim = read_data(tstart, tend, tstart_early, tend_late, fmin, fmax)

    if dimension == "2D":
        # we plot the waveforms
        ax0 = plt.subplot(4, 1, 1)
        plot_seismogram(ax0, st[0], st_notrim[0])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=10))

        # we want to plot data from 3 seismometers
        for i in range(3):
            # we initialise the subfigures 
            ax1 = plt.subplot(4, 3, 3 * i + 4)
            ax2 = plt.subplot(4, 3, 3 * i + 5)
            ax3 = plt.subplot(4, 3, 3 * i + 6)

            # we plot the E component (x) against the N component (y)
            plot_particle_motion_2D(ax1, st[3 * i + 0], st[3 * i + 1])

            # we plot the Z component (x) against the N component (y)
            plot_particle_motion_2D(ax2, st[3 * i + 2], st[3 * i + 1])

            # we plot the E component (x) against the Z component (y)
            plot_particle_motion_2D(ax3, st[3 * i + 0], st[3 * i + 2])

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
        #    savefile = localpath + str(tstart.year) +'_'+ str(tstart.month) +'_'+\
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
            # we plot the particle motion in 3D
            plot_particle_motion_3D(ax4, st.select(station = statlist[i]), Easting[i], Northing[i], depthstat[i])

            # we plot the conduit of Strokkur in 3D
            ax4.plot(Strokkurlon, Strokkurlat, Strokkurdep, "b--")

            # we format and label the axes
            ax4.yaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax4.xaxis.get_major_formatter().set_powerlimits([-2, 2])
            ax4.set_ylabel("Northing (m)")
            ax4.set_xlabel("Easting (m)")
            ax4.set_zlabel("Height (m)")

            # if(i==4):  ## -- saving --
            #    plt.savefig(localpath + 'Strokkur_particle_motion.png', format='png', dpi=300) #, transparent=True)
            #    plt.show()
            #    plt.close()

# we create the animation here. If you increase the frames, a longer batch of the
# seismic data will be processed as 1 frame equals a 1 second seismic window.
anim = animation.FuncAnimation(fig, animate, frames=10, interval=1000, blit=False)

# saving to m4 using ffmpeg writer
print("Saving video")
writervideo = animation.FFMpegWriter(fps=10)
anim.save(
    localpath + "Geyser_tremor_" + dimension + "...mp4", writer=writervideo
)
plt.close()
