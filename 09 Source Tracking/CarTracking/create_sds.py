from obspy import *
import os

st = read("waveform.mseed")
for tr in st:
    myday = str(tr.stats.starttime.julday)

    pathyear = str(tr.stats.starttime.year)
    # open catalog file in read and write mode in case we are continuing d/l,
    # so we can append to the file
    mydatapath = os.path.join("../data_sds", pathyear)

    # create datapath 
    if not os.path.exists(mydatapath):
        os.mkdir(mydatapath)

    mydatapath = os.path.join(mydatapath, tr.stats.network)
    if not os.path.exists(mydatapath):
        os.mkdir(mydatapath)

    mydatapath = os.path.join(mydatapath, tr.stats.station)

    # create datapath 
    if not os.path.exists(mydatapath):
                os.mkdir(mydatapath)

    mydatapathchannel = os.path.join(mydatapath,tr.stats.channel + ".D")

    if not os.path.exists(mydatapathchannel):
        os.mkdir(mydatapathchannel)

    netFile = tr.stats.network + "." + tr.stats.station +  "." + tr.stats.location + "." + tr.stats.channel+ ".D." + pathyear + "." + myday
    netFileout = os.path.join(mydatapathchannel, netFile)

    # try to open File
    try:
        netFileout = open(netFileout, 'ab')
    except:
        netFileout = open(netFileout, 'w')
        # header of the stream object which contains the output of the ADR
    tr.write(netFileout , format='MSEED')
    netFileout.close()
