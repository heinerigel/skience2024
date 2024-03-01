from obspy.clients.fdsn import Client
from obspy import *
import os
cl = Client("http://tarzan")
t1 = UTCDateTime("2001-11-08")
t2 = UTCDateTime("2001-11-08T23:50")
st = cl.get_waveforms(network="XM",station="*",location="*",channel="*",starttime=t1,endtime=t2)
#st.merge(method=1,fill_value="latest")
myday = str(st[0].stats.starttime.julday)

pathyear = str(st[0].stats.starttime.year)
    # open catalog file in read and write mode in case we are continuing d/l,
    # so we can append to the file
for i,tr in enumerate(st):
    mydatapath = os.path.join("./data_sds/", pathyear)
    # create datapath 
    if not os.path.exists(mydatapath):
        os.mkdir(mydatapath)

    mydatapath = os.path.join(mydatapath, st[i].stats.network)

    if not os.path.exists(mydatapath):
        os.mkdir(mydatapath)


    mydatapath = os.path.join(mydatapath, st[i].stats.station)
    # create datapath 
    if not os.path.exists(mydatapath):
       os.mkdir(mydatapath)

    mydatapathchannel = os.path.join(mydatapath,st[i].stats.channel + ".D")

    if not os.path.exists(mydatapathchannel):
        os.mkdir(mydatapathchannel)

    netFile = st[i].stats.network + "." + st[i].stats.station +  "." + st[i].stats.location + "." + st[i].stats.channel+ ".D." + pathyear + "." + myday
    netFileout = os.path.join(mydatapathchannel, netFile)

        # try to open File
    try:
        netFileout = open(netFileout, 'ab')
    except:
        netFileout = open(netFileout, 'w')

    # header of the stream object which contains the output of the ADR
    st[i].write(netFileout , format='MSEED')
    netFileout.close()
