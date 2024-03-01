#!/usr/bin/env python
import numpy as np
from numpy import array
from matplotlib.pyplot import *
from obspy.clients.fdsn import Client
from obspy import *
from mpl_toolkits.basemap import Basemap
import gzip


if __name__ == '__main__':
    # read in topo data (on a regular lat/lon grid)
    # (srtm data from: http://srtm.csi.cgiar.org/)
    srtm = np.loadtxt("./output_SRTMGL1.asc", skiprows=6)
    srtm = np.flipud(np.asarray(srtm))

    # origin of data grid as stated in srtm data file header
    # create arrays with all lon/lat values from min to max and
    lats = np.linspace(-7.686824838501053,-7.410920503923094,srtm.shape[0])
    lons = np.linspace(110.28845429420473, 110.64332127571106,srtm.shape[1])
    #lats = np.linspace(47.85, 47.67, srtm.shape[0])
    #lons = np.linspace(12.65, 13.0000, srtm.shape[1])

    # create Basemap instance with Mercator projection
    # we want a slightly smaller region than covered by our srtm data
    fig = figure()
    axXY = fig.add_subplot(1,1,1)
    #axXY = fig.add_axes()
    #axMS = fig.add_axes([0.1,0.1,0.2,0.2])
    m = Basemap(projection='merc', resolution="h", llcrnrlon=110.41, llcrnrlat=-7.58, \
            urcrnrlon=110.48, urcrnrlat=-7.51,ax=axXY)
    #m = Basemap(projection='merc', resolution="h", llcrnrlon=110.28845429420473, llcrnrlat=-7.686824838501053, \
    #        urcrnrlon=110.64332127571106, urcrnrlat=-7.410920503923094,ax=axXY)
    #m = Basemap(projection='merc', lon_0=13, lat_0=48, resolution="h",
    #            llcrnrlon=12.65, llcrnrlat=47.67, urcrnrlon=13., urcrnrlat=47.85,ax=axXY)

    # create grids and compute map projection coordinates for lon/lat grid
    x, y = m(*np.meshgrid(lons,lats))

    # Make contour plot
    cs = m.contour(x, y, srtm, 40, colors="k", alpha=0.3)
    #m.drawcountries(color="red", linewidth=1)

    # Draw a lon/lat grid (20 lines for an interval of one degree)
    m.drawparallels(np.linspace(-7.4, -7.7, 31), labels=[1,1,0,0], fmt="%.2f", dashes=[2,2])
    m.drawmeridians(np.linspace(110.2, 110.7, 31), labels=[0,0,1,1], fmt="%.2f", dashes=[2,2])

    stations_f = []
    t1 = UTCDateTime("2001-01-01")
    inv = read_inventory("../stationxml/*.xml")
    for i,stat in enumerate(inv[0]):
        stat_id = "XM.%s"%stat.get_contents()['channels'][0]
        coo = inv.get_coordinates(stat_id)
        ll = [stat.code,coo['longitude'],coo['latitude'],coo['elevation']]
        stations_f.append(ll)
        

#    x,y = m(data_1[0],data_1[1])
#    sca = m.scatter(x, y, data_1[3]*100,cmap=cm.viridis, c=data_1[4], alpha=0.6)
#    axXZ.scatter(x, data_1[2], data_1[3]*100, cmap=cm.viridis,c=data_1[4], alpha=0.6)
#    axZY.scatter(data_1[2], y, data_1[3]*100, cmap=cm.viridis,c=data_1[4], alpha=0.6)
#    axXZ.set_ylim(-8000,2000)
#    axZY.set_xlim(-8000,2000)

#    topo_x = srtm[100,:]
#    topo_y = srtm[:,200]
#    grid_x, tmp = m(lons, np.arange(len(lons)))
#    tmp, grid_y = m(np.arange(len(lats)), lats)
#    axXZ.plot(grid_x, topo_x ,color='black')
#    axZY.plot(topo_y , grid_y,color='black')


    for station in stations_f:
        x,y = m(station[1],station[2])
        m.plot([x], [y], "rv", markersize=5)
    #axXZ.plot([station[1]], [station[3]/1000.], "bv", markersize=7)
    #axZY.plot([station[3]/1000.], [station[2]], "bv", markersize=7)
    #axXY.text(station[1], station[2], station[0], ha="left", va="top")
    #axXZ.text(station[1], station[3]/1000., station[0], ha="left", va="top")
    #axZY.text(station[3]/1000., station[2], station[0], ha="left", va="top")

    #cb = fig.colorbar(sca, cax=axCB, format="%i", ticks=range(2002,2021))
    #leg1 = axXY.legend([sca],["0.2 1.0 2.0 3.0"],scatterpoints=4,markerscale=1,loc='upper left')
    #axXZ.set_xlabel("Longitude")
    #axZY.set_ylabel("Latitude")
    #axXZ.set_ylabel("Depth")
    #axMS.scatter([data_xx[0]],[data_xx[1]],s=[data_xx[2]*1000],marker="o")
    fig.savefig("merapi.png",dpi=300)
    show()
