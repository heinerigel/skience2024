#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from obspy import read_inventory, UTCDateTime
import cartopy.crs as ccrs
import csv

from math import asin, atan2, cos, degrees, radians, sin

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    """
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    a = radians(bearing)
    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
    lon2 = lon1 + atan2(
        sin(a) * sin(d/R) * cos(lat1),
        cos(d/R) - sin(lat1) * sin(lat2)
    )
    return (degrees(lat2), degrees(lon2))


if __name__ == '__main__':
    # read in topo data (on a regular lat/lon grid)
    # (srtm data from: http://srtm.csi.cgiar.org/)
    srtm = np.loadtxt("./output_SRTMGL1.asc", skiprows=6)
    srtm = np.flipud(np.asarray(srtm))

    min_lon = 110.41
    max_lon = 110.48
    min_lat = -7.58
    max_lat = -7.51

    # origin of data grid as stated in srtm data file header
    # create arrays with all lon/lat values from min to max and
    lats = np.linspace(-7.686824838501053,-7.410920503923094,srtm.shape[0])
    lons = np.linspace(110.28845429420473, 110.64332127571106,srtm.shape[1])

    # create Basemap instance with Mercator projection
    # we want a slightly smaller region than covered by our srtm data
    proj = ccrs.TransverseMercator(
        central_latitude=np.mean((min_lat, max_lat)),
        central_longitude=np.mean((min_lon, max_lon)))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection=proj)
    ax.set_extent((110.41,110.48, -7.58, -7.51), crs=proj)

    # create grids and compute map projection coordinates for lon/lat grid
    x, y = np.meshgrid(lons, lats)

    # Make contour plot
    cs = ax.contour(x, y, srtm, 40, colors="k", alpha=0.3, transform=proj)
    # https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.mpl.geoaxes.GeoAxes.html#cartopy.mpl.geoaxes.GeoAxes.gridlines
    ax.gridlines(draw_labels=True, color='k', linestyle='--')

    stations_f = []
    t1 = UTCDateTime("2001-01-01")
    inv = read_inventory("../stationxml/merapi_stationxml.xml")
    for stat in inv[0]:
        stat_id = "XM.%s" % stat.get_contents()['channels'][0]
        coo = inv.get_coordinates(stat_id)
        ll = [stat.code, coo['longitude'], coo['latitude'], coo['elevation']]
        stations_f.append(ll)



    for name, lon, lat, elev in stations_f:
        ax.plot([lon], [lat], "rv", markersize=5, transform=proj)
        try:
            path = "../6C-steer/6C-2001-10-26 04:59:58.350000_XM.%s_flinn.csv"%name
            baz = []
            mcorr = []
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    baz.append(float(row[1]))
                    mcorr.append(float(row[4]))

            baz = np.asarray(baz)
            mcorr = np.asarray(mcorr)
            for i in range(len(baz)):
                if np.abs(mcorr[i]) > 0.6:
                    y,x = get_point_at_distance(lat, lon, 0.1, baz[i])
                    dy = (y - lat)*10000
                    dx = (x - lon)*10000
                    ax.quiver(lon,lat,dx,dy,angles="xy",scale=0.1,color="green",alpha=np.abs(mcorr[i])/10) #,transform=proj)
        except:
            continue

    fig.savefig("merapi.png", dpi=300)
    plt.show()
