import antelope.datascope as datascope
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image, ImageChops
import glob
#################################
# SUBROUTINES 
#################################

def trim(im):
    # trim whitespace from the image file
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def yn_choice(message, default='y'):
    choices = 'Y/n' if default.lower() in ('y', 'yes') else 'y/N'
    choice = raw_input("%s (%s) " % (message, choices))
    values = ('y', 'yes', '') if default == 'y' else ('y', 'yes')
    return True if choice.strip().lower() in values else False

def remove_database(dbpath, ask=True):
	if os.path.exists("%s" % dbpath):
		doit = True
		if ask:
			doit = yn_choice("%s database exists. Remove?" % dbpath)
		if doit:
			os.remove(dbpath)
			for dbfile in glob.glob("%s.*" % dbpath):
				os.remove(dbfile)
			print "database %s removed" % dbpath
		else:
			print "You have chosen not to remove %s" % dbpath
	else:
		print "database %s not found" % dbpath

def dbsubset2db(dbpath, subset_expr, subsetdbpath):
	# open the origin table, join to event table, subset for preferred origins
	db = datascope.dbopen( dbpath, 'r')
	dborigin = db.lookup( table = 'origin' )
	dborigin = dborigin.join('event')
	dbnetmag = db.lookup( table = 'netmag' )
	dborigin = dborigin.subset("orid == prefor")
	print "subset %s with %s" % (dbpath, subset_expr)
	dborigin = dborigin.subset(subset_expr)
	dborigin = dborigin.sort('time')
	n = dborigin.nrecs()
	if n>0:	
		# ask to remove database if it already exists
		remove_database(subsetdbpath, False)
		print "unjoin to %s" % subsetdbpath
		dborigin.unjoin(subsetdbpath)
	else:
		print "no records to save to new database\n"

	db.free()

def dbgetorigins(dbpath, subset_expr):
	# open the origin table, join to event table, subset for preferred origins
	db = datascope.dbopen( dbpath, 'r')
	dborigin = db.lookup( table = 'origin' )
	dborigin = dborigin.join('event')
	dbnetmag = db.lookup( table = 'netmag' )
	dborigin = dborigin.subset("orid == prefor")

	# apply the optional subset expression if there is one, order by time, and display number of events.
	if subset_expr:
		dborigin = dborigin.subset(subset_expr)
	dborigin = dborigin.sort('time')
	n = dborigin.nrecs()
	#print "- number of events = {}".format(n)

	# if size of arrays already known, preallocation much faster than recreating each time with append
	dictorigin = dict()
	origin_id = np.empty(n)
	origin_ml = np.empty(n)
	origin_epoch = np.empty(n)

	# load origins from database and store them in a dictionary
	for dborigin[3] in range(n):
		(origin_id[dborigin[3]], origin_ml[dborigin[3]], origin_epoch[dborigin[3]]) = dborigin.getv('orid','ml','time')
		if origin_ml[dborigin[3]] < -1.0:
			db2 = dbnetmag.subset("orid == %d" % origin_id[dborigin[3]])
			maxmag = -1.0
			n_netmag = db2.nrecs()
			if n_netmag > 0:
				for db2[3] in range(n_netmag):
					(magtype, magnitude) = db2.getv('magtype', 'magnitude')
					if magnitude>maxmag:
						maxmag = magnitude
			origin_ml[dborigin[3]] = maxmag
			
	dictorigin['id'] = origin_id
	dictorigin['ml'] = origin_ml
	dictorigin['time'] = mpl.dates.epoch2num(origin_epoch)

	# close the database and free the memory. 
	# It seems that db.close and db.free both close the database, and closing twice produces error
	db.free()

	return dictorigin, n

def plot_time_ml(ax, dictorigin, x_locator, x_formatter, snum, enum):
	time = dictorigin['time']
	ml = dictorigin['ml']
	## we do not want to plot Ml = -999.0, the Antelope value for a non-existent Ml - filter them out
	#i = np.where( ml > -3.0)
	#time = time[i]
	#ml = ml[i]

	# plot the data
	ax.plot_date(time, ml, linestyle='o', markerfacecolor='None')
	ax.grid(True)
	ax.xaxis_date()
	plt.setp( ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=7 )
	ax.set_ylabel('Ml', fontsize=8)
	ax.xaxis.set_major_locator(x_locator)
	ax.xaxis.set_major_formatter(x_formatter)
	if snum and enum:
        	ax.set_xlim(snum, enum)
	return

def bin_counts(time, bin_edges_in):
	# count the number of "events" in each bin 
	# use this if want to produce counts (per bin) but not actually plot them!
        counts, bin_edges_out = np.histogram(time, bin_edges_in)
	return counts 

def plot_counts(ax, dictorigin, x_locator, x_formatter, bin_edges_in, snum, enum):
	# compute all data needed
        time = dictorigin['time']
        cumcounts = np.arange(1,np.alen(time)+1)
	if len(bin_edges_in) < 2:
		return
	binsize = bin_edges_in[1]-bin_edges_in[0]
	binsize_str = binsizelabel(binsize)

  	# plot 
	counts, bin_edges_out, patches = ax.hist(time, bin_edges_in, cumulative=False, histtype='bar', color='black', edgecolor=None)
        ax.grid(True)
        ax.xaxis_date()
        plt.setp( ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=7 )
        ax.set_ylabel("# Earthquakes\n%s" % binsize_str, fontsize=8)
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
	if snum and enum:
        	ax.set_xlim(snum, enum)

        ax2 = ax.twinx()
        p2, = ax2.plot(time,cumcounts,'g', lw=2.5)
        ax2.yaxis.get_label().set_color(p2.get_color())
	ytl_obj = plt.getp(ax2, 'yticklabels')  # get the properties for yticklabels
	#plt.getp(ytl_obj)                       # print out a list of properties
	plt.setp(ytl_obj, color="g")            # set the color of yticks to red
	plt.setp(plt.getp(ax2, 'yticklabels'), color='g') #xticklabels: same
        ax2.set_ylabel("Cumulative\n# Earthquakes", fontsize=8)
        ax2.xaxis.set_major_locator(x_locator)
        ax2.xaxis.set_major_formatter(x_formatter)
	if snum and enum:
        	ax2.set_xlim(snum, enum)
        return

def bin_irregular(time, y, bin_edges):
	# bin y against time according to bin_edges (not for binning counts, since they don't have a y value)

        # bin the data as for counts
        counts_per_bin, bin_edges_out = np.histogram(time, bin_edges)
        i_start = 0
        i_end = -1
        binned_y = np.empty(np.alen(counts_per_bin))

        for binnum in range(np.alen(counts_per_bin)):
                i_end += counts_per_bin[binnum]
                if i_start <= i_end:
                        binned_y[binnum] = np.sum(y[i_start:i_end+1])
                else:
                        binned_y[binnum] = 0
                i_start = i_end + 1
        return binned_y

def ml2energy(ml):
	energy = np.power(10, 1.5 * ml)
	return energy

def energy2ml(energy):
	ml = np.log10(energy)/1.5
	return ml

def plot_energy(ax, dictorigin, x_locator, x_formatter, bin_edges, snum, enum):

	# compute all data needed
        time = dictorigin['time']
	energy = ml2energy(dictorigin['ml'])
        cumenergy = np.cumsum(energy)
	binned_energy = bin_irregular(time, energy, bin_edges)
	if len(bin_edges) < 2:
		return
        barwidth = bin_edges[1:] - bin_edges[0:-1]
	binsize = bin_edges[1]-bin_edges[0]
	binsize_str = binsizelabel(binsize)

	# plot
        ax.bar(bin_edges[:-1], binned_energy, width=barwidth, color='black', edgecolor=None)

        # re-label the y-axis in terms of equivalent Ml rather than energy
        yticklocs1 = ax.get_yticks()
        ytickvalues1 = np.log10(yticklocs1) / 1.5
        yticklabels1 = list()
        for count in range(len(ytickvalues1)):
                yticklabels1.append("%.2f" % ytickvalues1[count])
        ax.set_yticks(yticklocs1)
        ax.set_yticklabels(yticklabels1)

        ax.grid(True)
        ax.xaxis_date()
        plt.setp( ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=7 )
        ax.set_ylabel("Energy %s\n(unit: Ml)" % binsize_str, fontsize=8)
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
	if snum and enum:
        	ax.set_xlim(snum, enum)

	# Now add the cumulative energy plot - again with yticklabels as magnitudes
        ax2 = ax.twinx()
        p2, = ax2.plot(time,cumenergy,'g',lw=2.5)

        # use the same ytick locations as for the left-hand axis, but label them in terms of equivalent cumulative magnitude
        yticklocs1 = ax.get_yticks()
        yticklocs2 = (yticklocs1 / max(ax.get_ylim())) * max(ax2.get_ylim() )
        ytickvalues2 = np.log10(yticklocs2) / 1.5
       	yticklabels2 = list()
        for count in range(len(ytickvalues2)):
                yticklabels2.append("%.2f" % ytickvalues2[count])
        ax2.set_yticks(yticklocs2)
        ax2.set_yticklabels(yticklabels2)

        ax2.yaxis.get_label().set_color(p2.get_color())
	ytl_obj = plt.getp(ax2, 'yticklabels')  # get the properties for yticklabels
	#plt.getp(ytl_obj)                       # print out a list of properties
	plt.setp(ytl_obj, color="g")            # set the color of yticks to red
	plt.setp(plt.getp(ax2, 'yticklabels'), color='g') #xticklabels: same
        ax2.set_ylabel("Cumulative Energy\n(unit: Ml)",fontsize=8)
        ax2.xaxis.set_major_locator(x_locator)
        ax2.xaxis.set_major_formatter(x_formatter)
	if snum and enum:
        	ax2.set_xlim(snum, enum)
        return

def floor(myvector, binsize):
        # rather than floor to the next lowest integer (i.e. multiple of 1), floor to the next lowest multiple of binsize
        return np.floor(myvector / binsize) * binsize

def ceil(myvector, binsize):
        # rather than ceil to the next highest integer (i.e. multiple of 1), ceil to the next highest multiple of binsize
        return np.ceil(myvector / binsize) * binsize

def binsizelabel(binsize):
	binsize_str = "" 
        if binsize == 1.0/1440:
		binsize_str = "per minute"
        elif binsize == 1.0/24:
		binsize_str = "per hour"
        elif binsize == 1.0: 
		binsize_str = "per day"
        elif binsize == 7.0: 
		binsize_str = "per week"
        elif binsize >= 28 and binsize <=31: 
		binsize_str = "per month"
        elif binsize >= 365 and binsize <= 366:
		binsize_str = "per year"
	return binsize_str

def autobinsize(daysdiff):
        # Try and keep to around 100 bins or less
        if daysdiff <= 2.0/24:  # less than 2 hours of data, use a binsize of 1 minute
                binsize = 1.0/1440
        elif daysdiff <= 4.0:  # less than 4 days of data, use a binsize of 1 hour
                binsize = 1.0/24
        elif daysdiff <= 100.0:  # less than 100 days of data, use a binsize of 1 day
                binsize = 1.0
        elif daysdiff <= 700.0: # less than 700 days of data, use a binsize of 1 week
                binsize = 7.0
        elif daysdiff <= 365.26 * 23: # less than 23 years of data, use a binsize of (approx) 1 month
                binsize = 365.26/12
        else:
                binsize = 365.26 # otherwise use a binsize of 1 year

        return binsize

def compute_bins(dictorigin, snum=None, enum=None, binsize=None):
	# If snum and enum are provided, enum will be end of last bin UNLESS you ask for binsize=365.26, or 365.26/12
	# in which case it will be end of year or month boundary
	# If snum and enum not given, they will end at next boundary - and weeks end on Sat midnight/Sunday 00:00

        # First lets calculate the difference in time between the first and last events
	if (snum==None):
        	snum = np.min(dictorigin['time']) # time of first event
        	enum = np.max(dictorigin['time']) # time of last event
        daysdiff = enum - snum

	if (binsize==None):
		binsize = autobinsize(daysdiff)

	# special cases
        if binsize == 365.26/12:
                # because a month isn't exactly 365.26/12 days, this is not going to be the month boundary
                # so let us get the year and the month for snum, but throw away the day, hour, minute, second etc
                sdate = mpl.dates.num2date(snum)
                sdate = datetime.datetime(sdate.year, sdate.month, 1, 0, 0, 0)
                thisyear = sdate.year
                thismonth = sdate.month
                snum = mpl.dates.date2num(sdate)
                bins = list()
                bins.append(snum)
                count = 0
                while bins[count] < enum + binsize:
                        count += 1
                        thismonth += 1
                        if thismonth > 12: # datetime.datetime dies if sdate.month > 12
                                thisyear += 1
                                thismonth -= 12
                        monthdate = datetime.datetime(thisyear, thismonth, 1, 0, 0, 0)
                        bins.append(mpl.dates.date2num(monthdate))
                bins = np.array(bins)
                enum = np.max(bins)

	elif binsize == 365.26: # binsize of 1 year
                # because a year isn't exactly 365.26 days, this is not going to be the year boundary
                # so let us get the year for snum, but throw away the month, day, hour, minute, second etc
                sdate = mpl.dates.num2date(snum)
                sdate = datetime.datetime(sdate.year, 1, 1, 0, 0, 0)
                snum = mpl.dates.date2num(sdate)
                bins = list()
                bins.append(snum)
                count = 0
                while bins[count] < enum + binsize:
                        count += 1
                        yeardate = datetime.datetime(sdate.year + count, 1, 1, 0, 0, 0)
                        bins.append(mpl.dates.date2num(yeardate))
                bins = np.array(bins)
                enum = np.max(bins)

	else: # the usual case
        	# roundoff the start and end times based on the binsize
		if snum==None and enum==None:
			print "snum and enum undefined - calculating"
        		snum = floor(snum, binsize) # start time
        		enum = ceil(enum, binsize) # end time
        	#bins = np.arange(snum, enum+binsize, binsize)
        	bins = np.arange(enum, snum-binsize, -binsize)
		bins = bins[::-1]

	print 'snum: %s' % datenum2datestr(snum)
	print 'enum: %s' % datenum2datestr(enum)
        return bins, snum, enum

def readplacesdb(dbplacespath):
	###########################################
	# should be able to do this with code like:
	#
	#db = datascope.dbopen( dbpath, 'r')
	#db = db.lookup( table = 'places' )
	#db = db.sort('lon')
	#n = db.nrecs()
	#if n > 0:
        #print "- number of places = {}".format(n)
        #for db[3] in range(n):
        #        (placename, placetype, placelat, placelon, placeelev) = db.getv('place','placetype','lat','lon','elev')
	#
	# but this does not work. Neither does the Perl equivalent. So although dbe opens the places database, there might
	# be a problem with the database. So just treat the table as a file and parse it instead.
	#
	dbtablepath = dbplacespath + ".places"
	placename = list()
	placetype = list()
	placelat = list()
	placelon = list()
	placeelev = list()
	placeradius = list()
	if os.path.exists(dbtablepath):
		f = open(dbtablepath)
		lines = f.readlines()
		f.close()
		for line in lines:
			elements = line.split()
			placename.append(elements[4])
			placetype.append(elements[5])
			placelat.append(elements[0])
			placelon.append(elements[1])
			placeelev.append(elements[2])
			placeradius.append(elements[3])
	else:
		print dbtablepath + " does not exist"		
	return {'place':placename, 'placetype':placetype, 'lat':placelat, 'lon':placelon, 'elev':placeelev, 'radius':placeradius}

def read_volcanoes(): # THIS IS A PREVIOUS VERSION I WROTE FOR GETTING VOLCANOES FROM PLACESDB THAT SEEMS TO WORK
        dbplacespath = 'places/volcanoes'
        dbhandle = datascope.dbopen( dbplacespath, 'r')
        dbptr = dbhandle.lookup( table = 'places' )
        n = dbptr.nrecs()
        dictplaces = dict()
        for dbptr[3] in range(n):
                thisrecord = {'place': "%s" % dbptr.getv('place'), 'lat': "%s" % dbptr.getv('lat'), 'lon': "%s" % dbptr.getv('lon') }
                dictplaces[dbptr[3]] =  thisrecord
        dbhandle.free()
        dbhandle.close()
        return dictplaces

def datenum2datetime(dnum):
	floordnum = int(np.floor(dnum)) # still a float, fromordinal needs an int
        dt = datetime.datetime.fromordinal(floordnum) + datetime.timedelta(days=dnum%1)
	return dt

def datetime2datestr(dt):
        dstr = dt.strftime('%Y/%m/%d %H:%M:%S')
	return dstr

def datenum2datestr(dnum):
	dt = datenum2datetime_alt(dnum)
	dstr = datetime2datestr(dt)
	return dstr

def datenum2datetime_alt(dnum):
	dt = mpl.dates.num2date(dnum)
	return dt

def datetime2datenum_alt(dt):
	dnum = mpl.dates.date2num(dt)
	return dnum