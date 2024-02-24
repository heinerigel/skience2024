#!/usr/bin/env python
# coding: utf-8

# In[ ]:


AVOSEIS = os.getenv("AVOSEIS");
#sys.path.append('~/src/python_gt/AVOSEIS_PYTHON')
sys.path.append(AVOSEIS + "/bin"); # path to deposited modgiseis
import antelope.datascope as datascope
import numpy as np
import matplotlib as mpl
if 'DISPLAY' in os.environ.keys():
        mpl.use("Agg")
import matplotlib.pyplot as plt
import modgiseis
import time
import datetime
import getopt
from PIL import Image
######################################################

def usage():
        print 'Usage: '+sys.argv[0]+' [-fhv] <catalogpath> <dbplacespath> <outputdir> <pngfile> <number_of_weeks_to_plot> <weeksagofilter>'
        print """
	[-f] fast mode - bases percentiles on reporting period only, not full history
	[-h] help
	[-v] verbose mode
        <catalogpath> must have an origin and event table present
        <dbplacespath> is a list of volcanoes and their lat, lon, elev and radii in places_avo_1.3 schema format
	<outputdir> is the directory to save <pngfile> to
	<number_of_weeks_to_plot> is the number of weeks that will show on the plot
	<weeksagofilter> only volcanoes with an earthquake within this many weeks of the current date will be displayed
	"""
        print """\nExample: 
        produce a weekly summary the AVO catalog for all volcanoes with an earthquake in the past 1 week \n
        %s /Seis/Kiska4/picks/Total/Total volcanoes_avo /usr/local/mosaic/AVO/avoseis/counts weekly_report.png 13 1
        """ % (sys.argv[0])


def y2percentile(y, thesepercentiles):
	notfound = True
	index = -1
	p = 100
	while notfound and index<100:
		index += 1
		#print index, thesepercentiles[index], y, thesepercentiles[index]>y
		if thesepercentiles[index] >= y:
			p = index
			notfound = False
	return p

def print_pixels(fighandle, axhandle, number_of_weeks_to_plot, NUMVOLCANOES):
	dpi = fighandle.get_dpi()
	sizeInInches = fighandle.get_size_inches()
	print "Figure Inches: %f x %f" % (sizeInInches[0], sizeInInches[1])
	print "dpi: %d" % dpi
	print "Figure Pixels: %f x %f" % (sizeInInches[0] * dpi, sizeInInches[1] * dpi)
	posbbox = axhandle.get_position()
	posbbox = posbbox.get_points()
	print posbbox
	pos = np.empty(4)
	pos[0] = posbbox[0][0]
	pos[1] = posbbox[0][1]
	pos[2] = posbbox[1][0] - pos[0]
	pos[3] = posbbox[1][1] - pos[1]
	print pos
	print "Axes Position: %f to %f x %f to %f" % (posbbox[0][0], posbbox[1][0], posbbox[0][1], posbbox[1][1])
	axeswidthinpixels = pos[2]*dpi*sizeInInches[0]
	axesheightinpixels = pos[3]*dpi*sizeInInches[1]
	print 'Axes size in pixels: %f x %f' % (axeswidthinpixels, axesheightinpixels)
	gridwidthinpixels = axeswidthinpixels / (NUMVOLCANOES + 1)
	gridheightinpixels = axesheightinpixels / (number_of_weeks_to_plot + 1)
	print 'Grid size in pixels: %f x %f' % (gridwidthinpixels, gridheightinpixels)

def main(argv=None):
        try:
                opts, args = getopt.getopt(argv, 'fvh')
                if len(args)<6:
			print "only got %d command line arguments, expected 6" % (len(args))
			print argv
                        usage()
                        sys.exit(2)
        except getopt.GetoptError,e:
                print e
                usage()
                sys.exit(2)

	# Command line arguments
        catalogpath = args[0]
	if not os.path.exists(catalogpath):
		sys.exit("catalogpath does not exist")
        dbplacespath = args[1]
	if not os.path.exists(dbplacespath):
		sys.exit("dbplacespath does not exist")
        outdir = args[2]
	if not os.path.exists(outdir):
		sys.exit("outdir does not exist")
	pngfile = args[3]
	number_of_weeks_to_plot = int(args[4])
	weeksagofilter = int(args[5])
	if (number_of_weeks_to_plot < weeksagofilter):
		sys.exit("weeksagofilter should be <= number_of_weeks_to_plot")

	# Command line switches
        verbose = False
	fastmode = False
        for o, a in opts:
                if o == "-v":
                        verbose = True
                elif o in ("-h", "--help"):
                        usage()
                        sys.exit()
		elif o in ("-f"):
			fastmode = True 
			# fastmode will only base percentiles on the reporting period, not the full history
                else:
                        assert False, "unhandled option"

	# Will percentiles figures be plotted?
	bool_plot_percentiles_figure = False
	if (number_of_weeks_to_plot == weeksagofilter) and not (fastmode):
		bool_plot_percentiles_figure = True 
		print "Percentiles figure will be produced"

        if verbose:
                print "catalogpath = " + catalogpath
                print "dbplacespath = " + dbplacespath
                print "outdir = " + outdir
		print "pngfile = " + pngfile
                print "number_of_weeks_to_plot = %d" % number_of_weeks_to_plot
		print "weeksagofilter = %d" % weeksagofilter

	# time now
	datetimenow = datetime.datetime.now() # datetime as a local time
	datetimenowutc = datetime.datetime.utcnow() # this will add the 8 or 9 hours difference
	timenowstr = datetimenowutc.strftime('%Y/%m/%d %H:%M:%S')
	# These both return epoch as correct UTC time based on raven which has local time clock
	epochnow = time.mktime(datetimenow.timetuple()) # epoch
	epochnow2 = datascope.stock.now()
	datenumnow = mpl.dates.epoch2num(epochnow) # datenumber
	secsperday = 60 * 60 * 24
	epoch_startOfReportingPeriod = epochnow - (secsperday * 7 * weeksagofilter)
	dnum_startOfReportingPeriod = datenumnow - (7 * weeksagofilter)
	epoch1989 = 599616000
	if verbose:
		print "timenowstr = " + timenowstr
		print "epoch_startOfReportingPeriod = %f " % epoch_startOfReportingPeriod

	# Load the list of volcano data
	dictplaces = modgiseis.readplacesdb(dbplacespath)
	place = dictplaces['place']
	lat = dictplaces['lat']
	lon = dictplaces['lon']
	radius = dictplaces['radius']
	n = place.__len__()
	if verbose:
		print "number of places = {}".format(n)
	
	# Initialize variables
	VOLCANO = list()
	BIN_EDGES = list()
	COUNTS = list()
	CUMML = list()

	# Because the event database may be very large and we may be opening it twice per volcano,
	# we will create a temporary subset database with just the last number_of_weeks_to_plot weeks
	# This will be used to see if the volcano will appear in the final plot
	# (In fast mode, it will also be used to compute percentiles - but they will be a bit meaningless)
	fastcatalogpath = "/tmp/weeklysummary_fastdb";
	modgiseis.dbsubset2db(catalogpath, "time >= %f" % epoch_startOfReportingPeriod, fastcatalogpath)
	if os.path.exists(fastcatalogpath):
		print "%s created successfully" % (fastcatalogpath)
	else:
		print "%s not created: this is usually because there were no earthquakes in the past %d weeks in %s" % (fastcatalogpath, weeksagofilter, catalogpath)
		print "%s was last updated at %s" % (catalogpath, time.ctime(os.path.getmtime(catalogpath)))
		sys.exit()
	if fastmode:
		catalogpath = fastcatalogpath
		print "catalogpath now = %s" % catalogpath
		# percentiles will be based on reporting period only, not full history
	
	if n > 0:
		print "- number of places = {}".format(n)
	
		for c in range(n): # for each volcano in the list
			nfilter = 0
				
			print "\nPROCESSING %s" % place[c]

			# how many earthquakes have there been at this volcano in the reporting period?
			subset_expr = "deg2km(distance(lat, lon, %s, %s))<%f" % (lat[c], lon[c], float(radius[c]))
			dictorigin, nfilter = modgiseis.dbgetorigins(fastcatalogpath, subset_expr)
	
			# load all time history and bin them 
			if (nfilter > 0):
				if fastmode:
					subset_expr = "time > %f && deg2km(distance(lat, lon, %s, %s))<%f" % (epoch1989, lat[c], lon[c], float(radius[c]))
				else:
					subset_expr = "time > %f && deg2km(distance(lat, lon, %s, %s))<%f" % (epoch1989, lat[c], lon[c], float(radius[c]))
				#print "'%s'" % subset_expr
	                	dictorigin, n = modgiseis.dbgetorigins(catalogpath, subset_expr)
				print "- number of events in all-time = {}".format(n)
				if n>0:
					timearray = dictorigin['time']
					time_firstevent = timearray[0] # assuming they are sorted
					bin_edges, snum, enum = modgiseis.compute_bins(dictorigin, np.min([time_firstevent, dnum_startOfReportingPeriod]), datenumnow, 7.0) # function name is a misnomer - we are computing bin_edges

					# now we get our array of counts per week
					counts = modgiseis.bin_counts(timearray, bin_edges)
	
					# cumulative magnitude
		        		energy = modgiseis.ml2energy(dictorigin['ml'])
		        		binned_energy = modgiseis.bin_irregular(timearray, energy, bin_edges)
					binned_ml = modgiseis.energy2ml(binned_energy)
	
					# summarise
					if verbose:	
						print 'firstevent: %s' % modgiseis.datenum2datestr(time_firstevent)
						print 'lastevent: %s' % modgiseis.datenum2datestr(timearray[-1])
						print "bin edges (length=%d): %s to %s " % (len(bin_edges), modgiseis.datenum2datestr(bin_edges[0]), modgiseis.datenum2datestr(bin_edges[-1]))
						print "counts length=%d " % len(counts)
						print "binned_ml length=%d " % len(binned_ml)
					if len(bin_edges)<(number_of_weeks_to_plot+1):
						sys.exit("bin_edges < number_of_weeks_to_plot+1")
	
					# append to lists to save for plotting	
					VOLCANO.append(place[c])
					BIN_EDGES.append(bin_edges)
					COUNTS.append(counts)
					CUMML.append(binned_ml)
				else:
					print "WARNING: Got some events from the fastcatalog, but not the total catalog"
			else:
				print "- 0 events in %s for %s" % (fastcatalogpath, place[c]) 	
	
	NUMVOLCANOES = len(VOLCANO)

	# Compute percentile distributions
	print "\nComputing percentiles"
	y = np.empty(NUMVOLCANOES)
	PERCENTILES = list()
	percentages = range(101)
	for i in range(NUMVOLCANOES):
		thesecounts = COUNTS[i]
		thesepercentiles = np.empty(101)
		thesepercentiles = np.percentile(thesecounts, percentages)
		PERCENTILES.append(thesepercentiles)

	if verbose:
		print "VOLCANO is a list of strings (length=%d):" % NUMVOLCANOES
		print VOLCANO
		print "BIN_EDGES is a list of numpy arrays (length=%d):" % len(BIN_EDGES)
		print "COUNTS is a list of numpy arrays (length=%d):" % len(COUNTS)
		print "CUMML is a list of numpy arrays (length=%d):" % len(CUMML)
		print "PERCENTILES is a list of numpy arrays (length=%d):" % len(PERCENTILES)

	# Plot & save percentiles figure
	if bool_plot_percentiles_figure:
		print "\nCreating percentiles figure"
		dimension = int(np.ceil(np.sqrt(NUMVOLCANOES)))
		figp = plt.figure()
		figpax = [None] * NUMVOLCANOES
		for i in range(NUMVOLCANOES):
			thesecounts = COUNTS[i]
			thesepercentiles = PERCENTILES[i]
			figpax[i] = figp.add_subplot(dimension, dimension, i)
			figpax[i].plot(range(0,100+1,1),thesepercentiles)
			figpax[i].set_title(VOLCANO[i], fontsize=8)
			figpax[i].set_xticks([0, 50, 100])
			figpax[i].set_yticks([0, max(thesecounts)])
			#figpax[i].set_xticklabels(['0','50','100'])
			plt.setp( figpax[i].get_xticklabels(), fontsize=6 )
			plt.setp( figpax[i].get_yticklabels(), fontsize=6 )
		figp.suptitle('Counts vs. percentile')
		figp.savefig("%s/countspercentiles.png" % (outdir), dpi=200)
		figp.clf()

	# Page size, dots-per-inch and scale settings
	print "\nSetting page size, dpi and scale"
	fig2 = plt.figure()
	if number_of_weeks_to_plot > NUMVOLCANOES:
		dpi = 6*(number_of_weeks_to_plot+1)
	else:
		dpi = 6*(NUMVOLCANOES+1)
	fig2.set_dpi(dpi)
	fig2.set_size_inches((10.0,10.0),forward=True)
	if number_of_weeks_to_plot > NUMVOLCANOES:
		axes_width = 0.8 * (NUMVOLCANOES + 1) / (number_of_weeks_to_plot + 1)
		axes_height = 0.8
	else:
		axes_width = 0.8
		axes_height = 0.8 * (number_of_weeks_to_plot + 1) / (NUMVOLCANOES + 1)
	fig2ax1 = fig2.add_axes([0.1, 0.85-axes_height, axes_width, axes_height])
	print_pixels(fig2, fig2ax1, number_of_weeks_to_plot, NUMVOLCANOES)

	# Colormap configuration	
	print "Configuring color map"
	import matplotlib.cm as cmx
	import matplotlib.colors as colors
	#colormapname = 'RdYlGr_r'
	colormapname = 'hot_r'
	MAXMAGCOLORBAR = 5.0
	MINMAGCOLORBAR = 1.0
	mycolormap = cm = plt.get_cmap(colormapname)
	cNorm = colors.Normalize(vmin=MINMAGCOLORBAR, vmax=MAXMAGCOLORBAR)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=mycolormap)
	#print scalarMap.get_clim()

	# Marker size configuration
	print "Configuring marker sizes"
	volcanolabels = list()
	SCALEFACTOR = 84.0 / dpi
	#MAXMARKERSIZE = 41.0 * SCALEFACTOR
	MAXMARKERSIZE = 43.0 * SCALEFACTOR
	MINMARKERSIZE = 4.0 * SCALEFACTOR
	PTHRESHOLD = 50

	# set up weekending list for yticklabels
	weekending = list()
	print ">SCAFFOLD"
	print number_of_weeks_to_plot
	print "<SCAFFOLD"
	for week in np.arange(-number_of_weeks_to_plot-1, 0, 1): # SCAFFOLD: GETTING ERROR HERE
		print ">SCAFFOLD"
		print week
		print bin_edges
		print "<SCAFFOLD"
		dstr = modgiseis.datenum2datestr(bin_edges[week]) # bin_edges is still set to BIN_EDGES for last element i
		weekending.append(dstr[5:10])
		print week, dstr

	weekindex = np.arange(-number_of_weeks_to_plot, 0, 1)
	for i in range(NUMVOLCANOES):
		print "\nPlotting data for %s" % VOLCANO[i]
		thesecounts = COUNTS[i]
		thesecumml = CUMML[i]
		volcanolabels.append("%s(%d)" % (VOLCANO[i], thesecounts[-1]))
		for w in weekindex:
			y = thesecounts[w]
			magnitude = thesecumml[w]
			p = y2percentile(y,PERCENTILES[i])
			if verbose:
				print w, weekending[w], y, p
			if y>0:
				colorVal = scalarMap.to_rgba(magnitude)
				msize = MINMARKERSIZE + (p-PTHRESHOLD) * (MAXMARKERSIZE - MINMARKERSIZE) / (100-PTHRESHOLD)
				if msize<MINMARKERSIZE:
					msize=MINMARKERSIZE
				fig2ax1.plot(i+0.5, w, 's', color=colorVal, markersize=msize, linewidth=0 );
				if msize > MAXMARKERSIZE * 0.3:
					fig2ax1.text(i+0.5, w, "%d" % y, horizontalalignment='center', verticalalignment='center', fontsize = 8 * SCALEFACTOR)

	print "\nAdding xticks, yticks, labels, grid"
	fig2ax1.set_axisbelow(True) # I think this puts grid and tickmarks below actual data plotted

	# x-axis
	fig2ax1.set_xticks(np.arange(.5,NUMVOLCANOES+.5,1))
	fig2ax1.set_xlim([-0.5, NUMVOLCANOES+0.5])
	fig2ax1.xaxis.grid(True, linestyle='-', color='gray')
	fig2ax1.set_xticklabels(volcanolabels)
	fig2ax1.xaxis.set_ticks_position('top')
	fig2ax1.xaxis.set_label_position('top')
	plt.setp( fig2ax1.get_xticklabels(), rotation=45, horizontalalignment='left', fontsize=10*SCALEFACTOR )

	# y-axis
	fig2ax1.set_yticks(np.arange(-number_of_weeks_to_plot-0.5, 0, 1))
	fig2ax1.set_yticklabels(weekending)
	fig2ax1.set_ylim([-number_of_weeks_to_plot - 0.5, -0.5])
	plt.setp( fig2ax1.get_yticklabels(), fontsize=10*SCALEFACTOR )

	#print "Adding watermark"
	#fig2ax1.text(NUMVOLCANOES/2, -number_of_weeks_to_plot/2, 'PROTOTYPE', fontsize=75*SCALEFACTOR, color='gray', ha='center', va='center', rotation=60, alpha=0.5)

	print "Saving figure"
	outfile = "%s/%s" % (outdir, pngfile)
	fig2.savefig(outfile, dpi=dpi)

	print "Removing whitespace"
	im = Image.open(outfile)
	im = modgiseis.trim(im)
	im.save(outfile) 
	
	# Legend
	print "\nPlotting legend"
	fig3 = plt.figure()
	fig3.set_dpi(dpi)
	fig3.set_size_inches((10.0,10.0),forward=True)
	fig3ax2 = fig3.add_axes([0.1, 0.85-axes_height/2, axes_width/20, axes_height/2])
	a = np.linspace(0, 1, 256).reshape(-1,1)
	fig3ax2.imshow(a, aspect='auto', cmap=plt.get_cmap(colormapname), origin='lower')
	fig3ax2.set_xticks([])
	fig3ax2.set_yticks(np.arange(0,256+1,256/(2*(MAXMAGCOLORBAR-MINMAGCOLORBAR)))) # the +1 is to label the top of the range since arange stops at 256-51.2 otherwise
	ytl = np.arange(MINMAGCOLORBAR, MAXMAGCOLORBAR+0.1, 0.5)
	ytl_list = list()
	for index in range(len(ytl)):
		ytl_list.append("%.1f" % ytl[index])
	fig3ax2.set_yticklabels(ytl_list)
	fig3ax2.set_ylabel('Cumulative\nMagnitude')

	print "Saving legend"
	colorbarfile = "%s/colorbar.png" % (outdir)
	fig3.savefig(colorbarfile, dpi=dpi)

	print "Removing whitespace from legend"
	im = Image.open(colorbarfile)
	im = modgiseis.trim(im)
	im.save(colorbarfile) 

	# Clean up	
	print "Cleaning up"
	modgiseis.remove_database(fastcatalogpath, False)

	print "Done.\n"

### END OF FUNCTIONS ###

# Program entry
if __name__ == "__main__":
        if len(sys.argv) > 1:
                main(sys.argv[1:])
        else:
                usage()

