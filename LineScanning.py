from PIL import Image
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import peakutils
import os
import os.path
import math
import pylab
import io
import sys
from tkinter.filedialog import askopenfilename


def Intensity(w, halfpoint, endsim, tc, tu, array = [], *args):	 # tc stands for time conversion and tu stands for time unit
	s = 25							     # space point row; this value can be manipulated
	plt.title("LineScan Data for 1 space point"); 
	plt.plot(array[s], 'r-');
	
	# plot general axis points
	plt.xticks([0, int(w/2), w], ['0 ms', "%d" %halfpoint + ' ' + tu, "%d" %endsim +' '+ tu]);					
	plt.xlabel('time');
	plt.ylabel('calcium intensity');
	plt.show();
	
def graphLinescan(img_name):
	# Load the line scan image
	lineScan = Image.open(img_name+".tif");

	# Now load the image as a matrix
	greyimg = lineScan.convert('L');
	matrix_img = np.asarray(greyimg); 
	
	# modify units for time and space points
	(numrows, numcols) = matrix_img.shape;   # calculate pixel length for width (x) and height (y) (numcols, numrows)
	print("numrows: ", numrows);
	print("numcols: ", numcols);
	
	if numcols > numrows:					 # default (horizontal image)
		x = numcols;
		y = numrows;
		a = 0;
	else:
		x = numrows;						# time is on the vertical axis because the image is flipped (vertical image)
		y = numcols;
		a = 1;
	
	print("x: ", x);
	print("y: ", y);
	
	# create array of time indices 
	time = np.arange(0, x, 1);		 #(start, end+1, stepsize) (total time = 2 ms per pixel)
			
	## Check if there is a corresponding .pty file that goes with the image
	exists = os.path.isfile(img_name+".pty");
	if (exists == True):	
		# access information from corresponding .pty file if one exists
		file = askopenfilename(filetypes=[("pty files", "*.pty")]); 		# open and read the .pty file
		ptyfile = io.open(file, 'r', encoding='utf-16');
		lines = ptyfile.readlines();
		
		# search for Width Convert Value
		for line in lines:
			if "WidthConvertValue" in line:
				value1 = line.split("=");		# obtain value after the = sign
				s1 = value1[1].strip();			# remove the ending new line character
				wcv = float(s1);					# cast the string to its numerical value
				break;

		# search for Time Per Pixel value
		for line in lines:		
			if "Time Per Pixel" in line:
				value2 = line.split("=");				# obtain value after the = sign
				s2 =  value2[1].strip();				# remove the ending new line character
				s2 = s2.replace('"', '');				# simplify string so that it can be type casted
				tpp = float(s2);						# cast the string to its numerical value
				break;
				
		# search for Unit for Time
		for i in range(len(lines)):
			line = lines[i];
			if "[Axis 4 Parameters]" in line:
				next_line = lines[i+1];
				if "AbsPositionUnitName" in next_line:
					value3 = next_line.split("=");				# obtain value after the = sign
					ap =  value3[1].strip();					# remove the ending new line character			
					ap = ap.replace('"', '');					# Abs Position (ap) or Time unit (tu)
					break;
					
		# Multiply space point factor		
		sp = wcv * y;					 # number of space points (width convert value) = .414 um per pixel
		sp = math.floor(sp);
		
		# calculate general axis points
		halfpoint = tpp*x/2
		endsim = tpp*x
	
	else:
		halfpoint = x/2					
		endsim = x
		tpp = 2.0	 					# time conversion factor
		ap = 'ms'						# time unit
		sp = 0.414*y;
		sp = math.floor(sp);
	
	# plot the intensity at the 50th space point with respect to time
	Intensity(x, halfpoint, endsim, tpp, ap, matrix_img);
	
	# plot the average/collapsed intensity at every point in space with respect to time 
	collapsed_intensity = np.sum(matrix_img, axis=a)/y; 	 #axis = 0 collapses rows vs.  axis = 1 collapses columns (axis = 0 collapses rows) : sum of all the rows divided by numrows	
	print("size of collapsed_intensity: ", len(collapsed_intensity));		
		
	plt.figure();
	plt.title('LineScan Data for all space points');
	plt.plot(collapsed_intensity); 
	
	# label general axis points
	plt.xticks([0, int(x/2), x], ['0 ms', "%d" %halfpoint + ' ' + ap, "%d" %endsim +' '+ ap]);	
	plt.xlabel('time');
	plt.ylabel('collapsed calcium intensity');
	plt.show();

	# detect areas and times where the linescan contains multiple sparks (not the highest intensity)	
	min_int = min(collapsed_intensity); 							#50?
	max_int = max(collapsed_intensity);
	
	print("min int: ", min_int);
	print("max int: ", max_int);
	print("done1")
	
	# first, find intensity peak (maximum) indices 
	ind = peakutils.indexes(collapsed_intensity, 0.6, 100);			# check for peak (maximum) points 
	peak_ind = time[ind];											# time indices (out of 10,000)
	print("Done detecting peaks");
	
	# find distance between each index to obtain WIDTH of each regular interval  
	diff_ind = peak_ind[1:peak_ind.size]-peak_ind[0:peak_ind.size-1];			#array holds distance between each peak index																		

	# check whether the midpoint intensity of each interval is greater than the minimal amount
	mid_indices = [];
	indices = [];
	tides = [];														# tide indices (0, number of tides -1)
	
## Use thresholding to detect calcium sparks
	# the higher the threshold value, the harder to detect sparks
	fact1 = 0.15;               										# 15% for LS1 img, 6% of the magnitude for LS2 img 
	min_threshold = min_int + fact1 * (max_int-min_int);				# must be above this threshold value to be a tide
	fact2 = 0.5															# 0.5 for LS1 img, 0.25
	max_threshold = max_int - (max_int-min_int)*fact2;
	print("minimum threshold: ", min_threshold);
	print("maximum threshold: ", max_threshold);
	print("done thresholding")
	for i in range(len(ind)-1):
		midpoint = (ind[i] + ind[i+1])/2							# array holds midpoint indices
		mid_indices.append(midpoint);								# array holds corresponding intensities
		mid_intensities = collapsed_intensity[mid_indices];
		# if the intensity at that point is greater than the threshold and the last peak has not yet been reached
		if (mid_intensities[i] > min_threshold) and (mid_intensities[i] < max_threshold):  # avoid accidentally detecting non-tides
			indices.append(mid_indices[i]);
			tides.append(mid_intensities[i]);						
	#print("[i]: ", indices[i]);
		
	# graph the intensities of the lineScan space points that detect calcium sparks (peaks should show areas where calcium sparks are present)
	plt.title('Calcium Spark Intensity Occurences');
	plt.plot(time, collapsed_intensity, 'r-');
	plt.xticks([0, int(x/2), x], ['0 ms', "%d" %halfpoint + ' ' + ap, "%d" %endsim +' '+ ap]);
	blue_dot = plt.plot(indices, tides, "bo", markersize = 10);				# place a blue dot at every tide 
	
	plt.xlabel('time');
	plt.ylabel('collapsed calcium intensity');
	plt.show();
	
	# Graph the 3D Surface Plot
	sp = np.arange(0, sp, 1);							# array of space points
	X = time; 			
	Y = sp;
	X, Y = np.meshgrid(X,Y);							# create a meshgrid
	Z = collapsed_intensity; 
	
	fig = plt.figure();
	ax = fig.gca(projection='3d');
	ax.plot_surface(X, Y, Z);
	plt.xticks([0, int(x/2), x], ['0 ms', "%d" %halfpoint + ' ' + ap, "%d" %endsim +' '+ ap]);
	plt.yticks([20, 60, 100, 140]);
	#plt.zticks([0, 60, 100, 140, 180]);
	
	ax.set_xlabel('time');
	ax.set_ylabel('space point');
	ax.set_zlabel('calcium intensity');
	
	plt.show();
	

def main():	
	graphLinescan("./Desktop/Frames/LineScans/LS1");
	#graphLinescan("./Desktop/Frames/LineScans/LS2");
	#graphLinescan("./Desktop/Frames/LineScans/ex1");
	#graphLinescan("./Desktop/Frames/LineScans/ex2");
	#graphLinescan("./Desktop/Frames/LineScans/wavesCR1"); 
main();