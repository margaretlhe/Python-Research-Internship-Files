#Flubberlength.py 
#Program calculates length of object through movement
#Margaret He
#Last Updated: 02/19/16

from PIL import Image
import numpy
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import os
from scipy.linalg import solve
		
		
# calculate x = length of flubber (in inches) using thresholding 
def FlubberLength(img_name):
	srcimg = Image.open(img_name);
	#srcimg.show();  #output source image

	# Crop Ruler Out (cut out left 600 pixels)
	left = 600;
	top = 0;
	width = 1100;
	height = 720;
	box = (left, top, left+width, top+height); 
	croppedimg = srcimg.crop(box);
	#croppedimg.show()  #output cropped image

	# Grey scale Conversion
	greyimg = croppedimg.convert('L');
	#greyimg.show(); #output grey image  

	# Now load the image as a matrix
	matrix_img = numpy.asarray(greyimg); 
	#print("The image displayed as a matrix:\n", matrix_img); #output matrix of the image

	# Threshold the image (5: darkest, 255: brightest)
	threshold = 220;

	# Matrix of Zeroes and NonZeroes (0: dark, 1: light/not dark)
	img_array = deepcopy(matrix_img);  # make a copy of the matrix img to edit 
	low_values_indices = img_array < threshold; # find which indices of the matrix contain low threshold values
	img_array[low_values_indices] = 0;   #Set all low values to 0
	#print("\nThe matrix of the image after thresholding:\n", img_array);
	#print(img_array[0, :]);	
		
	#Create a new image using the thresholded values
	thresh_img = Image.fromarray(img_array, 'L');
	#thresh_img.show(); #output new thresholded image

	# Calculate the flubber's pixel amount using the image's nonzero matrix values
	def pixel_Length(array = [], *args):
		# 720 rows, 1100 columns
		(numrows, numcols) = array.shape;
		
		#initial values
		pixLengths = [];
		numpixels = 0;
		
		# search the matrix columns	
		for col in range(numcols):
			for rowval in array[:, col]:
				if (rowval > 0):			# while a bright spot is detected
					numpixels += 1;
				else: #rowval == 0 
					if (numpixels > 0): # if there is a bright spot detected in the middle of the column
						pixLengths.append(numpixels);
						numpixels = 0;		# reset numpixels to 0 before checking next col
			#print("pixels for col ", col, ": ", numpixels);
			if(numpixels > 0 ):
				pixLengths.append(numpixels);
				numpixels = 0;		# reset numpixels to 0 before checking next col
					
		return max(pixLengths);	
						
	#Calculate the blob's number of pixels			
	numpixels = pixel_Length(img_array);

	# Calculate the length and height of the blob/flubber
	# 12 inch ruler = 647 pixels in length, so 1 inch = 53.92 pixels
	numinches = numpixels/53.92; 
	# print("\nThe pixel length of the flubber at this time frame is", pLength,"pixels or ",numinches, "inches.\n");
	
	return numinches;
	#return numpixels;
	
# Load image onto Python (change digits of png file to check for a different time frame)
#image files range from 001-158.png and 1000-1581.png)

def gatherData(arr1 = [], arr2 = [], *kwarg, timeout = 3600.0):
	for i in range(1, 4+1):   		# i goes to 2575
		fname  = "images%.3d.png" %i; 
		img = "C:/Users/Margaret/Desktop/Frames/GoodRecording/" + fname;
		fLength = FlubberLength(img);
		arr1.append(i);
		arr2.append(fLength);
		print("Processed image ",i,"\n");			#Progress bar
		

# calculate x' = change in flubber length within 2 different time frames
def graphROC(filtdata = [], filtslope = [], *kwarg):
	data = 0.0254*numpy.loadtxt("./Desktop/Frames/GoodRecording/boom.data");		# data includes all flubber lengths y
	#	inital data value at t = initial 
	data = (data - data[0])/data[0];
	mytimes  = numpy.arange(0, data.size*1/16,1/16);
	
	# plot unfiltered data
	plt.figure();
	plt.plot(data, 'r-');
	plt.title("Unfiltered Data");
	#plt.plot(mytimes[0:mytimes.size-2], data, 'r-');
	
	# find slope/rate of change of flubber length
	slope = [];
	diff = data[1:data.size-1]-data[0:data.size-2];							#calculate change in flubber length (or delta y)
	slope = diff/(1.0/16.0); 
	plt.plot(slope);
	plt.title("Original Slope");
	print(slope);
	
	# image filtering (filter out noise in data and slope)
	index = 0;
	alpha = 0.01;		  						# max noise range
	
	for s in slope:						# include all points inside this range in plot
		if (s < alpha) and (s > -alpha):
			filtdata.append(data[index+1]);	# using index + 1 because the second derivative accounts for the difference between y(index + 1) and (index)
			filtslope.append(s);
		index = index + 1;	
	#filtslope = [];
	#filtdata = numpy.asarray(filtdata); 
	#diff = filtdata[1:filtdata.size-1]-filtdata[0:filtdata.size-2];							#calculate change in flubber length (or delta y)
	#filtslope = diff/(1/16); 		
	# plot filtered data
	plt.figure();
	plt.plot(filtdata, 'r-');
	plt.title("Filtered Data");
	#plt.plot(mytimes[0:mytimes.size-2], filtdata, 'r-');
	script_dirname = os.path.dirname(os.path.realpath(__file__));
	plt.savefig(script_dirname+"./"+"fig_filtered_data.png", format='png', dpi=600);
	plt.show();
	
	# move slope to a slope array	
	#filtslope.append(slope[index]);	
	
	# plot filtered slope
	plt.figure();
	plt.plot(mytimes[0:mytimes.size-2], slope, 'b-');
	plt.title("Filtered Slope");
	script_dirname = os.path.dirname(os.path.realpath(__file__));
	plt.savefig(script_dirname+"./"+"fig_filtered_roc.png", format='png', dpi=600);
	plt.show();
	
	#plot filtered rate of rate of change
	#diff2 = (slopearr[1:slopearr.len()-1]-slopearr[0:slopearr.size-2])/(1/16);	#calculate second derivative
	#slopeofslope = diff2/(1/16);
	#plt.figure();
	#plt.plot(mytimes[0:mytimes.size-2], slopeofslope, 'g-');
	#script_dirname = os.path.dirname(os.path.realpath(__file__));
	#plt.savefig(script_dirname+"./"+"fig_filtered_sos.png", format='png', dpi=600);
	#plt.show();
	
	
def setMatrices(x = [], xslope = [], A = [], F = [], *kwarg):
		# Matrix A:       [ x(t1) x'(t1)]
		#				  [  ...	... ]
		#				  [ x(tn) x'(tn)]
		
		(numrows, numcols) = A.shape;		# 2575 rows by 2 columns 
		#data = numpy.loadtxt("./Desktop/Frames/GoodRecording/boom.data");	
		
		# input x (length) values into A matrix, column 0
		rowc0 = 0;
		while rowc0 < numrows: 	 # c starts at 0 so it stops at nrows-1
			if rowc0 <= numrows-1:
				A[rowc0, 0] = x[rowc0];
				rowc0 = rowc0 + 1;
			else:
				break;
	
		# input x' (change in length) values into A matrix, column 1
		rowc1 = 0;
		while rowc1 < numrows:
			if rowc1 <= numrows-1:
				A[rowc1, 1] = xslope[rowc1];
				rowc1 = rowc1 + 1;
			else:
				break;
				
		row = 0;
		#input force of binder clip clamp into F matrix
		force = .00281*9.8;								# mass = .00281 kilograms, gravitational acceleration: 9.8 m/s 
		rad = 0.0254*(1/8);
		Area = numpy.pi*(rad*rad);
		force = force/Area; 
		while row < numrows: 	 						# c starts at 0 so it stops at nrows-1
			if row <= numrows-1:
				if row < 1155:							# force is exerted when binder clip is attached to flubber
					F[row, 0] = force;
					row = row + 1;
				else:									# binder clip is removed from flubber
					F[row, 0] = 0;
					row = row + 1;
			else:
				break;
				
		
def main(): 
	time = [];
	length = [];
	#gatherData(time, length);						#no longer need to call this function because we stored all the flubber lengths in data

	# Plot flubber length
	#print("Length array: ", length);
	#plt.figure();
	#plt.plot(time, length, 'b-', label = 'Flubber Length', linewidth='2');
	#plt.xlabel('Image # (frame #)');				# 16 frames per second
	#plt.ylabel('Length of flubber (inches)');		
	#plt.title('Flubber Length');
	#plt.legend(loc='upper right', shadow=True, fontsize='x-large');
	#script_dirname = os.path.dirname(os.path.realpath(__file__));
	#plt.savefig(script_dirname+"./"+"fig_flubberlength.png", format='png', dpi=600);
	#plt.show();

	newfiltData = [];											# x
	newfiltSlope = [];											# x'	
		
	#Plot change in flubber length between two time frames
	graphROC(newfiltData, newfiltSlope);	
	
	#Set up Matrices 
	matrixA = numpy.zeros(shape = [len(newfiltData), 2]);		# 1813 x 2 matrix that holds x and x' values
	matrixF = numpy.zeros(shape = [len(matrixA), 1]);												# holds force (m*g) of (paper clip clamp) exerts on flubber at each time frame
	#matrixF = [];
	setMatrices(newfiltData, newfiltSlope, matrixA, matrixF);
	print(newfiltSlope);
	
	#Solve Voigt Model Equation to solve for (matrix X) elastic (k) and viscous (eta) parameters	
	k, eta = numpy.linalg.lstsq(matrixA, matrixF)[0];			# *SOLUTION HOLDERS: holds elastic (k) and viscous (eta) constants
		
	print("matrix A (holding data and slope): ");	
	print(matrixA);
	
	print("\nmatrixF (holding force of binder clip): ");
	print(matrixF);
	
	
	print("\n(Constants) Solutions to VME:\n"); 
	print("k = ",k, "\n");
	print("eta = ",eta, "\n"); 	
	#numpy.savetxt('FlubberLength.out', (k, eta), delimiter=',') 
main()