from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

data = np.loadtxt("./Desktop/Frames/GoodRecording/boom.data")
diff = data[1:data.size-1]-data[0:data.size-2]
slope = diff/(1/16)    # change in time is 1/16 because there are 16 pixel frames per sec

slopeslope = (slope[1:slope.size-1]-slope[0:slope.size-2])/(1/16);
mytimes  = np.arange(0, data.size*1/16,1/16);
#plt.plot(mytimes[0:mytimes.size-2], slop, 'b-');
#plt.plot(slope, 'b-');
#plt.show();

newfiltData = [];
index = 0;
alpha = 0.0001;
for s in slopeslope:
	if (s < alpha) and (s > -alpha):
		#print(index);
		#newdata = np.delete(data, index);
		newfiltData.append(data[index+1]);
	index = index + 1;	
plt.plot(newfiltData, 'r-');
#plt.plot(data, 'g-');
plt.show();