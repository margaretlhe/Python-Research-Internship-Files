#Program: Solves a system of ordinary differential equations
# Margaret He 
# Time Stamp: 10/29/15
 
import math
import numpy as np
import pylab as pyl
    
# some function z such that z(t+h) = z(t) + h*f(z,t)

def f(x, y, u, v, t, paramDict):				# calculates u'' and v'' in terms of dxdt and dydt 
	# Let x = u'
	dxdt = -(paramDict['k1'] + paramDict['k2'])*u + paramDict['k2']*v - \
		(paramDict['n1'] + paramDict['n2'])*x + paramDict['n2']*y - paramDict['F0']*t*(np.exp(-paramDict['g']*t));
	# Let y = v'
	dydt = paramDict['k2']*u - (paramDict['k2'] + paramDict['k3'])*v + \
		paramDict['n2']*x - (paramDict['n2'] + paramDict['n3'])*y;
	dudt = x;
	dvdt = y;
	return dxdt, dydt, dudt, dvdt;
	
	
	

def ForwardEuler(initCond, parameters):
	n = int(round(initCond['tph']/initCond['h']));                  # n = number of times you must find y(t) to find y(t+h)
	
	x = [0]*n;					   #allocate space for x array
	y = [0]*n;					   #allocate space for y array
	u = [0]*n;					   #allocate space for u array
	v = [0]*n;					   #allocate space for v array
	
	t = [0]*n;					   #allocate space for time array
	
	x[0] = initCond['x0'];
	y[0] = initCond['y0'];
	u[0] = initCond['u0'];
	v[0] = initCond['v0'];
	
	dt = initCond['h'];
	
	# Obtain values for x, y, and t
	for i in range(n-1):                 							 # repeat loops n times, i = index #
		f_x, f_y, f_u, f_v = f(x[i], y[i], u[i], v[i], t[i], parameters);	# calculate dxdt and dydt                  
		xnew = x[i]+ (dt*f_x);         							 # update each new value of x(t)
		
		#start at i+1 because initial conditions (i=0) should be set to 0 
		x[i+1] = xnew;              					     		 	
		y[i+1] = y[i] + (dt*f_y);								#append newest y value
		
		u[i+1] = u[i] + (dt*f_u);								#append newest u value	
		v[i+1] = v[i] + (dt*f_v);								#append newest v value
		
		tnew = t[i] + dt;						                 # input the latest time into the array
		t[i+1] = tnew;
		tph = tnew;
	return x, y, u, v, t;

# parameters
parameters = {'k1': 1.0, 'k2': 1.0, 'k3' : 1.0, 'n1' : 1.0, 'n2' : 1.0, 'n3' : 1.0, 'g' : 1.0, 'F0' : 1.0};

# initial conditions
initCond = {'h' : .001, 'tph' : 100.0, 'x0' : 0.0, 'y0' :  0.0, 'u0' : 0.0, 'v0' : 0.0};

# main 
if (initCond['tph'] >= 0):
	x_tph, y_tph, u_tph, v_tph, tph = ForwardEuler(initCond, parameters);   # calculate x and y values at time tph using Euler's method
	
	# Plot u and v
	pyl.plot(tph, u_tph, 'b-', label='numerical solution for u');
	pyl.plot(tph, v_tph, 'r-', label='numerical solution for v'); 
	pyl.legend(loc='upper right'); 
	pyl.show();
        
else:
    print("You entered an invalid time value.")     #time passed has to be positive
    


        
