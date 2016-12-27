# Forward Euler Program: Solves ordinary differential equations
# Margaret He // Program 1
# Time Stamp: 10/17/15
 
import math
import numpy as np
import pylab as pyl
    
# y(t+h) = total amount of bacteria present (after a specified amount of time (sec) passes)
# y(t+h) = y(t) + h*f(y,t)


def getUserInput():

    # First ask user to input h, the change in time in sec
    h = eval(input("Enter the unit of time change you would like to input into the bacterial growth equation.\nh = "))
                          
    # Ask user what value of t+h they want to use to calculate y(t+h) or amount of bacteria after t+h time
    tph = eval(input("Enter the amount of time you want to calculate the bacteria growth for.\nt+h= "))
   
    return h, tph



# global arrays
y = []                                  # y[i] is the solution at time t[i]: y(t[i])
t = []                                  



def f(y, t):                              # dy/dt or f(y(t))
    dydt = (y**2) + (2*y)
    return dydt

def realSol(t):
    # return (- 2.0*np.exp(2.0*t)/(np.exp(2.0*t)-3.0));
	return(np.exp(-20*t));
	
vecRealSol = np.vectorize(realSol);
    
def ForwardEuler(dt, y0, T):
    # f = f(y,t)
    # dt = h
    # y0 = y(0) or initial y
    # T = tph   

    y.append(y0)                       #input y(0) to the array first
    t.append(0)

    n = int(round(T/dt))               # n = number of times you must find y(t) to find y(t+h)
    for i in range(n):                 # repeat loops n times, i = index #
        f_y = f(y[i], t[i])            # calculate f(y,t)                  
        ynew = y[i]+ (dt*f_y)          # update each new value of y(t)
        y.append(ynew)                 # input the latest y(t) value into the array
        y_tph = ynew
        tnew = t[-1] + dt              # input the latest time into the array
        t.append(tnew)
        tph = tnew
    return y, t


# main
#h, tph = getUserInput()            # h and(t+h) are input in
h = 0.1; 
tph = .5; 
if (tph == 0):            # Base Case     
    print("Total amount of bacteria present after 0 seconds: 1")

elif (tph > 0):
    y_tph, tph = ForwardEuler(h, 1, tph)   #Calculate y(tph) using Euler's method
    print("Total amount of bacteria present after", tph, "seconds: ",y_tph)
    pyl.plot(tph, y_tph, 'b-', label='ODE numerical solution');
    pyl.plot(tph, vecRealSol(tph), 'r-', label='real solution');
    pyl.legend(loc='upper right'); 
    pyl.show();
        
else:
    print("You entered an invalid time value.")     #time passed has to be positive
    


        
