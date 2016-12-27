# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:18:55 2016

"""
import pylab
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import ode
import sys


def KV_ode(y,t, params):
    n = 1; 
    m=2.64*10**(-3);#weight in kg
    F=m*9.8; A=1; rad = 0.0254*(1/8);
    A = np.pi*(rad*rad); 
    dydt = np.zeros((n,1));
    
    tau = params['tau'];
    eta = params['eta'];
    E   = params['E'];
    dydt = (1.0/eta)*((F/A)*(t<tau)-E*y); 

    return dydt
    
def KV_numeric(t, L0, E, eta, tau):
    params = {'tau':tau, 'E':E, 'eta':eta};
        
    solver = ode(KV_ode).set_integrator('dop853');#'vode', method='bdf', order=4, 
        #nsteps=10, rtol=1e-10,atol=1e-10);
    #final time of the integration
    tfin = t;
    #initial time  
    t0 = 0.0;
    #step size h
    h = 1; 
    #number of steps to take from t0 to tfin
    nsteps = np.floor((tfin-t0)/h)+1;
    #this is the actual times
    #time = np.linspace(0, tfin, nsteps);    
    yinit = np.array([0.0]);
    k=1;
    st = np.zeros((int(nsteps), 1));
    strain = np.zeros((int(nsteps),1));
    solver.set_initial_value(yinit, t0).set_f_params(params);
    while solver.successful() and k < nsteps:
        solver.integrate(solver.t + h);
        #store the calculated solution at these times
        st[k] = solver.t;
        
        strain[k] = solver.y[0];
        k += 1; 
    
    return float(L0*(1+strain[-1]));

def KV_plot(t, L0, E, eta, tau):
    params = {'tau':tau, 'E':E, 'eta':eta};
        
    solver = ode(KV_ode).set_integrator('vode', method='bdf', order=6, 
        nsteps=10000, rtol=1e-6,atol=1e-6);
    #final time of the integration
    tfin = t;
    #initial time  
    t0 = 0.0;
    #step size h
    h = 0.1; 
    #number of steps to take from t0 to tfin
    nsteps = np.floor((tfin-t0)/h)+1;
    #this is the actual times
    #time = np.linspace(0, tfin, nsteps);   
    yinit = np.array([0]);
    k=1;
    st = np.zeros((int(nsteps), 1));
    strain = np.zeros((int(nsteps),1));
    solver.set_initial_value(yinit, t0).set_f_params(params);
    while solver.successful() and k < nsteps:
        solver.integrate(solver.t + h);
        #store the calculated solution at these times
        st[k] = solver.t;
        strain[k] = solver.y[0];
        k += 1; 
    plt.plot(st, L0*(strain+1)); 
    #plt.plot(st, strain); 
    
def KV_numfit(t, L0, E, eta, tau):
    #print(t)
    n = len(t);
    print(n)
    rets = [];
    for i in range(0, n):
        rets.append(KV_numeric(t[i], L0, E, eta, tau));
        
    return rets;
    
def func(x, a, b, c):
    return a*np.exp(-b*x)+c;
    
def stress(t, tau):
    m=2.64*10**(-3);#weight in kg
    F=m*9.8; A=1; rad = 0.0254*(1/8);
    A = np.pi*(rad*rad);    
    return (F/A)*((t>0))*((t<tau))/1000; 
    
def voigtModel(t, L0, E, eta,tau):
    m=2.64*10**(-3);#weight in kg
    F=m*9.8; A=1; rad = 0.0254*(1/8);
    tau = 70.0; 
    A = np.pi*(rad*rad);
    ret = (F/A)*((1.0)/E)*((1-np.exp(-E*t/eta))-(1-np.exp(-E*(t-tau)/eta))*(t>tau));
    #ret = (F/A)*(1/E)*(-np.exp(-E*t/eta)+1-(t>tau));     
    return L0*(1+ret);
    
#fname = "E:/ExtDriveETC/Downloads/Photos/GoodRecording/boom_filtB.data";
#boom_filtB.data
fname = "F:/Box Sync/Box Sync/ResearchLogETC/mentoring/MargaretDavis/imaging/boom_filtB.data"; 
data = 0.0254*pylab.loadtxt(fname);
times = np.arange(0,(data.size)*(1/16),1/16);

popt, pcov = curve_fit(voigtModel,times,data, p0=(10,10,10,60));

#KV_plot(100, 2, 889.22, 3880, 70)
#sys.exit();

#popt, pcov = curve_fit(KV_numfit,times,data, p0=(10,10,10,70));

fig, ax1 = plt.subplots(); 
ax2 = ax1.twinx(); 
ax1.plot(times,data, 'r-', label='Filtered Experimental Data');
newtimes = np.arange(0,5000*(1/16), 1/16);
ax1.plot(newtimes, voigtModel(newtimes, popt[0], popt[1], popt[2], popt[3]), 'b-', label='Voigt Model');
ax1.set_xlabel('time (s)'); ax1.set_ylabel('Length (m)');
ax2.plot(times, stress(times, popt[3]), 'k--', label='fitted app stress');
ax2.set_ylabel('Applied stress in KPa');
ax1.legend(loc='upper right', shadow=True, fontsize='x-large');
ax2.legend(loc='lower right', shadow=True, fontsize='x-large');

plt.show();

#pylab.savefig("E:/ExtDriveETC/Downloads/Photos/GoodRecording/"+"fitFunc.png", format="png", dpi=600);
#xdata = np.linspace(0, 4, 50);
#y = func(xdata, 2.5, 1.3, 0.5);
#ydata = y+0.2*np.random.normal(size=len(xdata));

#popt, pcov = curve_fit(func, xdata, ydata)
#E = 889.22 Pa
#eta = 3880.3 Pa.s or 3.88 kPa.s