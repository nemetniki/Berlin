#!/usr/bin/python3.4
import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.integrate import quad
from mpl_toolkits.mplot3d.axes3d import Axes3D

#mpl.rcParams['mathtext.fontset'] = 'cm'
#mpl.rcParams['mathtext.rm'] = 'serif'
#mpl.rc('font',family='FreeSerif')
#mpl.rc('xtick',labelsize=25)
#mpl.rc('ytick',labelsize=25)

now    = time.time()

c      = 0.3
gd     = 0.7
oma    = 10 #in GHz
kappav = np.array([0.0001,0.1,1])  #in GHz

def ex(t):
	return np.exp(-kappa*t)
def Ca(t):
	return np.cos(oma*t)
def Sa(t):
	return np.sin(oma*t)

def Nd2k(k,t):
	cof = np.abs(gd/(kappa+1j*oma))**2*(g02/c/k)**2
	den = 1/np.abs(-1j*c*k+B)**2
	Ck  = np.cos(c*k*t)
	Sk  = np.sin(c*k*t)
	A2  = (c*k)**2 * (2*Ca(t)-ex(t)) * ex(t)
	B2  = (kappa**2+oma**2) * (2*Ck-1)
	AB  = oma*(Ca(t)+Ck-(Ca(t)*Ck+Sk*Sa(t)))*ex(t) + \
kappa*(Sa(t)-Sk-(Sa(t)*Ck-Sk*Ca(t)))*ex(t)
	return cof * ( 1 - den*(A2 + B2 - 2*c*k*AB) )
	
k = np.linspace(-3,35,1000)
t = np.linspace(0,50,2001)*2*np.pi
k,t = np.meshgrid(k,t)


fig = plt.figure(figsize=(15,18))
for i in range(0,kappav.size):
	ax    = fig.add_subplot(kappav.size,1,i+1,projection='3d')
	kappa = kappav[i]
	g02 = kappa*2*c/np.pi
	B     = 1j*oma + kappa
	surf  = ax.plot_surface(k,t/2/np.pi,Nd2k(k,t),cmap=mpl.cm.jet)
plt.show()
#plt.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric3_T=0.png")


