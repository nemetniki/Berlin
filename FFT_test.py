#!/usr/bin/python3.4
import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.integrate import quad

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rc('font',family='FreeSerif')
mpl.rc('xtick',labelsize=25)
mpl.rc('ytick',labelsize=25)

####################
### TIMER STARTS ###
####################
now = time.time()

###############################
### TIME-DEPENDENT FUNCTION ###
###############################
def rho_d(kappa, oma, gam):

	return np.exp(-(t)**2)

##################	
### PARAMETERS ###
##################
D      = 0.7
oma    = 10 #in GHz
kappav = np.array([0.0001,0.1,1])  #in GHz
gam    = 0.001 #in GHz
hbar   = 6.62607004 
kb     = 1.38064852
T      = 0.00001
nd     = 1./(np.exp(hbar*oma/kb/T)-1)
endt   = 1000.
Nt     = 2**20
t      = np.linspace(0,endt,Nt)
endk   = 5000.
Nk     = 10000
ome    = 0.
k      = np.linspace(-endk,endk,Nk) + ome*100.
print(k)
therm  = hbar/(kb*T)

################
### PLOTTING ###
################
colors={'red':(241/255.,88/255.,84/255.),\
        'orange':(250/255,164/255.,58/255.),\
        'pink':(241/255,124/255.,176/255.),\
        'brown':(178/255,145/255.,47/255.),\
        'purple':(178/255,118/255.,178/255.),\
        'green':(96/255,189/255.,104/255.),\
        'blue':(93/255,165/255.,218/255.),\
        'yellow':(222/255., 207/255., 63/255),\
        'black':(0.,0.,0.)}
collab = ['green','orange','purple']
linew  = [3,2,2,4]
linest = ['-','--','-.',':']

fig,ax = plt.subplots(1,2,figsize=(25,8))

#for i in range(0,kappav.size):
#for i in range(kappav.size-1,-1,-1):

kappa=0
#	print("kappa is: ",kappav[i])
#	sys.stdout.flush()	

	#################################################
	### EVALUATION OF THE TIME-DEPENDENT SOLUTION ###
	#################################################
rho=rho_d(kappa, oma, gam)
norm = np.abs(np.sum(rho*2*endt/(Nt)))**2
print(norm)
evol = rho/np.sqrt(norm)

	#########################
	### FOURIER TRANSFORM ###
	#########################
fourr = np.fft.fft(evol)
four = np.fft.fftshift(fourr)*2*endt/(Nt-1)/np.sqrt(norm)
freqr = np.fft.fftfreq(Nt,(endt)/(Nt-1))
freq = np.fft.fftshift(freqr)

	############
	### PLOT ###
	############
ax[0].plot(freq*2*np.pi,four.real,color=colors[collab[0]],ls=linest[0],lw=linew[0])
ax[0].plot(t,evol,color=colors[collab[0]],ls=linest[0],lw=linew[0])
ax[1].plot(freq*2*np.pi,four.imag,color=colors[collab[0]],ls=linest[0],lw=linew[0])
ax[0].grid(True)
ax[0].legend([0.0001,0.1,1],fontsize=20)
ax[0].set_xlabel('Frequency',fontsize=30)
ax[0].set_ylabel('Refractive index',fontsize=30)
#ax[0].set_ylim(10**(-8),1)
ax[0].set_xlim(-10,10)
ax[0].set_ylim(0.,1.)
ax[1].grid(True)
ax[1].legend([0.0001,0.1,1],fontsize=20)
ax[1].set_xlabel('Frequency',fontsize=30)
ax[1].set_ylabel('Absorption',fontsize=30)
#ax[1].set_ylim(10**(-8),1)
ax[1].set_xlim(-10,10)


##################
### TIMER ENDS ###
##################
end=time.time()
h = int((end-now)/3600.)
m = int((end-now)/60.-h*60)
s = int((end-now)-h*3600-m*60)
print('%02d:%02d:%02d' %(h,m,s))
plt.show()


