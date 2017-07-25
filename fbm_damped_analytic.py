#!/usr/bin/python3.4
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rc('font',family='FreeSerif')
mpl.rc('xtick',labelsize=25)
mpl.rc('ytick',labelsize=25)

now = time.time()

##############################################
### FUNCTION FOR TIME-DEPENDENT ABSORPTION ###
##############################################
def rho_d(D, gamma, oma, kappa, nd, t):
	t    = t*2*np.pi
	B    = 1j*oma + kappa
	dBa2 = 1/(oma**2 + kappa**2)
	return np.exp(- D**2 * dBa2 * ( nd*(1+np.exp(-2*kappa*t)) + .5*(3-np.exp(-2*kappa*t)) + 1j*np.sin(oma*t)*np.exp(-kappa*t) - np.exp(-kappa*t)*np.cos(oma*t)*(1+2*nd) - 1j*oma/2/kappa*(1-np.exp(-2*kappa*t)) + 2*kappa*t + 2*kappa*dBa2*(-2*kappa+ B*np.exp(-np.conjugate(B)*t) + np.conjugate(B)*np.exp(-B*t))))* np.exp(-gamma*t)*.5
#	return np.exp(-D**2*dBa2*(.5*(1-1j*oma/kappa)*(1-np.exp(-2*kappa*t))+1j*np.exp(-kappa*t)*np.sin(oma*t)-np.exp(-kappa*t)*np.cos(oma*t)-dBa2*(oma**2-3*kappa**2+2*kappa*(B*np.exp(-np.conjugate(B)*t)+np.conjugate(B)*np.exp(-B*t)))))*.5*np.exp(-gamma*t)
	#return np.exp(-D**2/oma**2*((2*nd+1)*(1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t))*.5*np.exp(-gamma*t)

##################
### PARAMETERS ###
##################
t     = np.linspace(0,4000,300001)
D     = 0.7
oma   = 10 #in GHz
kappav = np.array([0.0001,0.1,1])  #in GHz
gamma = 0.001 #in GHz
hbar  = 6.62607004 #???* 10**(-2)
kb    = 1.38064852
T     = 3000.00001
nd    = 1/(np.exp(hbar*oma/kb/T)-1)
print(nd)

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

for i in range(0,kappav.size):
	kappa=kappav[i]
	print(kappav[i])	
	plt.figure(1,figsize=(13,8))
	evol=rho_d(D,gamma,oma, kappa,nd,t)
	plt.plot(t,evol.real,t,evol.imag)
	plt.grid(True)
	fourr = np.fft.fft(evol)
	four = np.fft.fftshift(fourr)
	freqr = np.fft.fftfreq(t.size,t[1]-t[0])
	freq = np.fft.fftshift(freqr)
	plt.figure(2,figsize=(13,8))
	plt.grid(True)
#plt.semilogy(freq,four.real,ls='None',marker='x',markersize=2)#,freq,four.imag)
	#print(collab[i])
	plt.semilogy(freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[0])#,freq,four.imag)
plt.legend([0.0001,0.1,1],fontsize=20)
plt.xlabel('Frequency',fontsize=30)
plt.ylabel('$\Re[P(\omega)]$',fontsize=30)
plt.xlim(-35,35)
plt.ylim(10**(-2),10**4)

### TIMER ENDS ###
end=time.time()
h = int((end-now)/3600.)
m = int((end-now)/60.-h*60)
s = int((end-now)-h*3600-m*60)
print('%02d:%02d:%02d' %(h,m,s))
plt.show()
#plt.savefig("./plot.png")


