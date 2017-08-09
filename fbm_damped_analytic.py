#!/usr/bin/python3.4
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

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

#	return (1 + D**2*dBa2*( 1+np.exp(-2*kappa*t)-np.exp(-kappa*t)*2*np.cos(oma*t) )) * \
#	np.exp(- D**2 * dBa2 * ( .5*(3-np.exp(-2*kappa*t)) + 1j*np.sin(oma*t)*np.exp(-kappa*t) - \
#	np.exp(-kappa*t)*np.cos(oma*t) - 1j*oma/2/kappa*(1-np.exp(-2*kappa*t)) + 2*kappa*t + \
#	2*kappa*dBa2*(-2*kappa+ B*np.exp(-np.conjugate(B)*t) + np.conjugate(B)*np.exp(-B*t))))* np.exp(-gamma*t)*.5
	return np.exp(- D**2 * dBa2 * ( nd*(1+np.exp(-2*kappa*t)) + .5*(3-np.exp(-2*kappa*t)) + 1j*np.sin(oma*t)*np.exp(-kappa*t) - np.exp(-kappa*t)*np.cos(oma*t)*(1+2*nd) - 1j*oma/2/kappa*(1-np.exp(-2*kappa*t)) + 2*kappa*t + 2*kappa*dBa2*(-2*kappa+ B*np.exp(-np.conjugate(B)*t) + np.conjugate(B)*np.exp(-B*t))))* np.exp(-gamma*t)*.5
#	return np.exp(-D**2*dBa2*(.5*(1-1j*oma/kappa)*(1-np.exp(-2*kappa*t))+1j*np.exp(-kappa*t)*np.sin(oma*t)-np.exp(-kappa*t)*np.cos(oma*t)-dBa2*(oma**2-3*kappa**2+2*kappa*(B*np.exp(-np.conjugate(B)*t)+np.conjugate(B)*np.exp(-B*t)))))*.5*np.exp(-gamma*t)
#	return np.exp(-D**2/oma**2*((2*nd+1)*(1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t))*.5*np.exp(-gamma*t)
#	return (1+D**2/oma**2*2*(1-np.cos(oma*t)))*np.exp(-D**2/oma**2*( (1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t) )*.5*np.exp(-gamma*t)

##################
### PARAMETERS ###
##################
endt  = 4000
Nt    = 2**18
t     = np.linspace(0,endt, Nt)
D     = 0.7
oma   = 10 #in GHz
kappav = np.array([0.0001,0.1,1])  #in GHz
gamma = 0.001 #in GHz
hbar  = 6.62607004
kb    = 1.38064852
T     = 0.00001
nd    = 1./(np.exp(hbar*oma/kb/T)-1)

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

#fig,ax = plt.subplots(1,3,figsize=(40,8))
fig,ax = plt.subplots(1,2,figsize=(25,8))

#kappa=0.
#evol=rho_d(D,gamma,oma, kappa,nd,t)
#ax[0].plot(t,np.abs(evol)**2,color=colors[collab[0]],ls=linest[0],lw=linew[0])
#fourr = np.fft.fft(evol)
#four = np.fft.fftshift(fourr)*2/fourr.size
#freqr = np.fft.fftfreq(t.size,endt/(Nt-1))
#freq = np.fft.fftshift(freqr)
#ax[1].semilogy(freq,four.real,color=colors[collab[0]],ls=linest[0],lw=linew[0])

for i in range(0,kappav.size):
#for i in range(kappav.size-1,-1,-1):
	kappa=kappav[i]
	print("kappa is: ",kappav[i])
	sys.stdout.flush()	

	evol=rho_d(D,gamma,oma, kappa,nd,t)
	ax[0].plot(t,np.abs(evol)**2,color=colors[collab[i]],ls=linest[i],lw=linew[0])

	fourr = np.fft.fft(evol)
	four = np.fft.fftshift(fourr)*2/fourr.size
	freqr = np.fft.fftfreq(t.size,endt/(Nt-1))
	freq = np.fft.fftshift(freqr)

	ax[1].semilogy(freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[0])#,freq,four.imag)
#	ax[1].semilogy(freq,four.imag,color=colors[collab[i]],ls=linest[i],lw=linew[0])#,freq,four.imag)
#	ax[2].semilogy(freq,four.real**2+four.imag**2,color=colors[collab[i]],ls=linest[i],lw=linew[0])#,freq,four.imag)

ax[0].grid(True)
ax[1].grid(True)
#ax[2].grid(True)
ax[0].legend([0.0001,0.1,1],fontsize=20)
ax[1].legend([0.0001,0.1,1],fontsize=20)
ax[0].set_xlabel('$t$ (10 ps)',fontsize=30)
ax[1].set_xlabel('$\omega$ (100 GHz)',fontsize=30)
#ax[2].set_xlabel('Frequency',fontsize=30)
ax[0].set_ylabel('$\left|P(t)\\right|^2$',fontsize=30)
ax[1].set_ylabel('$\Re{P(\omega)}$',fontsize=30)
#ax[2].set_ylabel('$|P(\omega)|^2$',fontsize=30)
ax[0].set_ylim(-0.01,.25)
ax[0].set_xlim(0,200)
ax[1].set_xlim(-30,30)
ax[1].set_ylim(10**(-8),1)
#ax[2].set_ylim(10**(-4),10**4)

### TIMER ENDS ###
end=time.time()
h = int((end-now)/3600.)
m = int((end-now)/60.-h*60)
s = int((end-now)-h*3600-m*60)
print('%02d:%02d:%02d' %(h,m,s))
#plt.show()
#plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_nodamp_evol+spec_Fock.png")
#plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_damp_evol+spec_Fock.png")
#plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_nodamp_evol+spec_T=0.png")
plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_damp_evol+spec_T=0.png")


