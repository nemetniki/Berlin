#!/usr/bin/python3.4
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.misc import factorial

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rc('font',family='FreeSerif')
mpl.rc('xtick',labelsize=25)
mpl.rc('ytick',labelsize=25)

now = time.time()

###############################################
### FUNCTIONS FOR TIME-DEPENDENT ABSORPTION ###
###############################################
def rho_damp_T(D, gamma, oma, kappa, nd, t):
	t    = t
	B    = 1j*oma + kappa
	dBa2 = 1/(oma**2 + kappa**2)

	return np.exp(- D**2 * dBa2 * ( nd*(1+np.exp(-2*kappa*t)) + .5*(3-np.exp(-2*kappa*t)) + 1j*(oma**2-3*kappa**2)*dBa2*np.sin(oma*t)*np.exp(-kappa*t) - np.exp(-kappa*t)*np.cos(oma*t)*(1+2*nd+1j*4*oma*kappa*dBa2) + 1j*oma/2/kappa*(1-np.exp(-2*kappa*t)) + 2*np.conjugate(B)*t + 2*kappa*dBa2*(-2*np.conjugate(B)+ B*np.exp(-np.conjugate(B)*t) + np.conjugate(B)*np.exp(-B*t))))* np.exp(-gamma*t)*.5
#	return np.exp(-D**2*dBa2*(.5*(1-1j*oma/kappa)*(1-np.exp(-2*kappa*t))+1j*np.exp(-kappa*t)*np.sin(oma*t)-np.exp(-kappa*t)*np.cos(oma*t)-dBa2*(oma**2-3*kappa**2+2*kappa*(B*np.exp(-np.conjugate(B)*t)+np.conjugate(B)*np.exp(-B*t)))))*.5*np.exp(-gamma*t)

def rho_damp_F(D, gamma, oma, kappa, nd, t, NFock):
	t    = t
	B    = 1j*oma + kappa
	dBa2 = 1/(oma**2 + kappa**2)
	gam2 = D**2*dBa2*( 1+np.exp(-2*kappa*t)-np.exp(-kappa*t)*2*np.cos(oma*t) )
	
	cav_coef = 0.
	for iF in range(0,NFock+1):
		cav_coef += (-gam2)**iF/( factorial(iF)**2*factorial(NFock-iF) )
	cav_coef = cav_coef*factorial(NFock)

	#return (1 + D**2*dBa2*( 1+np.exp(-2*kappa*t)-np.exp(-kappa*t)*2*np.cos(oma*t) )) * \
	return cav_coef * \
	np.exp(- D**2 * dBa2 * ( .5*(3-np.exp(-2*kappa*t)) + 1j*(oma**2-3*kappa**2)*dBa2*np.sin(oma*t)*np.exp(-kappa*t) - \
	np.exp(-kappa*t)*np.cos(oma*t)*(1+1j*4*oma*kappa*dBa2) + 1j*oma/2/kappa*(1-np.exp(-2*kappa*t)) + 2*np.conjugate(B)*t + \
	2*kappa*dBa2*(-2*np.conjugate(B)+ B*np.exp(-np.conjugate(B)*t) + np.conjugate(B)*np.exp(-B*t))))* np.exp(-gamma*t)*.5

def rho_nodamp_T(D, gamma, oma,nd, t):
	return np.exp(-D**2/oma**2*((2*nd+1)*(1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t))*.5*np.exp(-gamma*t)

def rho_nodamp_F(D, gamma, oma,nd, t,NFock):
	gam2 = D**2/oma**2*2*(1-np.cos(oma*t))
	cav_coef = 0.
	for iF in range(0,NFock+1):
		cav_coef += (-gam2)**iF/( factorial(iF)**2*factorial(NFock-iF) )
	cav_coef = cav_coef*factorial(NFock)
	return cav_coef*np.exp(-D**2/oma**2*( (1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t) )*.5*np.exp(-gamma*t)

##################
### PARAMETERS ###
##################
damp  = True
Fock  = False
NFock = 10
show  = False
T     = 0.00001

D     = 0.005#0.7
oma   = 0.01#10.#np.pi/8.#10 #in GHz
kappav = np.array([0.001])#np.array([0.001,0.1,1])  #in GHz
gamma = 0.001 #in GHz

hbar  = 6.62607004
kb    = 1.38064852
nd    = 1./(np.exp(hbar*oma/kb/T)-1)

endt  = 60000#6000
Nt    = 2**18
t     = np.linspace(0,endt, Nt)

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

if Fock==True:
	rho_norm = rho_nodamp_F(D,gamma,10, nd,t,NFock)
	print("Fock")
else:
	rho_norm = rho_nodamp_T(D,gamma,10, nd,t)
	print("no Fock")
norm = np.abs(np.sum(rho_norm*2*endt/(Nt)))**2

if damp==True:
	for i in range(0,kappav.size):
		kappa=kappav[i]
		print("kappa is: ",kappav[i])
		sys.stdout.flush()	
		
		if Fock==True:
			rho_wn = rho_damp_F(D,gamma,oma, kappa,nd,t,NFock)
		else:
			rho_wn = rho_damp_T(D,gamma,oma, kappa,nd,t)
		evol   = rho_wn/np.sqrt(norm)
		ax[0].plot(t,np.abs(rho_wn)**2,color=colors[collab[i]],ls=linest[i],lw=linew[0])
	
		fourr = np.fft.fft(evol)
		four = np.fft.fftshift(fourr)/(Nt)*endt/np.sqrt(norm)*2
		freqr = np.fft.fftfreq(t.size,endt/(Nt-1))
		freq = np.fft.fftshift(freqr)
	
		ax[1].plot(2*np.pi*freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[0])#,freq,four.imag)
	#	ax[1].semilogy(freq,four.imag,color=colors[collab[i]],ls=linest[i],lw=linew[0])#,freq,four.imag)
	#	ax[2].semilogy(freq,four.real**2+four.imag**2,color=colors[collab[i]],ls=linest[i],lw=linew[0])#,freq,four.imag)
else:
	if Fock==True:
		rho_wn = rho_nodamp_F(D,gamma,oma,nd,t,NFock)
	else:
		rho_wn = rho_nodamp_T(D,gamma,oma,nd,t,)

#norm = np.abs(np.sum(rho_wn*2*endt/(Nt)))**2
	evol   = rho_wn/np.sqrt(norm)
	ax[0].plot(t,np.abs(rho_wn)**2,color=colors[collab[0]],ls=linest[0],lw=linew[0])
	fourr = np.fft.fft(evol)
	four = np.fft.fftshift(fourr)*2/(Nt)*endt/np.sqrt(norm)
	freqr = np.fft.fftfreq(t.size,endt/(Nt-1))
	freq = np.fft.fftshift(freqr)
	ax[1].semilogy(2*np.pi*freq,four.real,color=colors[collab[0]],ls=linest[0],lw=linew[0])

ax[0].grid(True)
ax[1].grid(True)
#ax[2].grid(True)
ax[0].legend([0.001,0.1,1],fontsize=20)
ax[1].legend([0.001,0.1,1],fontsize=20)
ax[0].set_xlabel('$t$ (10 ps)',fontsize=30)
ax[1].set_xlabel('$\omega$ (100 GHz)',fontsize=30)
#ax[2].set_xlabel('Frequency',fontsize=30)
ax[0].set_ylabel('$\left|P(t)\\right|^2$',fontsize=30)
ax[1].set_ylabel('$\Re{P(\omega)}$',fontsize=30)
#ax[2].set_ylabel('$|P(\omega)|^2$',fontsize=30)
#ax[0].set_ylim(-0.01,.25)
#ax[0].set_xlim(0,2000)
ax[0].set_xlim(0,500)
ax[1].set_xlim(-4*oma,4*oma)
#ax[1].set_xlim(-1,1)
#ax[1].set_ylim(10**(-8),1)
#ax[2].set_ylim(10**(-4),10**4)

### TIMER ENDS ###
end=time.time()
h = int((end-now)/3600.)
m = int((end-now)/60.-h*60)
s = int((end-now)-h*3600-m*60)
print('%02d:%02d:%02d' %(h,m,s))

if show==True:
	plt.show()
else:
	if damp==True:
		if Fock==True:
			plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_damp_evol+spec_Fock%d_D=%dp10.png" % (NFock,D*10))
#			plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_damp_evol+spec_Fock_s.png")
		else:
			plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_damp_evol+spec_T=0_D=%dp10.png" % (D*10))
#			plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_damp_evol+spec_T=0_s.png")
	else:
		if Fock==True:
			plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_nodamp_evol+spec_Fock%d_D=%dp10.png" % (NFock,D*10))
#			plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_nodamp_evol+spec_Fock_s.png")
		else:
			plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_nodamp_evol+spec_T=0_D=%dp10.png" % (D*10))
#			plt.savefig("/home/niki/Dokumente/Python/Analytic plots/analytic_nodamp_evol+spec_T=0_s.png")


