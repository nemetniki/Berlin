#!/usr/bin/python3.4
import matplotlib as mpl
mpl.use('Agg')
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
def rho_nodamp_T(D, gamma, oma,nd, t):
	return np.exp(-D**2/oma**2*((2*nd+1)*(1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t))*.5*np.exp(-gamma*t)

def rho_nodamp_F(D, gamma, oma,nd, t):
	return (1+D**2/oma**2*2*(1-np.cos(oma*t)))*np.exp(-D**2/oma**2*( (1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t) )*.5*np.exp(-gamma*t)

def rho_d(gd, gam, oma, kappa, nd, endt, Nt, k, therm, Fock):

	dt  = (endt)/(Nt-1)
	dk  = k[1]-k[0]
	c    = 0.003
	g02  = kappa*2*c/np.pi
	B    = 1j*oma + kappa
	coef = np.abs(gd/B)**2
	
	#################
	### |N_d^k|^2 ###
	#################
	def Nd2k(k,ex,Ca,Sa):

		cof = coef*g02/(c*k)**2
		den = 1/(kappa**2+(oma-c*k)**2)
		Ck  = np.cos(c*k*t)
		Sk  = np.sin(c*k*t)
		A2  = (c*k)**2 * (2*Ca-ex) * ex
		B2  = (kappa**2+oma**2) * (2*Ck-1)
		AB  = oma*(Ck+(Ca-Ca*Ck-Sk*Sa)*ex) + kappa*(-Sk+(Sa-Sa*Ck+Sk*Ca)*ex)

		return cof * ( 1 - den*(A2 + B2 - 2*c*k*AB) )

	nk = 1/(np.exp(therm*c*np.abs(k))-1)
	
	####################################
	### EVALUATION FOR EACH TIMESTEP ###
	####################################
	rho_final = np.zeros(Nt, complex)
	for it in range(0,Nt):

		t    = it*dt#*2*np.pi
		ex   = np.exp(-kappa*t)
		Ca   = np.cos(oma*t)
		Sa   = np.sin(oma*t)

		gamma2  = coef * ( 1 + ex**2 - 2*ex*Ca ) #|gamma|^2
#		rho_cav = (1 + gamma2) * np.exp(- 0.5* gamma2)
		
		ksum = np.sum(Nd2k(k,ex,Ca,Sa)*(2*nk+1)*dk)			
#		rho_bath = np.exp(-.5*(ksum))

		rho_phi  = np.exp( -1j * coef * ( ex*Sa- oma/2/kappa*(1-ex**2) ) )
		if Fock==True:
			rho_final[it] = .5 * (1+gamma2) * np.exp( - 0.5*(gamma2+ksum) - 1j * coef * ( ex*Sa- oma/2/kappa*(1-ex**2) ) - 1j*ome*t - gam*t)
		else:
			rho_final[it] = .5 * np.exp( - 0.5*(gamma2*(2*nd+1)+ksum) - 1j * coef * ( ex*Sa- oma/2/kappa*(1-ex**2) ) - 1j*ome*t - gam*t)

	return rho_final

##################	
### PARAMETERS ###
##################
Fock   = False
show   = False

D      = .7
oma    = 10 #in GHz
kappav = np.array([0.0001,0.1,1])  #in GHz
gam    = 0.001 #in GHz
hbar   = 6.62607004 
kb     = 1.38064852
T      = 3000.#00001
nd     = 1./(np.exp(hbar*oma/kb/T)-1)
endt   = 20000.
Nt     = 2**18
t      = np.linspace(0,endt,Nt)
endk   = 5000.
Nk     = 10000
ome    = 0.
k      = np.linspace(-endk,endk,Nk)# + ome*100.
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

if Fock==True:
	rho_norm = rho_nodamp_F(D,gam,oma, nd,t)
else:
	rho_norm = rho_nodamp_T(D,gam,oma, nd,t)
norm = np.abs(np.sum(rho_norm*2*endt/(Nt)))**2

for i in range(0,kappav.size):
#for i in range(kappav.size-1,-1,-1):

	kappa=kappav[i]
	print("kappa is: ",kappav[i])
	sys.stdout.flush()	

	#################################################
	### EVALUATION OF THE TIME-DEPENDENT SOLUTION ###
	#################################################
	rho_wn=rho_d(D,gam,oma,kappa,nd,endt,Nt,k,therm,Fock)
	evol = rho_wn/np.sqrt(norm)
	ax[0].plot(t,np.abs(rho_wn)**2,color=colors[collab[i]],ls=linest[i],lw=linew[0])
	now2 = time.time()
	nowh = int((now2-now)/3600.)
	nowm = int((now2-now)/60.-nowh*60)
	nows = int((now2-now)-nowh*3600-nowm*60)
	print("at cycle %d after the integration: %02d:%02d:%02d"% (i, nowh, nowm, nows))
	sys.stdout.flush()	

	#########################
	### FOURIER TRANSFORM ###
	#########################
	fourr = np.fft.fft(evol)
	four = np.fft.fftshift(fourr)*2/(Nt)*endt/np.sqrt(norm)
	freqr = np.fft.fftfreq(Nt,(endt)/(Nt-1))
	freq = np.fft.fftshift(freqr)

	now3 = time.time()
	nowh = int((now3-now)/3600.)
	nowm = int((now3-now)/60.-nowh*60)
	nows = int((now3-now)-nowh*3600-nowm*60)
	print("at cycle %d after the FFT: %02d:%02d:%02d"% (i, nowh, nowm, nows))
	sys.stdout.flush()	

	############
	### PLOT ###
	############
	ax[1].semilogy(2*np.pi*freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[0])
	ax[1].grid(True)

ax[0].grid(True)
ax[1].grid(True)
ax[0].legend([0.0001,0.1,1],fontsize=20)
ax[1].legend([0.0001,0.1,1],fontsize=20)
ax[0].set_xlabel('$t$ (10 ps)',fontsize=30)
ax[1].set_xlabel('$\omega$ (100 GHz)',fontsize=30)
ax[0].set_ylabel('$\left|P(t)\\right|^2$',fontsize=30)
ax[1].set_ylabel('$\Re{P(\omega)}$',fontsize=30)
#ax[0].set_ylim(-0.01,.25)
ax[0].set_xlim(0,200)
ax[1].set_ylim(10**(-8),10)
ax[1].set_xlim(-40,40)


##################
### TIMER ENDS ###
##################
end=time.time()
h = int((end-now)/3600.)
m = int((end-now)/60.-h*60)
s = int((end-now)-h*3600-m*60)
print('%02d:%02d:%02d' %(h,m,s))

if show==True:
	plt.show()
else:
	if Fock==True:
		plt.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_kend=5000_therm_T=0_Fock1_s.png")
	else:
		plt.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_kend=5000_therm_T=%d_wide_s.png" % T)


