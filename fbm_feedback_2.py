#!/usr/bin/python3.4
import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import psutil
from scipy.misc import factorial

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rc('font',family='FreeSerif')
mpl.rc('xtick',labelsize=25)
mpl.rc('ytick',labelsize=25)

from memory_profiler import profile

####################
### TIMER STARTS ###
####################
start = time.time()

###############################
### TIME-DEPENDENT FUNCTION ###
###############################
def rho_nodamp_T(D, gamma, oma,nde, t):
	return np.exp(-D**2/oma**2*(nde*(1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t))*.5*np.exp(-gamma*t)

def rho_nodamp_F(D, gamma, oma,nde, t):
	return (1+D**2/oma**2*2*(1-np.cos(oma*t)))*np.exp(-D**2/oma**2*( (1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t) )*.5*np.exp(-gamma*t)

#@profile
def f(m, tp, expA, expB):
	res    = np.zeros(2*(1+k.size),dtype=np.complex128)
	nv     = np.arange(0,m+1)
	An     = np.array([A,]*(m+1)).transpose()
	res[2:2+k.size]  = Ck * kAB**m * ( expA - expB * np.sum((An+B)**nv/factorial(nv)*tp**nv,axis=1) ) #Gk
	An = None
	res[1] = -D/B * kappa**m * ( expB * np.sum(tp**(m-nv)*Br**nv/factorial(m-nv)) - Br**m ) #gamma
	nv = None
	res[0] = D * expB * kappa**m / factorial(m) * tp**m # F
	for n in range(0,m+1):
		lv=np.arange(0,n+1)
		if n==0:
			LNkn  = expB-1
		else:
			LNkn += (A+B)**n * ( expB * np.sum(tp**(n-lv)*Br**lv/factorial(n-lv)) - Br**n )
	res[2+k.size:] = Ck * kAB**m * ( Ar*(expA-1) + Br*LNkn ) # Nk
	LNkn = None
	return res

#@profile
def rho_fb(Nt, tau, dt, k, nke, nde, A, Ar, B, Br, D, Ck, kAB, Fock):

	phi_env = np.zeros(Nt,dtype=np.complex128)
	phi_cav = np.zeros(Nt,dtype=np.complex128)
	rho_fin = np.zeros(Nt,dtype=np.complex128)

	for mp in range(0,M+1):
		if mp==M and Nt>tau:
			tmax = Nt-M*tau			
		else:
			tmax = min(tau,Nt)
		for it in range(0,tmax):

			
			for ms in range(0,mp+1):
				tp   = ((mp-ms)*tau+it)*dt
				
				expA = np.exp(A*tp)
				expB = np.exp(-B*tp)

				if ms==0: #res: F,gamma,Gk,Nk
					res = np.concatenate(( np.array([D * expB, -D * Br * ( expB - 1 )]),\
								Ck * ( expA - expB ),\
								Ck * ( Ar*(expA-1) + Br * (expB-1) ) ))
				else:
					res += f(ms, tp, expA, expB)
		
			phi_cav[mp*tau+it] = res[0] * np.conjugate(res[1])
			phi_env[mp*tau+it] = np.sum( res[2:2+k.size] * np.conjugate(res[2+k.size:]) * dk)

			Bath = -.5 * np.sum( np.abs(res[2+k.size:])**2 * nke * dk)

			if Fock==True:
				Cav_coef = np.complex_( 1. + np.abs(res[1])**2 )
				Cav_exp  = np.complex_( -.5 * np.abs(res[1])**2 )
			else:
				Cav_coef = np.complex_(1.)
				Cav_exp = np.complex_( -.5 * np.abs(res[1])**2 * nde )

			res = None

			Phi = np.imag( np.sum( (phi_cav[0:(it+1+mp*tau)] + phi_env[0:(it+1+mp*tau)])*dt ) )
			rho_fin[it+mp*tau] = .5 * Cav_coef * np.exp( Cav_exp + Bath - 1j*Phi- gam*(it+mp*tau)*dt )

	return rho_fin
		

##################	
### PARAMETERS ###
##################
Fock   = False
show   = True
#Trick  = False
Tau    = True

D      = .7
oma    = 10 #in 100GHz
ome    = 0.
kappav = np.array([0.001,0.005,0.01])  #in 100GHz
gam    = 0.001 #in 100GHz
c      = 0.003

endk   = 2000#900#0.
labek  = int(endk/1000.)
Numk   = 12000#5500#0
labNk  = int(Numk/1000.)
k      = np.linspace(-endk,endk,Numk)# + ome*100.
dk     = k[1]-k[0]
A      = -1j*c*k
Ar     = 1/A

hbar   = 6.62607004 
kb     = 1.38064852
T      = 0.00001
therm  = hbar/(kb*T)
hbar   = None
kb     = None
nde    = 2./(np.exp(therm*oma)-1) + 1.
nke    = 2./(np.exp(therm*c*np.abs(k))-1) + 1.
therm = None

endt   = 750#600#0.
labet  = int(endt/1000.)
Nt     = 2**13#6
labNt  = int(np.log2(Nt))
t      = np.linspace(0,endt,Nt)
kaptau = 1.
dt     = (endt)/(Nt-1)


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
#fig2,ax2 = plt.subplots(2,2,figsize=(25,18))

if Fock==True:
	rho_norm = rho_nodamp_F(D,gam,oma, nde,t)
else:
	rho_norm = rho_nodamp_T(D,gam,oma, nde,t)
norm = np.abs(np.sum(rho_norm*2*endt/(Nt)))**2
rho_norm=None

for i in range(0,kappav.size):
#for i in range(kappav.size-1,-1,-1):

	kappa=kappav[i]
	print("kappa is: ",kappa)
	tau    = int(kaptau/kappa/dt)
	print("tmax-tau is: ",endt-tau*dt)
	print(Nt,tau)
	M      = int(Nt/tau)
	print("M is: ",M)
	sys.stdout.flush()	

	B   = 1j*oma + kappa
	Br  = 1/B
	kAB = kappa/(A+B)

	g0  = np.sqrt(kappa*2*c/np.pi)
	if Tau==True:
		Ck  = -1j*D*g0*np.sin(k*c*.5*tau*dt)/(A+B)
	else:
		Ck  = -1j*g0*D/(A+B)

	#################################################
	### EVALUATION OF THE TIME-DEPENDENT SOLUTION ###
	#################################################
	rho_wn=rho_fb(Nt,tau,dt,k,nke,nde,A,Ar,B,Br,D,Ck,kAB,Fock)
	evol = rho_wn/np.sqrt(norm)

	ax[0].plot(t,np.abs(rho_wn)**2,color=colors[collab[i]],ls=linest[i],lw=linew[0])
	rho_wn=None

	now2 = time.time()
	nowh = int((now2-start)/3600.)
	nowm = int((now2-start)/60.-nowh*60)
	nows = int((now2-start)-nowh*3600-nowm*60)
	print("at cycle %d after the integration: %02d:%02d:%02d"% (i, nowh, nowm, nows))
	sys.stdout.flush()	

	#########################
	### FOURIER TRANSFORM ###
	#########################
#	fourr = np.fft.fft(evol)
	four = np.fft.fftshift(np.fft.fft(evol))*2/(Nt)*endt/np.sqrt(norm)
	evol=None
	if i==kappav.size-1:
		norm=None
	freq = np.fft.fftshift(np.fft.fftfreq(Nt,(endt)/(Nt-1)))

	now3 = time.time()
	nowh = int((now3-start)/3600.)
	nowm = int((now3-start)/60.-nowh*60)
	nows = int((now3-start)-nowh*3600-nowm*60)
	print("at cycle %d after the FFT: %02d:%02d:%02d"% (i, nowh, nowm, nows))
	sys.stdout.flush()	

	############
	### PLOT ###
	############
	ax[1].semilogy(2*np.pi*freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[0])
	ax[1].grid(True)

ax[0].grid(True)
ax[1].grid(True)
ax[0].legend([0.001,0.005,0.01],fontsize=20)
ax[1].legend([0.001,0.005,0.01],fontsize=20)
ax[0].set_xlabel('$t$ (10 ps)',fontsize=30)
ax[1].set_xlabel('$\omega$ (100 GHz)',fontsize=30)
ax[0].set_ylabel('$\left|P(t)\\right|^2$',fontsize=30)
ax[1].set_ylabel('$\Re{P(\omega)}$',fontsize=30)
#ax[0].set_ylim(-0.01,.25)
ax[0].set_xlim(0,400)
ax[0].set_xlim(0,endt)

#ax[1].set_ylim(10**(-8),10)
ax[1].set_xlim(-40,40)


##################
### TIMER ENDS ###
##################
end=time.time()
h = int((end-start)/3600.)
m = int((end-start)/60.-h*60)
s = int((end-start)-h*3600-m*60)
print('%02d:%02d:%02d' %(h,m,s))

if show==True:
	plt.show()
else:
	if Tau==True:

		if Fock==True:
			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_fb_T=0_tau=%.2f_Fock1_new.png" % (kaptau))
		else:
			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_fb_T=%d_tau=%.2f_new.png" % (T,kaptau))
	else:
		if Fock==True:
			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_fb_T=0_notau_Fock1.png" % (labek,labNk,labet,labNt))
		else:
			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_fb_T=%d_notau.png" % (labek,labNk,labet,labNt,T))

