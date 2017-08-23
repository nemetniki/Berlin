#!/usr/bin/python3.4
import matplotlib as mpl
mpl.use('Agg')
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

@profile
def rho_fb(Nt, tau, dt, k, nke, nde, A, Ar, B, Br, D, Ck, kAB, Fock):

	phi_env = np.zeros(Nt,dtype=np.complex128)
	phi_cav = np.zeros(Nt,dtype=np.complex128)
	rho_fin = np.zeros(Nt,dtype=np.complex128)
	Phi     = np.zeros(tau)

	for m in range(0,M+1):

		looptime=time.time()
		h       = int((looptime-start)/3600.)
		mi      = int((looptime-start)/60.-h*60)
		s       = int((looptime-start)-h*3600-mi*60)
#		print("m is %d at %02d:%02d:%02d" % (m,h,mi,s))
#		sys.stdout.flush()	

		tvec = np.linspace(0,(tau-1)*dt,tau)+m*tau*dt
#		expB = np.exp(-B*tvec)
		tsec   = np.array([tvec,]*k.size).transpose()
#		Arv,expBv = np.meshgrid(Ar,np.expB)
		nv    = np.arange(0,m+1)

		expA      = np.exp(A*tsec)
		
		for n in range(0,m+1):

			if n==0:
				L_Nk_n  = ( np.exp(-B*tsec)-1)
				L_Gk_n  = np.ones(tsec.shape)+0*1j
			else:
				tsecl   =  np.array([ tvec,]*(n+1) ).transpose()
				lv      = np.arange(0,n+1)
				L_Nk_l  = np.array([np.sum( tsecl**(n-lv)*Br**lv/factorial(n-lv), axis=1),]*tsec.shape[1]).transpose()
				lv      = None
				tsecl   = None
				L_Nk_n += (A+B)**n * ( np.exp(-B*tsec)*L_Nk_l - Br**n )
				L_Gk_n += (A+B)**n / factorial(n) * tsec**n

		if m==0:				
			Nk     = Ck * ( (expA-1)/A + Br * L_Nk_n )
			L_Nk_n = None
			Gk     = Ck*(expA- np.exp(-B*tsec))
			expA = None
			tsec = None
			if m==M:
				phi_env[(M*tau):Nt] = np.sum( Gk[0:(Nt-M*tau),:] * np.conjugate(Nk)[0:(Nt-M*tau),:] * dk, axis=1)
			else:
				phi_env[(m*tau):((m+1)*tau)] = np.sum( Gk * np.conjugate(Nk) * dk, axis=1)
		else:
			Nk    += Ck * ( kAB**m * ( Ar*(expA-1) + Br * L_Nk_n ) )
			L_Nk_n = None
			Gk    += Ck * kAB**m * ( expA -  np.exp(-B*tsec)*L_Gk_n )	
			L_Gk_n = None
			expA = None
			tsec = None
			if m==M:
				phi_env[(M*tau):Nt] = np.sum( Gk[0:(Nt-M*tau),:] * np.conjugate(Nk)[0:(Nt-M*tau),:] * dk, axis=1)
				Gk = None
				kAB= None
			else:
				phi_env[(m*tau):((m+1)*tau)] = np.sum( Gk * np.conjugate(Nk) * dk, axis=1)

		Bath = -.5 * np.sum( np.abs(Nk)**2 * nke * dk ,axis=1)
		if m==M:
			Nk = None					

		if m==0:
			ga     = -D*Br*( np.exp(-B*tvec)-1)
			F      = D* np.exp(-B*tvec)
			if m==M:
				phi_cav[(M*tau):Nt] = F[0:(Nt-M*tau)] * np.conjugate(ga)[0:(Nt-M*tau)]
			else:
				phi_cav[(m*tau):((m+1)*tau)] = F * np.conjugate(ga)
		else:
			tsecn  = np.array([tvec,]*(m+1)).transpose()
			ga    += -D*Br*kappa**m*( np.exp(-B*tvec)*np.sum(tsecn**(m-nv)*Br**nv/factorial(m-nv)) - Br**m )
			tsecn  = None
			F     += D*kappa**m/factorial(m)* np.exp(-B*tvec)*tvec**m
			if m==M:
				phi_cav[(M*tau):Nt] = F[0:(Nt-M*tau)] * np.conjugate(ga)[0:(Nt-M*tau)]
				F = None	
			else:
				phi_cav[(m*tau):((m+1)*tau)] = F * np.conjugate(ga)
	 
		if Fock==True:
			Cav_coef = np.complex_( 1. + np.abs(ga)**2 )
			Cav_exp  = np.complex_( -.5 * np.abs(ga)**2 )
		else:
			Cav_coef = np.ones(ga.size)
			Cav_exp = np.complex_( -.5 * np.abs(ga)**2 * nde )
	
	
		if m==M:
			ga = None
			for it in range(0,(Nt-M*tau)):
				Phi[it] = np.imag( np.sum( (phi_cav[0:(it+1+M*tau)] + phi_env[0:(it+1+M*tau)])*dt ) )
			rho_fin[(M*tau):Nt] = .5 * Cav_coef[0:(Nt-M*tau)] * np.exp( Cav_exp[0:(Nt-M*tau)] + Bath[0:(Nt-M*tau)] - 1j*Phi[0:(Nt-M*tau)] - gam*np.arange((M*tau),Nt)*dt )
			Bath                = None
#			Phi                 = None
		else:
			for it in range(0,tau):
				Phi[it] = np.imag( np.sum( (phi_cav[0:(it+1+m*tau)] + phi_env[0:(it+1+m*tau)])*dt ) )
			rho_fin[(m*tau):((m+1)*tau)] = .5 * Cav_coef * np.exp( Cav_exp + Bath - 1j*Phi - gam*np.arange((m*tau),((m+1)*tau))*dt )
			Bath                = None
#			Phi                 = None

	return rho_fin
		

##################	
### PARAMETERS ###
##################
Fock   = False
show   = False
#Trick  = False
Tau    = True

D      = .7
oma    = 10 #in 100GHz
ome    = 0.
kappav = np.array([0.001,0.005,0.01])  #in 100GHz
gam    = 0.001 #in 100GHz
c      = 0.003

endk   = 2000#9000.
labek  = int(endk/1000.)
Numk   = 12000#55000
labNk  = int(Numk/1000.)
k      = np.linspace(-endk,endk,Numk)# + ome*100.
dk     = k[1]-k[0]
A      = -1j*c*k
Ar     = 1/A

hbar   = 6.62607004 
kb     = 1.38064852
T      = 0.00001
therm  = hbar/(kb*T)
hbar=None
kb  =None
nde    = 2./(np.exp(therm*oma)-1) + 1.
nke    = 2./(np.exp(therm*c*np.abs(k))-1) + 1.
therm = None

endt   = 1500.#6000.
labet  = int(endt/1000.)
Nt     = 2**14#6
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
#	tp1 = np.linspace(0,((M+1)*tau-1)*dt,(M+1)*tau)
#	tpm = np.append(tp1,np.zeros((M+1)*tau-tp1.size)).reshape((M+1,tau))
#	expB   = np.exp(-B*tpm)

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
#	freq = freqr
#	freqr=None


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
#	if Trick==True:
#
#		if Fock==True:
#			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_kend%de_Nk%de_tend%de_Nt2e%d_fb_T=0_Fock1_trick2.png" % (labek,labNk,labet,labNt))
#			fig2.savefig("/home/niki/Dokumente/Python/Numerical plots/Check_fb_k%de_%deNk_t%de_2e%dNt_T=0_Fock1_trick2.png" % (labek,labNk,labet,labNt))
#		else:
#			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_kend%de_Nk%de_tend%de_Nt2e%d_fb_T=%d_wide_trick2.png" % (labek,labNk,labet,labNt,T))
#			fig2.savefig("/home/niki/Dokumente/Python/Numerical plots/Check_fb_k%de_%deNk_t%de_2e%dNt_T=%d_wide_trick2.png" % (labek,labNk,labet,labNt,T))
#	else:
#		if Fock==True:
#			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_kend%de_Nk%de_tend%de_Nt2e%d_fb_T=0_Fock1.png" % (labek,labNk,labet,labNt))
#			fig2.savefig("/home/niki/Dokumente/Python/Numerical plots/Check_fb_k%de_%deNk_t%de_2e%dNt_T=0_Fock1.png" % (labek,labNk,labet,labNt))
#		else:
#			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_kend%de_Nk%de_tend%de_Nt2e%d_fb_T=%d_wide.png" % (labek,labNk,labet,labNt,T))
#			fig2.savefig("/home/niki/Dokumente/Python/Numerical plots/Check_fb_k%de_%deNk_t%de_2e%dNt_T=%d_wide.png" % (labek,labNk,labet,labNt,T))

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

