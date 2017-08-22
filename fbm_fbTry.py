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

def rho_fb(Nt, tau, dt, k, nk, nd, A, Ar, B, Br, D, Ck, kAB):

	phi_env = np.zeros(Nt,complex)
	phi_cav = np.zeros(Nt,complex)
	rho_fin = np.zeros(Nt,complex)
	M       = 0

	for it in range(0,Nt):
		if it%tau==0 and it!=0:
	#		M = int(it/tau)
			M += 1
			print("M increased")

		Nk = np.zeros(k.size,complex)
		Gk = np.zeros(k.size,complex)
		ga = 0.+0.*1j
		for m in range(0,M+1):
			tp   = (it-m*tau)*dt
			expA = np.exp(A*tp)
			expB = np.exp(-B*tp)
			
			L_Gk_n = np.zeros(k.size,complex)
			L_Nk_n = np.zeros(k.size,complex)
			for n in range(0,m+1):
				lv = np.arange(0,n+1)
				L_Nk_l = np.sum( tp**(n-lv) * Br**lv / factorial(n-lv) )
				L_Nk_n += (A+B)**n * ( expB*L_Nk_l - Br**n )

				L_Gk_n += (A+B)**n / factorial(n) * tp**n
			
			Nk += Ck * ( kAB**m * ( Ar*(expA-1) + Br * L_Nk_n ) )
	
			nv = np.arange(0,m+1)
			L_ga_n = np.sum( tp**(m-nv) * Br**nv / factorial(m-nv) )
			ga += -D*Br * kappa**m * ( expB*L_ga_n - Br**m )
			
			Gk += Ck * kAB**m * ( expA - expB*L_Gk_n )

		mv = np.arange(0,M+1)
		F  = D * np.sum( kappa**mv / factorial(mv) * np.exp(-B*(it-mv*tau)*dt) * ((it-mv*tau)*dt)**mv )
		
		if Fock==True:
			Cav_coef = 1. + np.abs(ga)**2
			Cav_exp  = -.5 * np.abs(ga)**2
		else:
			Cav_coef = 1.
			Cav_exp = -.5 * np.abs(ga)**2 * (2*nd+1)

		Bath = -.5 * np.sum( np.abs(Nk)**2 * (2*nk+1) * dk )
		
		phi_env[it] = np.sum( Gk * np.conjugate(Nk) * dk )
		phi_cav[it] = F * np.conjugate(ga)
		Phi         = np.imag( np.sum( (phi_cav[0:(it+1)] + phi_env[0:(it+1)])*dt ) )
	
		rho_fin[it] = .5 * Cav_coef * np.exp( Cav_exp + Bath - 1j*Phi - gam*it*dt )

	return rho_fin
		

##################	
### PARAMETERS ###
##################
Fock   = True
show   = False

D      = .7
oma    = 10 #in 100GHz
ome    = 0.
kappav = np.array([0.0001])#,0.1,1])  #in 100GHz
gam    = 0.001 #in 100GHz
c      = 0.003

endk   = 5000.
Nk     = 20000
k      = np.linspace(-endk,endk,Nk)# + ome*100.
dk     = k[1]-k[0]
A      = -1j*c*k
Ar     = 1/A

hbar   = 6.62607004 
kb     = 1.38064852
T      = 0.00001
therm  = hbar/(kb*T)
nd     = 1./(np.exp(hbar*oma/kb/T)-1)
nk     = 1/(np.exp(therm*c*np.abs(k))-1)

endt   = 4000.
Nt     = 2**15
t      = np.linspace(0,endt,Nt)
tau    = 1.1*Nt
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

if Fock==True:
	rho_norm = rho_nodamp_F(D,gam,oma, nd,t)
else:
	rho_norm = rho_nodamp_T(D,gam,oma, nd,t)
norm = np.abs(np.sum(rho_norm*2*endt/(Nt)))**2

for i in range(0,1):#kappav.size):
#for i in range(kappav.size-1,-1,-1):

	kappa=kappav[i]
	print("kappa is: ",kappav[i])
	sys.stdout.flush()	

	g0  = np.sqrt(kappa*2*c/np.pi)
	gk  = g0#*np.sin(k*c*.5*tau*dt)
	B   = 1j*oma + kappa
	Br  = 1/B
	Ck  = -1j*gk*D/(A+B)
	kAB = kappa/(A+B)

	#################################################
	### EVALUATION OF THE TIME-DEPENDENT SOLUTION ###
	#################################################
	rho_wn=rho_fb(Nt,tau,dt,k,nk,nd,A,Ar,B,Br,D,Ck,kAB)
	evol = rho_wn/np.sqrt(norm)
#	ax[0].plot(t,np.abs(rho_wn)**2,color=colors[collab[i]],ls=linest[i],lw=linew[0])
	ax[0].plot(t,np.real(rho_wn),color=colors[collab[0]],ls=linest[0],lw=linew[0])
	ax[0].plot(t,np.imag(rho_wn),color=colors[collab[1]],ls=linest[1],lw=linew[0])
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
#ax[0].set_xlim(0,200)
ax[0].set_xlim(0,endt)

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
		plt.savefig("/home/niki/Dokumente/Python/Numerical plots/Test_numeric2_kend5e_Nk100e_tend20e_Nt2e18_fb_T=0_Fock1_s.png")
	else:
		plt.savefig("/home/niki/Dokumente/Python/Numerical plots/Test_numeric2_kend5e_Nk100e_tend20e_Nt2e18_fb_T=%d_wide_s.png" % T)


