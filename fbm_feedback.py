#!/usr/bin/python3.4
import matplotlib as mpl
#mpl.use('Agg')
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

def rho_fb(D, gam, oma, kappa, nd, endt, Nt, tau, k, therm, Fock):

	dt  = (endt)/(Nt-1)
	dk  = k[1]-k[0]
	c   = 0.003
	g0  = np.sqrt(kappa*2*c/np.pi)
	gk  = g0#*np.sin(k*c*tau*dt)
	A   = -1j*c*k
	B   = 1j*oma + kappa
	Br  = 1/B
	Ar  = 1/A
	Ck  = -1j*gk*D/(A+B)
	kAB = kappa/(A+B)
#	Ntau = int(Nt/tau)

	nk = 1/(np.exp(therm*c*np.abs(k))-1)
#	print(nk)

	F  = np.zeros(Nt,complex)
	ga = np.zeros(Nt,complex)
	vk = np.zeros(Nt,complex)
	rho_final = np.zeros(Nt,complex)

	for it in range(0,Nt):
		M = int(it/tau)

		loop_Nk_m = np.zeros(k.size,complex)
		loop_ga_m = 0.+0.*1j
		loop_Gk_m = np.zeros(k.size,complex)
		loop_F_m  = 0.+0.*1j
		for m in range(0,M+1):
			
#			print("m = ",m)
			tp = (it-m*tau) * dt
			expB = np.exp(-B*tp)
			expA = np.exp(A*tp)
			loop_Nk_n = np.zeros(k.size,complex)
			loop_Gk_n = np.zeros(k.size,complex)
			for n in range(0,m+1):
				loop_Gk_n = loop_Gk_n + (A+B)**n / factorial(n) * tp**n
										
				lv = np.arange(0,n+1)
				loop_Nk_l = np.sum(tp**(n-lv) * Br**lv / factorial(n-lv))
				
				loop_Nk_n += (A+B)**n * ( expB*loop_Nk_l - Br**n )
			
			nv = np.arange(0,m+1)
			loop_ga_n = np.sum( tp**(m-nv) * Br**nv / factorial(m-nv) )

			loop_Nk_m += kAB**m * ( Ar*(expA-1) + Br*loop_Nk_n )
#			loop_ga_m += kappa**m * ( expB*loop_ga_n - Br**m )
			loop_Gk_m = loop_Gk_m + kAB**m * ( expA - expB*loop_Gk_n )
#			loop_F_m  += kappa**m/factorial(m) * expB * tp**m
		
		mv     = np.arange(0,M+1)
		expBv  = np.exp(-B*(it-mv*tau)*dt)
		ga[it] = -D/B * np.sum( kappa**mv * (expBv*loop_ga_n - Br**mv) )
		Nk     = Ck * loop_Nk_m
		F[it]  = D * np.sum( kappa**mv/factorial(mv) * expBv * ((it-mv*tau)*dt)**mv )
		Gk     = Ck * loop_Gk_m
		
		cav    = - 0.5 * np.abs(ga[it])**2 * (2*nd+1)
		bath   = - 0.5 * np.sum( np.abs(Nk)**2*(2*nk+1)  * dk )
		vk[it] = np.sum( Gk*np.conjugate(Nk) ) * dk
		Phi    = np.imag( np.sum( F[1:(it+1)]*np.conjugate(ga[1:(it+1)]) + vk[1:(it+1)] ) ) * dt
		
		rho_final[it] = .5 * np.exp( cav  + bath - 1j*Phi - gam*it*dt )				
	print("bath",bath)
	t=np.linspace(0,endt,Nt)
	integ = np.abs(D/B)**2*(2*kappa*t + (1-np.exp(-2*kappa*t)) +2*kappa/np.abs(B)**2*(-2*kappa+np.conjugate(B)*np.exp(-B*t)+B*np.exp(-np.conjugate(B)*t)))
	calc=-integ
	print("calc",calc)
	Nka = np.abs(Nk)**2
	print("Nka",Nka)
	Nk2 = np.abs(g0*D/A/B)**2*( 1 - 1/np.abs(A+B)**2* ( np.abs(A)**2*(2*np.cos(oma*t[-1])-np.exp(-kappa*t[-1])) * np.exp(-kappa*t[-1]) +\
							np.abs(B)**2*(2*np.cos(c*k*t[-1])-1)-\
							2*oma*c*k*( ( np.cos(oma*t[-1])-np.cos((oma-c*k)*t[-1]))*np.exp(-kappa*t[-1] ) + np.cos(c*k*t[-1]) )-
							2*kappa*c*k*( ( np.sin(oma*t[-1])-np.sin((oma-c*k)*t[-1]))*np.exp(-kappa*t[-1] ) - np.sin(c*k*t[-1]) ) ) )
	print("Nk2",Nk2)
	print("phi",Phi)
	phic = np.abs(D/B)**2 * ( np.exp(-kappa*t[-1])*np.sin(oma*t[-1]) - oma/2/kappa*(1-np.exp(-2*kappa*t[-1])) )
	print("phic",phic)
	return rho_final

##################	
### PARAMETERS ###
##################
Fock   = False
show   = True

D      = .7
oma    = 10 #in GHz
kappav = np.array([0.0001])#,0.1,1])  #in GHz
gam    = 0.001 #in GHz
hbar   = 6.62607004 
kb     = 1.38064852
T      = 0.00001
nd     = 0#1./(np.exp(hbar*oma/kb/T)-1)
endt   = 16000.
Nt     = 2**18
t      = np.linspace(0,endt,Nt)
tau    = 10*Nt
endk   = 5000.
Nk     = 30000
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
	rho_wn=rho_fb(D,gam,oma,kappa,nd,endt,Nt,tau,k,therm,Fock)
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
		plt.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_kend=5000_fb_T=0_Fock1_s.png")
	else:
		plt.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_kend=5000_fb_T=%d_wide_s.png" % T)


