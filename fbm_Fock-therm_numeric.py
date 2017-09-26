#!/usr/bin/python3.4
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.integrate import quad
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
### For norm ###
def rho_nodamp_T(D, gamma, oma, t):
	return np.exp(-D**2/oma**2*(nde*(1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t))*.5*np.exp(-gamma*t)

def rho_nodamp_F(D, gamma, oma, t,NFock):
	gam2 = D**2/oma**2*2*(1-np.cos(oma*t))
	cav_coef = 0.
	for iF in range(0,NFock):
		cav_coef += (-gam2)**iF/( factorial(iF)**2*factorial(NFock-iF) )
	cav_coef = cav_coef*factorial(NFock)
	return cav_coef*np.exp(-D**2/oma**2*( (1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t) )*.5*np.exp(-gamma*t)

def rho_d(gd, gam, oma, kappa, nd, dt, Nt, k, dk, therm, Fock):

	#################
	### |N_d^k|^2 ###
	#################
	def Nd2k(k,ex,Ca,Sa,Ck,Sk,Cak,Sak):

		A2  = 2*dc**2*(1-Ck)
		B2  = dB * ( ex**2 + 1 - 2*Ca*ex )
		AB  = -2*dc*dB*( kappa* ( (Sak-Sa)*ex+Sk ) + oma* ( (Cak-Ca)*ex-Ck+1 ) )

		return coef2 * ( A2 + B2 + AB )

	def phif(k,ex,Ca,Sa,Ck,Sk,Cak,Sak):

		Cc  = kappa*(1-Cak*ex)
		t1  = -oma/(2*kappa) * dB * (1-ex**2)
		t2  = dc*dB*( (kappa*Ca-(oma-c*k)*Sa)*ex - oma*Sk - kappa*Ck )
		t3  = dB*den*( (2*oma-c*k)*Cc - (kappa**2-oma*(oma-c*k))*Sak*ex )
		t4  = dc*den*(Cc+(oma-c*k)*Sak*ex) 
		t5  = dc**2 * (Sk-c*k*t)
		return coef2 * (t1+t2+t3+t4+t5)

	nk = 1/(np.exp(therm*c*np.abs(k))-1)
	
	####################################
	### EVALUATION FOR EACH TIMESTEP ###
	####################################
	rho_final = np.zeros(Nt, complex)
	ksum_bn = np.zeros(Nt, complex)
	ksum_bc = np.zeros(Nt, complex)
	ksum_pn = np.zeros(Nt, complex)
	ksum_pc = np.zeros(Nt, complex)
	for it in range(0,Nt):

		t    = it*dt#*2*np.pi
		ex   = np.exp(-kappa*t)
		Ca   = np.cos(oma*t)
		Sa   = np.sin(oma*t)
		Ck   = np.cos(c*k*t)
		Sk   = np.sin(c*k*t)
		Cak = Ca*Ck+Sa*Sk
		Sak = Sa*Ck-Ca*Sk
		Fg   = coef * ( Sa*ex + oma/(2*kappa)*(ex**2-1) )	
		
		gamma2  = coef * ( 1 + ex**2 - 2*ex*Ca ) #|gamma|^2
		
		ksum_b = np.sum(Nd2k(k,ex,Ca,Sa,Ck,Sk,Cak,Sak)*(2*nk+1)*dk)
		ksum_bn[it] = ksum_b

		ksum_p = np.sum(phif(k,ex,Ca,Sa,Ck,Sk,Cak,Sak)*dk)
		ksum_pn[it] = ksum_p+Fg

		if Fock==True:
			Cav_coef = 0.
			for iF in range(0,NFock+1):
				Cav_coef += (-gamma2)**iF / ( factorial(iF)**2 * factorial(NFock-iF) )
			Cav_coef = factorial(NFock) * Cav_coef
#			rho_final[it] = .5 * (1+gamma2) * np.exp( - 0.5*(gamma2+ksum_b) - 1j * (ksum_p+Fg) - 1j*ome*t - gam*t)
			rho_final[it] = .5 * Cav_coef * np.exp( - 0.5*(gamma2+ksum_b) - 1j * (ksum_p+Fg) - 1j*ome*t - gam*t)
		else:
			rho_final[it] = .5 * np.exp( - 0.5*(gamma2*(2*nd+1)+ksum_b) - 1j * (ksum_p+Fg) - 1j*ome*t - gam*t)

		ksum_bc[it] = 2*gd**2/np.abs(B)**2*(2*kappa*t+(1-np.exp(-2*kappa*t))+2*kappa/np.abs(B)**2*(-2*kappa+np.conjugate(B)*np.exp(-B*t)+B*np.exp(-np.conjugate(B)*t)))
		ksum_pc[it] = gd**2/(2*kappa*np.abs(B)**4)*( oma*( np.abs(B)**2*(1-4*kappa*t-np.exp(-2*kappa*t)) +\
									8*kappa**2*(1-np.cos(oma*t)*np.exp(-kappa*t)) ) +\
								2*kappa*np.sin(oma*t)*(oma**2-3*kappa**2)*np.exp(-kappa*t) )
	return rho_final,ksum_bn-ksum_bc,ksum_pn-ksum_pc
#	return rho_final,ksum_bn-ksum_bc,ksum_bc,ksum_pn-ksum_pc,ksum_pc

##################	
### PARAMETERS ###
##################
Fock   = False
show   = False

gd     = 5.#.7
oma    = 10.#np.pi/8. #in GHz
kappav = np.array([0.1,0.2,0.5])  #in GHz
#kappav = np.array([0.001,0.1,1])  #in GHz
gam    = 0.001 #in GHz
hbar   = 6.62607004 
kb     = 1.38064852
T      = 3000.#00001
c      = 0.003
nd     = 1./(np.exp(hbar*oma/kb/T)-1)
endt   = 6000.
labet  = int(endt/1000.)
Nt     = 2**18
labNt  = int(np.log2(Nt))
t      = np.linspace(0,endt,Nt)
dt     = (endt)/(Nt-1)
endk   = 20000.
labek  = int(endk/1000.)
Nk     = 120000
labNk  = int(Nk/1000.)
ome    = 0.
k      = np.linspace(-endk,endk,Nk)# + ome*100.
dk     = k[1]-k[0]
dc     = 1/(c*k)
therm  = hbar/(kb*T)
NFock = 10


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
collab = ['green','orange','purple','yellow']
linew  = [3,2,2,4]
linest = ['-','--','-.',':']

fig,ax = plt.subplots(1,2,figsize=(25,8))
fig2,ax2 = plt.subplots(kappav.size,2,figsize=(25,30))
#fig3,ax3 = plt.subplots(kappav.size,2,figsize=(25,30))

# Determining the norm
if Fock==True:
	rho_norm = rho_nodamp_F(gd,gam,10,t,NFock)
else:
	nde    = 2./(np.exp(therm*10)-1) + 1.
	rho_norm = rho_nodamp_T(gd,gam,10,t)
norm = np.abs(np.sum(rho_norm*2*endt/(Nt)))**2

for i in range(0,kappav.size):
#for i in range(kappav.size-1,-1,-1):

	kappa=kappav[i]
	print("kappa is: ",kappav[i])
	sys.stdout.flush()	

	g02   = kappa*2*c/np.pi
	B     = 1j*oma + kappa
	den   = 1/(kappa**2+(oma-c*k)**2)
	coef2 = g02*gd**2*den
	dB    = 1/(kappa**2+oma**2)
	coef  = gd**2*dB

	#################################################
	### EVALUATION OF THE TIME-DEPENDENT SOLUTION ###
	#################################################
	rho_wn,ksum_b,ksum_p=rho_d(gd,gam,oma,kappa,nd,dt,Nt,k,dk,therm,Fock)
	evol = rho_wn/np.sqrt(norm)
	ax2[i,0].plot(t,ksum_b,color=colors[collab[i]],ls=linest[0],lw=linew[0],label="$\kappa=%.3f$" % kappa)
	ax2[i,1].plot(t,ksum_p,color=colors[collab[i]],ls=linest[0],lw=linew[0],label="$\kappa=%.3f$" % kappa)
	ax2[i,0].grid(True)
	ax2[i,1].grid(True)
	ax2[i,0].set_ylabel('$\Delta ksum_b$',fontsize=30)
	ax2[i,1].set_ylabel('$\Delta ksum_p$',fontsize=30)
	ax2[i,0].set_xlabel('$t$',fontsize=30)
	ax2[i,1].set_xlabel('$t$',fontsize=30)
	ax2[i,0].legend(fontsize=20)
	ax2[i,1].legend(fontsize=20)

#	ax3[i,0].plot(t,ksum_bc,color=colors[collab[i]],ls=linest[0],lw=linew[0],label="$\kappa=%.1f$" % kappa)
#	ax3[i,1].plot(t,ksum_pc,color=colors[collab[i]],ls=linest[0],lw=linew[0],label="$\kappa=%.1f$" % kappa)
#	ax3[i,0].grid(True)
#	ax3[i,1].grid(True)
#	ax3[i,0].set_ylabel('$ksum_b$',fontsize=30)
#	ax3[i,1].set_ylabel('$ksum_p$',fontsize=30)
#	ax3[i,0].set_xlabel('$t$',fontsize=30)
#	ax3[i,1].set_xlabel('$t$',fontsize=30)
#	ax3[i,0].legend(fontsize=20)
#	ax3[i,1].legend(fontsize=20)
	ax[0].plot(t,np.abs(rho_wn)**2,color=colors[collab[i]],ls=linest[i],lw=linew[0],label="$\kappa=%.3f$" % kappa)

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
	ax[1].plot(2*np.pi*freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[0],label="$\kappa=%.3f$" % kappa)
	ax[1].grid(True)

ax[0].grid(True)
ax[1].grid(True)
ax[0].legend(fontsize=20)
ax[1].legend(fontsize=20)
#ax[0].legend([0.001,0.01,0.1],fontsize=20)
#ax[1].legend([0.001,0.01,0.1],fontsize=20)
ax[0].set_xlabel('$t$ (10 ps)',fontsize=30)
ax[1].set_xlabel('$\omega$ (100 GHz)',fontsize=30)
ax[0].set_ylabel('$\left|P(t)\\right|^2$',fontsize=30)
ax[1].set_ylabel('$\Re{P(\omega)}$',fontsize=30)
#ax[0].set_ylim(-0.01,.25)
if T>.1:
	ax[0].set_xlim(0,500)
else:
	ax[0].set_xlim(0,500)#endt)
#ax[1].set_ylim(10**(-8),1)
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
		fig.savefig("/home/niki/Dokumente/Python/Numerical plots/Without feedback/Fock/Fock%d/oma=10/kend%de_Nk%de_endt%de_Nt2e%d_T=0_D=%dp10.png" % (NFock,labek,labNk,labet,labNt,gd*10))
		fig2.savefig("/home/niki/Dokumente/Python/Numerical plots/Without feedback/Fock/Fock%d/oma=10/check_kend%de_Nk%de_endt%de_Nt2e%d__T=0_D=%dp10.png" % (NFock,labek,labNk,labet,labNt,gd*10))
#		fig3.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_orig_kend%de_Nk%de_endt%de_Nt2e%d__T=0_Fock1_D=%dp10.png" % (labek,labNk,labet,labNt,gd*10))
	else:
		fig.savefig("/home/niki/Dokumente/Python/Numerical plots/Without feedback/Thermal/T=%d/oma=10/kend%de_Nk%de_endt%de_Nt2e%d_D=%dp10.png" % (T,labek,labNk,labet,labNt,gd*10))
		fig2.savefig("/home/niki/Dokumente/Python/Numerical plots/Without feedback/Thermal/T=%d/oma=10/check_kend%de_Nk%de_endt%de_Nt2e%d_D=%dp10.png" % (T,labek,labNk,labet,labNt,gd*10))
#		fig3.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_orig_kend%de_Nk%de_endt%de_Nt2e%d_therm_T=%d_D=%dp10.png" % (labek,labNk,labet,labNt,T,gd*10))


