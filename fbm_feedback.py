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

################################
### TIME-DEPENDENT FUNCTIONS ###
################################

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

### For density ####
def rho_fb(Nt, tau, dt, k, nk, A, Ar, B, Br, D, Ck, kAB, Fock):

	# All time points are needed in the contributions of phi, because of the time integral,
	# and all time points are needed for the density for the Fourier Transform
	phi_env = np.zeros(Nt,dtype=np.complex128) # phi contribution coming from the bath (Gk and Nk)
	phi_cav = np.zeros(Nt,dtype=np.complex128) # phi contribution coming from the cavity (F and gamma)
	rho_fin = np.zeros(Nt,dtype=np.complex128) # final solution
	M       = 0 # number of tau intervals considered

	for it in range(0,Nt):
		if it%tau==0 and it!=0: # increase the number of tau intervals, if tau is reached before endt
	#		M = int(it/tau)
			M += 1
			now2 = time.time()
			nowh = int((now2-now)/3600.)
			nowm = int((now2-now)/60.-nowh*60)
			nows = int((now2-now)-nowh*3600-nowm*60)
			print("M increased, now it is %d at %02d:%02d:%02d" % (M,nowh, nowm, nows))
			sys.stdout.flush()	

		# It is important to reset Nk, Gk and gamma each time step
		Nk = np.zeros(k.size,dtype=np.complex128)
		Gk = np.zeros(k.size,dtype=np.complex128)
		ga = 0.+0.*1j #gamma
		for m in range(0,M+1):
			tp   = (it-m*tau)*dt # time combination needed for calculations
			expA = np.exp(A*tp)
			expB = np.exp(-B*tp)
			
			L_Gk_n = np.zeros(k.size,dtype=np.complex_) # loop with index n for Gk
			L_Nk_n = np.zeros(k.size,dtype=np.complex_) # loop with index n for Nk
			for n in range(0,m+1):
				lv = np.arange(0,n+1) # index l
				L_Nk_l = np.complex_( np.sum( tp**(n-lv) * Br**lv / factorial(n-lv) ) ) # loop with index l for Nk
				L_Nk_n += (A+B)**n * ( expB*L_Nk_l - Br**n ) # loop with index n for Nk

				L_Gk_n += (A+B)**n / factorial(n) * tp**n # loop with index n for Gk
			
			Nk += Ck * ( kAB**m * ( Ar*(expA-1) + Br * L_Nk_n ) ) # summing up Nk contributions
	
			nv = np.arange(0,m+1) # index n 
			L_ga_n = np.complex_( np.sum( tp**(m-nv) * Br**nv / factorial(m-nv) ) ) # loop with index n for gamma
			ga += -D*Br * kappa**m * ( expB*L_ga_n - Br**m ) # summing up contributions for gamma
			
			Gk += Ck * kAB**m * ( expA - expB*L_Gk_n ) # summing up contributions for Gk

		mv = np.arange(0,M+1) # index m
		F  = np.complex_( D * np.sum( kappa**mv / factorial(mv) * np.exp(-B*(it-mv*tau)*dt) * ((it-mv*tau)*dt)**mv ) )# summing up contributions for F
		
		ga2 = np.abs(ga)**2 #abs(gamma)^2

		# Determining the different coefficients and exponentials for the cavity density (originating from different types of initial excitations)
		if Fock==True:
			Cav_coef = 0. # it is important to reset this coefficient each time step
			for iF in range(0,NFock+1):
				Cav_coef += (-ga2)**iF / ( factorial(iF)**2 * factorial(NFock-iF) )
			Cav_coef = factorial(NFock) * Cav_coef
#			Cav_coef = np.complex_( 1. + np.abs(ga)**2 )
			Cav_exp  = np.complex_( -.5 * ga2 )
		else:
#			nde    = 2./(np.exp(therm*oma)-1) + 1.
			Cav_coef = np.complex_( 1.)
			Cav_exp = np.complex_( -.5 * ga2 * nde ) # thermal excitation needs the mean phonon number nde

		Bath = -.5 * np.sum( np.abs(Nk)**2 * nke * dk ) # thermal bath
		
		# phi contributions for each time step
		phi_env[it] = np.sum( Gk * np.conjugate(Nk) * dk )
		phi_cav[it] = F * np.conjugate(ga)
	
		Phi         = np.imag( np.sum( (phi_cav[0:(it+1)] + phi_env[0:(it+1)])*dt ) ) # evaluation of the time-integral

		rho_fin[it] = .5 * Cav_coef * np.exp( Cav_exp + Bath - 1j*Phi - gam*it*dt ) # final density for each time step

	return rho_fin
		

##################	
### PARAMETERS ###
##################
Fock   = False
show   = False
#Trick  = False
Tau    = False
Change = "tau" # "oma", "kappa"
plottime = 500.

D      = 1.#.7

if Change == "tau":
	oma    = 10.#np.pi/8. #in 100GHz
	tauv   = np.array([1.,12.,43.,84.])
#	tauv   = np.array([1.,1002.,2003.,4004.,10005.])
	kappa  = 0.2
#	kappa  = 0.001
elif Change == "oma":
	#omav   = np.arange(1,4)*np.pi/(2.**5.)
#	omav   = np.array([2,5])*np.pi/(2.**5.)
	omav   = (4+0.5*np.array([1,5]))*np.pi
#	omav   = np.arange(4,6)*np.pi/(2.**5.)
	kappa  = .001
elif Change == "kappa":
	kappav = np.array([0.001,0.01,0.1])  #in 100GHz
	oma = 10

ome    = 0.
gam    = 0.001 #in 100GHz
c      = 0.003

endk   = 9000
labek  = int(endk/1000.)
Numk   = 55000
labNk  = int(Numk/1000.)
k      = np.linspace(-endk,endk,Numk)# + ome*100.
dk     = k[1]-k[0]
A      = -1j*c*k
Ar     = 1/A

hbar   = 6.62607004 
kb     = 1.38064852
T      = 3.#00001
therm  = hbar/(kb*T)
nke    = 2./(np.exp(therm*c*np.abs(k))-1) + 1.
NFock  = 1

endt   = 6000.
labet  = int(endt/1000.)
Nt     = 2**18
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
collab = ['green','orange','purple',"blue","black"]
linew  = [2.5,2.5,2.5,3,5]
linest = ['-','--','-.',':',"-"]

if Change == "tau":
	row = 1#3
	leng = 8#25
else:
	row = 1
	leng = 8

fig,ax = plt.subplots(row,2,figsize=(25,leng))
#fig2,ax2 = plt.subplots(2,2,figsize=(25,18))

# Determining the norm
if Fock==True:
	rho_norm = rho_nodamp_F(D,gam,10,t,NFock)
else:
	nde    = 2./(np.exp(therm*10)-1) + 1.
	rho_norm = rho_nodamp_T(D,gam,10,t)
norm = np.abs(np.sum(rho_norm*2*endt/(Nt)))**2

if Tau==True:
	if Change == "oma":
		endloop = omav.size
	elif Change == "tau":
		endloop = tauv.size
	elif Change == "kappa":
		endloop = kappav.size
	else:
		print("What should I iterate over?")
else:
	endloop = 1

for i in range(0,endloop):

	if Change == "oma":
		oma = omav[i]
#		omlab = 8*oma/np.pi
		omlab = oma/np.pi
		print("om_a is %.1f * pi" % omlab)
	elif Change == "kappa":
		kappa=kappav[i]
		kaplab = kappa*100.
		print("kappa is: ",kappa)
#	tau    = int(kaptau/kappa/dt)
#	print("tmax-tau is: ",endt-tau*dt)

	g0  = np.sqrt(kappa*2*c/np.pi)
	if Tau==True:
		if Change == "tau":
			tau = int((100+0.05*tauv[i])*np.pi/dt)
#			tau = int(1000.*tauv[i]/dt)
#			tau = int(500./dt*tauv[i])
#			tau = int(1/(2*np.pi*tauv[i])/dt)
		else:
			tau=int(2001./dt)			
#			tau = int(Nt/3.)
		gk  = g0*np.sin(k*c*.5*tau*dt)
	else:
		tau = 2*Nt
		gk  = g0
	omt  = ( (oma*tau*dt) % (2*np.pi) ) / np.pi
	kapt = kappa*tau*dt
	taulab = tau*dt*10.
#	print(oma*tau*dt)
	sys.stdout.flush()	
	B   = 1j*oma + kappa
	Br  = 1/B
	Ck  = -1j*gk*D/(A+B)
	kAB = kappa/(A+B)
	print(omt)

	#################################################
	### EVALUATION OF THE TIME-DEPENDENT SOLUTION ###
	#################################################
	rho_wn=rho_fb(Nt,tau,dt,k,nke,A,Ar,B,Br,D,Ck,kAB,Fock)
	evol = rho_wn/np.sqrt(norm)
#	time_out = np.transpose(np.vstack((t.real,np.real(np.abs(rho_wn)**2),rho_wn)))
	time_out = np.transpose(np.vstack((t.real,rho_wn)))
	if Fock==True:
		if Tau==True:
			np.savetxt("./Data/Fock%d/time_evol_D=%dp10_tau=%d_omt=%d_kapt=%d_endk=%de_Nk=%de_endt=%de_Nt=2e%d_kap=%dp10.txt" % (NFock,D*10,tau*dt, omt,kapt,labek,labNk,labet,labNt,kappa*10),\
			time_out)
		else:
			np.savetxt("./Data/Fock%d/nofb/time_evol_D=%dp10_tau=%d_omt=%d_kapt=%d_endk=%de_Nk=%de_endt=%de_Nt=2e%d_kap=%dp10.txt" % (NFock,D*10,tau*dt, omt,kapt,labek,labNk,labet,labNt,kappa*10),\
			time_out)
	else:
		if Tau==True:
			np.savetxt("./Data/T=%d/time_evol_D=%dp10_tau=%d_omt=%d_kapt=%d_endk=%de_Nk=%de_endt=%de_Nt=2e%d_kap=%dp10.txt" % (T,D*10,tau*dt, omt,kapt,labek,labNk,labet,labNt,kappa*10),\
			time_out)
		else:
			np.savetxt("./Data/T=%d/nofb/time_evol_D=%dp10_tau=%d_omt=%d_kapt=%d_endk=%de_Nk=%de_endt=%de_Nt=2e%d_kap=%dp10.txt" % (T,D*10,tau*dt, omt,kapt,labek,labNk,labet,labNt,kappa*10),\
			time_out)
	time_out = None

	if Change == "kappa":
		ax[0].plot(t,np.abs(rho_wn)**2,color=colors[collab[i]],ls=linest[i],lw=linew[i], \
			label="$\kappa\\tau$ =%.2f$" % (kapt) )
#	elif Change =="tau":
#		sti = i%2
#		if i == 0 or i == 3:
#			ax[0,0].plot(t,np.abs(rho_wn)**2,color=colors[collab[sti]],ls=linest[sti],lw=linew[sti], \
#				label="$\omega_d\\tau$ mod$ (2\pi) =%.1f  \pi$" % (omt) )
#		elif i == 1 or i == 4:
#			ax[1,0].plot(t,np.abs(rho_wn)**2,color=colors[collab[sti]],ls=linest[sti],lw=linew[sti], \
#				label="$\omega_d\\tau$ mod$ (2\pi) =%.1f  \pi$" % (omt) )
#		else:
#			ax[2,0].plot(t,np.abs(rho_wn)**2,color=colors[collab[sti]],ls=linest[sti],lw=linew[sti], \
#				label="$\omega_d\\tau$ mod$ (2\pi) =%.1f  \pi$" % (omt) )
	elif Change =="oma" or Change=="tau":
		ax[0].plot(t,np.abs(rho_wn)**2,color=colors[collab[i]],ls=linest[i],lw=linew[i], \
			label="$\omega_d\\tau$ mod$ (2\pi) =%.1f  \pi$" % (omt) )
#	ax[0].semilogy(t,np.abs(rho_wn)**2,color=colors[collab[i]],ls=linest[i],lw=linew[0])
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
#	freq_out = np.transpose(np.vstack((freq.real,four.real,four)))
	freq_out = np.transpose(np.vstack((freq.real,four)))
	if Fock==True:
		if Tau==True:
			np.savetxt("./Data/Fock%d/FT_D=%dp10_tau=%d_omt=%d_kapt=%d_endk=%de_Nk=%de_endt=%de_Nt=2e%d_kap=%dp10.txt" % (NFock,D*10,tau*dt, omt,kapt,labek,labNk,labet,labNt,kappa*10),\
			freq_out)
		else:
			np.savetxt("./Data/Fock%d/nofb/FT_D=%dp10_tau=%d_omt=%d_kapt=%d_endk=%de_Nk=%de_endt=%de_Nt=2e%d_kap=%dp10.txt" % (NFock,D*10,tau*dt, omt,kapt,labek,labNk,labet,labNt,kappa*10),\
			freq_out)
	else:
		if Tau==True:
			np.savetxt("./Data/T=%d/FT_D=%dp10_tau=%d_omt=%d_kapt=%d_endk=%de_Nk=%de_endt=%de_Nt=2e%d_kap=%dp10.txt" % (T,D*10,tau*dt, omt,kapt,labek,labNk,labet,labNt,kappa*10),\
			freq_out)
		else:
			np.savetxt("./Data/T=%d/nofb/FT_D=%dp10_tau=%d_omt=%d_kapt=%d_endk=%de_Nk=%de_endt=%de_Nt=2e%d_kap=%dp10.txt" % (T,D*10,tau*dt, omt,kapt,labek,labNk,labet,labNt,kappa*10),\
			freq_out)
	freq_out=None

	now3 = time.time()
	nowh = int((now3-now)/3600.)
	nowm = int((now3-now)/60.-nowh*60)
	nows = int((now3-now)-nowh*3600-nowm*60)
	print("at cycle %d after the FFT: %02d:%02d:%02d"% (i, nowh, nowm, nows))
	sys.stdout.flush()	

	############
	### PLOT ###
	############
	if Change == "kappa":
		ax[1].plot(2*np.pi*freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[i],label="$\kappa=%.1f$ GHz" % kaplab )
	elif Change == "oma":
		ax[1].plot(2*np.pi*freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[i],label="$\omega_d=%d \\frac{\pi}{80}$ THz" % omlab )
	elif Change == "tau":
		ax[1].plot(2*np.pi*freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[i],label="$\kappa\\tau=%.2f$" % kapt )
#		if i == 0 or i == 3:
#			ax[0,1].semilogy(2*np.pi*freq,four.real,color=colors[collab[sti]],ls=linest[sti],lw=linew[sti],label="$\\tau=%.0f$ ns" % taulab )
#		elif i == 1 or i == 4:
#			ax[1,1].semilogy(2*np.pi*freq,four.real,color=colors[collab[sti]],ls=linest[sti],lw=linew[sti],label="$\\tau=%.0f$ ns" % taulab )
#		else:
#			ax[2,1].semilogy(2*np.pi*freq,four.real,color=colors[collab[sti]],ls=linest[sti],lw=linew[sti],label="$\\tau=%.0f$ ns" % taulab )

#if Change == "tau":
#	for rowi in range(3):
#		for coli in range(2):
#			ax[rowi,coli].grid(True)
#			ax[rowi,coli].legend(fontsize=18,loc="best")
#		ax[rowi,0].set_xlabel('$t$ (10 ps)',fontsize=30)
#		ax[rowi,1].set_xlabel('$\omega$ (100 GHz)',fontsize=30)
#		ax[rowi,0].set_ylabel('$\left|P(t)\\right|^2$',fontsize=30)
#		ax[rowi,1].set_ylabel('$\Re{P(\omega)}$',fontsize=30)
#		ax[rowi,0].set_xlim(0,plottime)
#		ax[rowi,1].set_xlim(-40,40)
#else:
for coli in range(2):
	ax[coli].grid(True)
	ax[coli].legend(fontsize=18,loc="best")
ax[0].set_xlabel('$t$ (10 ps)',fontsize=30)
ax[1].set_xlabel('$\omega$ (100 GHz)',fontsize=30)
ax[0].set_ylabel('$\left|P(t)\\right|^2$',fontsize=30)
ax[1].set_ylabel('$\Re{P(\omega)}$',fontsize=30)
ax[0].set_xlim(0,plottime)
ax[1].set_xlim(-40,40)
#ax[1].set_ylim(10**(-8),1)

##################
### TIMER ENDS ###
##################

if show==True:
	end=time.time()
	h = int((end-now)/3600.)
	m = int((end-now)/60.-h*60)
	s = int((end-now)-h*3600-m*60)
	print('%02d:%02d:%02d' %(h,m,s))

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
			if Change == "tau":
				fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Fock/Fock%d/oma=10/num_fb_D=%dp10_T=%d_omtp10=%d_ktau=%d.png" % (NFock,D*10,T,omt*10,kapt))
#				fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Fock/Fock%d/oma=pip8/num_fb_D=%dp10_T=%d_omtp10=%d_ktau=%d.png" % (NFock,D*10,T,omt*10,kapt))
			elif Change == "oma":
				fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Fock/Fock%d/tau=2001/num_fb_D=%dp10_T=%d_omtp10=%d_ktau=%d.png" % (NFock,D*10,T,omt*10,kapt))
			elif Change == "kappa":
#				fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Fock/Fock%d/oma=10/num_fb_D=%dp10_T=%d_tau=%d_omtp=%d.png" % (NFock,D*10,T,tau*dt,omt*10))
				fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Fock/Fock%d/oma=pip8/num_fb_D=%dp10_T=%d_tau=%d_omtp=%d.png" % (NFock,D*10,T,tau*dt,omt*10))
		else:
			if Change == "tau":
				fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Thermal/T=%d/oma=10/num_fb_D=%dp10_omtp10=%d_ktau=%d.png" % (T,D*10,omt*10,kapt))
#				fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Thermal/T=%d/oma=pip8/num_fb_D=%dp10_omtp10=%d_ktau=%d.png" % (T,D*10,omt*10,kapt))
			elif Change == "oma":
				fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Thermal/T=%d/tau=2001/num_fb_D=%dp10_omtp10=%d_ktau=%d.png" % (T,D*10,omt*10,kapt))
			elif Change == "kappa":
				fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Thermal/T=%d/oma=10/num_fb_D=%dp10_tau=%d_omtp%d.png" % (T,D*10,tau*dt,omt*10))
	else:
		if Fock==True:
			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Fock/Fock%d/oma=10/num_fb_D=%dp10_T=%d_notau_kap=%dp1000.png" % (NFock,D*10,T,kappa*1000))
#			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Fock/Fock%d/oma=pip8/num_fb_D=%dp10_T=%d_notau.png" % (NFock,D*10,T))
#			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Fock/Fock%d/tau=2001/num_fb_D=%dp10_T=%d_notau_om%d.png" % (NFock,D*10,T,omlab*10))
#			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_fb_T=0_notau_Fock1_log.png")
		else:
			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Thermal/T=%d/oma=10/num_fb_D=0%d_notau_kap=%dp1000.png" % (T,D*10,kappa*1000))
#			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/With feedback/Thermal/T=%d/tau=2001/num_fb_D=0%d_notau_om%d.png" % (T,D*10,omlab*10))
#			fig.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_fb_T=%d_notau_log.png" % T)
	end=time.time()
	h = int((end-now)/3600.)
	m = int((end-now)/60.-h*60)
	s = int((end-now)-h*3600-m*60)
	print('%02d:%02d:%02d' %(h,m,s))

