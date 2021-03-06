#!/usr/bin/python3.4
import matplotlib as mpl
#mpl.use('Agg')
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

now = time.time()

#def rho_d(D, gamma, oma, kappa, nd, t):
#	t    = t*2*np.pi
#	B    = 1j*oma + kappa
#	dBa2 = 1/(oma**2 + kappa**2)
#	return np.exp(- D**2 * dBa2 * ( nd*(1+np.exp(-2*kappa*t)) + .5*(3-np.exp(-2*kappa*t)) + 1j*np.sin(oma*t)*np.exp(-kappa*t) - np.exp(-kappa*t)*np.cos(oma*t)*(1+2*nd) - 1j*oma/2/kappa*(1-np.exp(-2*kappa*t)) + 2*kappa*t + 2*kappa*dBa2*(-2*kappa+ B*np.exp(-np.conjugate(B)*t) + np.conjugate(B)*np.exp(-B*t))))* np.exp(-gamma*t)*.5
#	return np.exp(-D**2*dBa2*(.5*(1-1j*oma/kappa)*(1-np.exp(-2*kappa*t))+1j*np.exp(-kappa*t)*np.sin(oma*t)-np.exp(-kappa*t)*np.cos(oma*t)-dBa2*(oma**2-3*kappa**2+2*kappa*(B*np.exp(-np.conjugate(B)*t)+np.conjugate(B)*np.exp(-B*t)))))*.5*np.exp(-gamma*t)
	#return np.exp(-D**2/oma**2*((2*nd+1)*(1-np.cos(oma*t))+1j*np.sin(oma*t)-1j*oma*t))*.5*np.exp(-gamma*t)

def rho_d(gd, gam, oma, kappa, nd, endt,Nt, k):
	dt  = (endt)/(Nt+1)
	dk  = k[1]-k[0]
	c    = 0.3
	g02  = kappa*2*c/np.pi
	B    = 1j*oma + kappa
	coef = np.abs(gd/B)**2

	def Nd2k(k,ex,Ca,Sa):
		cof = coef*g02/(c*k)**2
		den = 1/(kappa**2+(oma-c*k)**2)
		Ck  = np.cos(c*k*t)
		Sk  = np.sin(c*k*t)
		A2  = (c*k)**2 * (2*Ca-ex) * ex
		B2  = (kappa**2+oma**2) * (2*Ck-1)
		AB  = oma*(Ck+(Ca-Ca*Ck-Sk*Sa)*ex) + kappa*(-Sk+(Sa-Sa*Ck+Sk*Ca)*ex)
		return cof * ( 1 - den*(A2 + B2 - 2*c*k*AB) )
	
	rho_final = np.zeros(Nt, complex)
	for it in range(0,Nt):
		t    = it*dt*2*np.pi
		ex   = np.exp(-kappa*t)
		Ca   = np.cos(oma*t)
		Sa   = np.sin(oma*t)

		gamma2  = coef * ( 1 + ex**2 - 2*ex*Ca ) #|gamma|^2
		rho_cav = np.exp(-.5 * gamma2 * (2*nd+1))
		
		ksum = np.sum(Nd2k(k,ex,Ca,Sa)*dk)
			
		rho_bath = np.exp(-.5*(ksum))
		rho_phi  = np.exp( -1j * coef * ( ex*Sa- oma/2/kappa*(1-ex**2) ) )
		rho_final[it] = .5*rho_cav*rho_bath*rho_phi*np.exp(-gam*t)
	return rho_final
	

D      = 0.7
oma    = 10 #in GHz
kappav = np.array([0.0001,0.1,1])  #in GHz
gam    = 0.001 #in GHz
#hbar   = 6.62607004 #???* 10**(-2)
#kb     = 1.38064852
#T      = 0.00001
nd     = 1.#/(np.exp(hbar*oma/kb/T)-1)
endt   = 1000.
Nt     = 2**16
endk   = 50.
Nk     = 200
k      = np.linspace(-endk,endk,Nk)

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
linew  = [2,3,5,4]
linest = ['-','--','-.',':']

fig,ax = plt.subplots(1,2,figsize=(40,6))

for i in range(kappav.size-1,-1,-1):
	kappa=kappav[i]
	print("kappa is: ",kappav[i])
	sys.stdout.flush()	
	evol=rho_d(D,gam,oma,kappa,nd,endt,Nt,k)

	now2 = time.time()
	nowh = int((now2-now)/3600.)
	nowm = int((now2-now)/60.-nowh*60)
	nows = int((now2-now)-nowh*3600-nowm*60)
	print("at cycle %d after the integration: %02d:%02d:%02d"% (i, nowh, nowm, nows))
	sys.stdout.flush()	

#	plt.plot(t,evol.real,t,evol.imag)
#	plt.grid(True)
	fourr = np.fft.fft(evol)
	four = np.fft.fftshift(fourr)
	freqr = np.fft.fftfreq(Nt,(endt+1)/Nt)
	freq = np.fft.fftshift(freqr)

	now3 = time.time()
	nowh = int((now3-now)/3600.)
	nowm = int((now3-now)/60.-nowh*60)
	nows = int((now3-now)-nowh*3600-nowm*60)
	print("at cycle %d after the FFT: %02d:%02d:%02d"% (i, nowh, nowm, nows))
	sys.stdout.flush()	

	ax[0].semilogy(freq,four.real,color=colors[collab[i]],ls=linest[i],lw=linew[0])#,freq,four.imag)
	ax[1].semilogy(freq,four.imag,color=colors[collab[i]],ls=linest[i],lw=linew[0])#,freq,four.imag)
ax[0].grid(True)
ax[1].grid(True)
ax[0].legend([1,0.1,0.0001],fontsize=20)
ax[1].legend([1,0.1,0.0001],fontsize=20)
ax[0].set_xlabel('Frequency',fontsize=30)
ax[1].set_xlabel('Frequency',fontsize=30)
ax[0].set_ylabel('$\Re{P(\omega)}$',fontsize=30)
ax[1].set_ylabel('$\Im{P(\omega)}$',fontsize=30)
ax[0].set_ylim(10**(-4),10**4)
ax[1].set_ylim(10**(-4),10**4)
#plt.xlim(-35,35)

end=time.time()
h = int((end-now)/3600.)
m = int((end-now)/60.-h*60)
s = int((end-now)-h*3600-m*60)
print('%02d:%02d:%02d' %(h,m,s))
plt.show()
#plt.savefig("/home/niki/Dokumente/Python/Numerical plots/numeric2_100_T=0.png")


