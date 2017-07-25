import numpy as np
import matplotlib.pyplot as plt

time=np.linspace(0,1000,10001)
sol=np.fft.fft(np.sin(time*2*np.pi))
print(time.shape[-1],time[1]-time[0])
freq=np.fft.fftfreq(time.shape[-1],time[1]-time[0])
plt.plot(freq,sol.real,freq,sol.imag)
plt.show()
