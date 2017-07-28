import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

a = np.linspace(0,5,11)
def f(x,a):
	return x/a
print(f(3,a))

integ = np.zeros(a.size)
integerr = np.zeros(a.size)
for ia in range(0, a.size):
	parval    = a[ia]
	integ[ia],integerr[ia] = .5*integrate.quad(f,-2,4,args=(parval,))

plt.plot(a,integ)
plt.show()

