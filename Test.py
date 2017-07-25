import numpy as np
import matplotlib.pyplot as plt
def function(a,b):
	return a+b
x=np.linspace(0,10,100)
plt.plot(x,function(5,x))
plt.show()
