import numpy as np
import matplotlib.pyplot as plt

def coeff_gen(k):
    return 0.25 if k == 0 else ((1-np.cos((np.pi*k)/2) + 1j*np.sin((np.pi*k)/2))/(1j*2*np.pi*k)) #return function from question 1.5a

def fseries(K,a,f_0,f_s,t):
	y_K = np.zeros(np.size(t), dtype=complex)
	
	#loop through, keeping index
	for index, coeff in enumerate(a):

		#translate index to k
		k = index-K #subtract k to convert e.g. at index 0, k = -K

		T_0 = 1/f_0

		#copy formula, adding to y_K
		y_K += coeff * np.exp(1j*((2*np.pi*k)/(T_0)) * t)

	return y_K

K = 100
f_0 = 100
f_s = 4000
duration = 0.05

# Generate proper time steps based on f_s
t = np.arange(0, duration, 1/f_s)

a_k = [coeff_gen(k-K) for k in range(K*2 + 1)] #list comprehensions are good

y_K = fseries(K, a_k, f_0, f_s, t)

plt.plot(t, np.real(y_K))

plt.show()