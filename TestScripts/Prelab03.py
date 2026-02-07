import matplotlib.pyplot as plt
import numpy as np


#fourier series math from 3.2

def coeff_gen(k):
    return 0.25 if k == 0 else ((1-np.cos((np.pi*k)/2) + 1j*np.sin((np.pi*k)/2))/(1j*2*np.pi*k)) #return function from question 1.5a

def coeff_cos(k):
    if (k == 1) or (k == -1):
          return 0.5
    else:
        return 0

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

# parameters
K = 4000   #number of frequency bins
f_01 = 100e3    #carrier
f_02 = 5e3      #signal
f_s = 400e3    #sampling frequency
duration = 0.0005 #length in time

t = np.arange(0, duration, 1/f_s)

# generate coefficients
a_k1 = [coeff_cos(k-K) for k in range(K*2 + 1)]
a_k2 = [coeff_cos(k-K) for k in range(K*2 + 1)]

# generate time domain signals
c_t = fseries(K, a_k1, f_01, f_s, t)
m_t = fseries(K, a_k2, f_02, f_s, t)

# combine carrier and signal

s_t = m_t*c_t

#get frequency domain of these signals

f = np.fft.fftshift(np.fft.fftfreq(K)) * f_s #frequency spectrum
m_f = np.fft.fftshift(np.fft.fft(m_t, K))
c_f = np.fft.fftshift(np.fft.fft(c_t, K))
s_f = np.fft.fftshift(np.fft.fft(s_t, K))



#plot

fig, axs = plt.subplots(2)
fig.suptitle('Frequency and Time domain of AM')
axs[0].plot(f, m_f, label="Modulator")
axs[0].plot(f, c_f, label="carrier")
axs[0].plot(f, s_f, label="Product")
axs[0].set(xlabel="Frequency", ylabel="Magnitude")
axs[0].legend()
axs[1].plot(t, m_t, label="Modulator")
axs[1].plot(t, s_t, label= "product")
axs[1].set(xlabel="Time", ylabel="Magnitude")
axs[1].legend()

plt.show()