import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


T_s = 1
A = np.sqrt(3/T_s)
f = np.linspace(-20, 20, 10000)

P_f = ((A*T_s)/2) * np.pow(np.sinc(f * (T_s/2)),2)*np.exp(-1j*np.pi*f*T_s)

P_f_p = P_f * np.conj(P_f)

def integrand(f):
    return ((A*T_s)/2) * np.pow(np.sinc(f * (T_s/2)),2) * ((A*T_s)/2) * np.pow(np.sinc(f * (T_s/2)),2)

#compute integral
width = 5
slices = 1e4
target_percent = 0.90

outarr = [scipy.integrate.quad(integrand, -lim * width/slices, lim * width/slices)[0] for lim in range(int(slices+1))]

freqs = [lim*width/slices for lim in range(int(slices+1))]

diff = [abs(val - target_percent) for val in outarr]

# print frequency width for percentage in diff
print(freqs[np.argmin(diff)])

#plot to estimate 60db line
P_f_db = (10 * np.log10(np.real(P_f_p)/np.max(P_f_p)))

plt.plot(f, P_f_db)
plt.ylim((-60, 0))
plt.show()

