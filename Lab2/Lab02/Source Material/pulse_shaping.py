import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Define the pulse shaping module (Rectangular or Root-raised-cosine filters are used in the Tx and Rx side).
#(split the filters to TX & RX side so the receiver filter could be used to attenuate out-of-band channel noise)
#Source code for the commpy filters invoked here could be found in the following link:
#https://pydoc.net/scikit-commpy/0.3.0/commpy.filters/


#The pulse shaping module takes the following arguments as inputs:

# a:                        Base band symbols (containing both the preamble and payload). 

# M:                        This is the transmitter-side oversampling factor, each symbol (bit) being transmitted is represented
#                           by M samples

# fs:                       This is the sampling rate of the DAC

# pulse_shape:              "rect" (rectangular) or "rrc" (root-raised cosine)

# alpha:                    This is the roll-off factor for RRC (valid value [0,1])

# L:                        This is the length (span) of the pulse-shaping filter (in the unit of symbols)

#The pulse shaping module returns the following argument as output:

#baseband                   This is the baseband signal

#filter length: make it a parameter w/ default value
#TODO: introduce memory and make pulse shaping streaming
def pulse_shaping(a, M, fs, pulse_shape, alpha, L):
        
        #Upsample by a factor of M
        y = signal.upfirdn([1],a,M)

        if(pulse_shape == 'rect'):

                Ts = 1/fs

                #rectangular pulse
                h = np.ones(M)

        baseband = np.convolve(y,h)


        return baseband

        




