# Run this cell to generate convert and save the script into a Python Program.

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define the symbol_mod module

## The symbol_mod module takes the following arguments as inputs:

### packet_bits                 The bit array to be mapped into symbols (including both the preamble bits and the
#                             payload bits)

### scheme                      A string indicating which scheme is adopted (e.g.: "OOK", "QPSK")

### preamble_length             Length of the preamble (in bits)

## The symbol_mod function returns the following argument as output:

### baseband_symbols:           The baseband symbols obtained after mapping the bits

def symbol_mod(packet_bits, scheme, preamble_length):

        if(scheme == 'OOK'):

                preamble = packet_bits[0:preamble_length]
                payload = packet_bits[preamble_length:len(packet_bits)]
                preamble_symbols = 1.0*preamble
                payload_symbols = 1.0*payload
                baseband_symbols = np.append(preamble_symbols,payload_symbols)

        if(scheme == 'QPSK'):

                preamble = packet_bits[0:preamble_length]
                payload = packet_bits[preamble_length:len(packet_bits)]

# Setting the Preamble bits
                baseband_symbols_I = 1.0*preamble
                baseband_symbols_Q = np.zeros(preamble_length)

# Imagine the two message bits arriving as being independently generated,\
# now split the payload into two to generate two message streams for I and Q channels.
                I_bits = np.array([idx for n,idx in enumerate(payload) if n % 2 == 0]) #payload[::2]
                Q_bits = np.array([idx for n,idx in enumerate(payload) if n % 2 != 0]) 
# Now modulate the two streams separately according to the BPSK modulation scheme,\
# do not forget to scale the symbols appropriately to maintain the symbol energy.
                I_symbols = (1/(np.sqrt(2))) * (I_bits * 2 - 1)
                Q_symbols = (1/(np.sqrt(2))) * (Q_bits * 2 - 1)

# Setting the Preamble Symbols
                preamble_symbols = baseband_symbols_I + 1j*baseband_symbols_Q
# Similarly set the data symbols
                data_symbols = I_symbols + 1j*Q_symbols

# Scale QPSK payload to have the same per channel average signal energy as OOK
                data_symbols = data_symbols

# Add the preamble to the payload to create a packet
                baseband_symbols = np.append(preamble_symbols, data_symbols)

        return baseband_symbols
