import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Define the symbol_mod module

#The symbol_mod module takes the following arguments as inputs:

# packet_bits                 The bit array to be mapped into symbols (including both the preamble bits and the
#                             payload bits)

# scheme                      A string indicating which scheme is adopted (e.g.: "OOK", "QPSK")

# preamble_length             Length of the preamble (in bits)

#The symbol_mod function returns the following argument as output:

# baseband_symbols:           The baseband symbols obtained after mapping the bits


def symbol_mod(packet_bits, scheme, preamble_length): 


        if(scheme == 'OOK'):

                preamble = packet_bits[0:preamble_length]
                payload = packet_bits[preamble_length:len(packet_bits)]
                preamble_symbols = 1.0*preamble
                payload_symbols = 1.0*payload                
                baseband_symbols = np.append(preamble_symbols,payload_symbols)

        return baseband_symbols
        
        
