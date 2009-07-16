#!/usr/bin/env python

from numpy import *
from numpy.fft import *
from pylab import *

import model

def fft_squid(path):
    squid_model = model.NoisySquid(path)
    squid_model.getContext().setClock(3, 20e-3, 0)
    squid_model.currentInjection.min = 0.0e-6
    squid_model.currentInjection.max = 0.1e-6
    squid_model.getContext().reset()
    squid_model.run(200e-3)
    signal = array(squid_model.vmTable)
    squid_model.save_all_plots()
#    savetxt('vm.plot', signal)
    transform = fft(signal)
    xx = array(range(len(transform)))
    ax1 = subplot(121)
    #plot(xx / 10.0 , signal * 1e3, 'r')
    savetxt("t_series.plot", xx/10.0)
#     ax1 = axis([0, 0, 0.5, 0.8])
#     xlabel("time (ms)")
#     ylabel("membrane potential (mV)")
#     ax2 = twinx()
#     plot(xx / 10.0 , array(squid_model.iInjectTable) * 1e9, 'b--')
#     ylabel("injection current (nA)")
#     ax2.yaxis.tick_right()
#     title("(A)", fontname="Times New Roman", fontsize=10, fontweight="bold")
#     show()
#     savefig("Vm.jpg", dpi=600)
    
    savetxt('fftreal.plot', transform.real, fmt="%13.12G")
    savetxt('fftimag.plot', transform.imag, fmt="%13.12G")
#     figure(2)
#     title("(B)", fontname="Times New Roman", fontsize=10, fontweight="bold")
#     subplot(122)
#     ax3 = axes([0.65, 0.0, 0.35, 0.8])
#     plot(xx[1:] / 10.0, (abs(transform)**2)[1:])
#     ylabel("power spectrum (arbitrary unit)",  fontname="Times New Roman", fontsize=10, fontweight="normal")
#     savefig("fft.jpg", dpi=900)
#     show()

if __name__ == "__main__":
    fft_squid("/squid_fft")
