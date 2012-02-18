# squid.py --- 
# 
# Filename: squid.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Mon Feb 13 11:35:11 2012 (+0530)
# Version: 
# Last-Updated: Sat Feb 18 17:53:15 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 666
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Rewrite of the squid demo for new moose.
# 
# 

# Change log:
# 
# 
# 

# Code:

import numpy
import pylab
import moose

GAS_CONSTANT = 8.314
FARADAY = 9.65e4
CELSIUS_TO_KELVIN = 273.15

def reversal_potential(temp, c_out, c_in):
    """Compute the reversal potential based on Nernst equation."""
    # NOTE the 70 mV added for compatibility with original HH
    return (GAS_CONSTANT * temp / FARADAY) * 1000.0 * numpy.log(c_out/c_in) + 70.0 + SquidModel.EREST_ACT

class SquidComp(moose.Compartment):
    """Compartment class enhanced with specific values set and
    calculated using dimensions."""
    def __init__(self, path):
        moose.Compartment.__init__(self, path)

    @property
    def xarea(self):
        """Area of cross section in cm^2 when length and diameter are in um"""
        return 1e-8 * numpy.pi * self.diameter * self.diameter / 4.0 # cm^2

    @property
    def area(self):
        """Area in cm^2 when length and diameter are in um"""
        return 1e-8 * self.length * numpy.pi * self.diameter # cm^2
    
    @property
    def specific_ra(self):
        return self.Ra * self.xarea / self.length
    @specific_ra.setter
    def specific_ra(self, value):
        self.Ra = value * self.length / self.xarea
        
    @property
    def specific_cm(self):
        return self.Cm / self.area
    @specific_cm.setter
    def specific_cm(self, value):
        self.Cm = value * self.area

    @property
    def specific_gl(self):
        return 1.0/(self.Rm * self.area)
    @specific_gl.setter
    def specific_gl(self, value):
        self.Rm = 1.0/(value * self.area)

    @property
    def specific_rm(self):
        return self.Rm * self.area
    @specific_rm.setter
    def specific_rm(self, value):
        self.Rm = value / self.area

class IonChannel(moose.HHChannel):
    def __init__(self, name, compartment, specific_gbar, e_rev, Xpower, Ypower=0.0, Zpower=0.0):
        moose.HHChannel.__init__(self, '%s/%s' % (compartment.path, name))
        self.Gbar = specific_gbar * compartment.area
        self.Ek = e_rev
        self.Xpower = Xpower
        self.Ypower = Ypower
        self.Zpower = Zpower
        moose.connect(self, 'channel', compartment, 'channel')
        
    def setupAlpha(self, gate, params, vdivs, vmin, vmax):
        if gate == 'X' and self.Xpower > 0:
            gate = moose.HHGate(self.path + '/gateX')
        elif gate == 'Y' and self.Ypower > 0:
            gate = moose.HHGate(self.path + '/gateY')
        else:
            return False
        gate.setupAlpha([params['A_A'],
                         params['A_B'],
                         params['A_C'],
                         params['A_D'],
                         params['A_F'],
                         params['B_A'],
                         params['B_B'],
                         params['B_C'],
                         params['B_D'],
                         params['B_F'],
                         vdivs, vmin, vmax])        
        return True
    
    @property
    def alpha_m(self):
        if self.Xpower == 0:
            return numpy.array([])        
        return numpy.array(moose.HHGate('%s/gateX' % (self.path)).tableA)
    @property
    def beta_m(self):
        if self.Xpower == 0:
            return numpy.array([])        
        return numpy.array(moose.HHGate('%s/gateX' % (self.path)).tableB) - numpy.array(moose.HHGate('%s/gateX' % (self.path)).tableA)
    @property
    def alpha_h(self):
        if self.Ypower == 0:
            return numpy.array([])        
        return numpy.array(moose.HHGate('%s/gateY' % (self.path)).tableA)
    @property
    def beta_h(self):
        if self.Ypower == 0:
            return numpy.array([])        
        return numpy.array(moose.HHGate('%s/gateY' % (self.path)).tableB) - numpy.array(moose.HHGate('%s/gateY' % (self.path)).tableA)
    
        
class SquidModel(moose.Neutral):
    EREST_ACT = 0.0 # can be -70 mV if not following original HH convention
    VMIN = -30.0
    VMAX = 120.0
    VDIVS = 150
    Na_m_params = {'A_A':0.1 * (25.0 + EREST_ACT),
                   'A_B': -0.1,
                   'A_C': -1.0,
                   'A_D': -25.0 - EREST_ACT,
                   'A_F':-10.0,
                   'B_A': 4.0,
                   'B_B': 0.0,
                   'B_C': 0.0,
                   'B_D': 0.0 - EREST_ACT,
                   'B_F': 18.0}
    Na_h_params = {'A_A': 0.07,
                   'A_B': 0.0,
                   'A_C': 0.0,
                   'A_D': 0.0 - EREST_ACT,
                   'A_F': 20.0,
                   'B_A': 1.0,
                   'B_B': 0.0,
                   'B_C': 1.0,
                   'B_D': -30.0 - EREST_ACT,
                   'B_F': -10.0}
    K_n_params = {'A_A': 0.01*(10.0 + EREST_ACT),
                  'A_B': -0.01,
                  'A_C': -1.0,
                  'A_D': -10.0 - EREST_ACT,
                  'A_F': -10.0,
                  'B_A': 0.125,
                  'B_B': 0.0,
                  'B_C': 0.0,
                  'B_D': 0.0 - EREST_ACT,
                  'B_F': 80.0}
    
    def __init__(self, path):
        moose.Neutral.__init__(self, path)
        self.temperature = CELSIUS_TO_KELVIN + 6.3        
        self.K_out = 10.0
        self.Na_out = 460.0
        # Modified internal concentrations used to give HH values of
        # equilibrium constants from the Nernst equation at 6.3 deg C.
        # HH 1952a, p. 455        
        self.K_in = 301.4
        self.Na_in = 70.96
        self.Cl_out = 540.0        
        self.Cl_in = 100.0

        self.specific_gNa = 120.0 # mmho/cm^2
        self.specific_gK = 36.0 # mmho/cm^2

        self.squid_axon = SquidComp('%s/squid' % (self.path))
        self.squid_axon.length = 500 # um
        self.squid_axon.diameter = 500 # um
        self.squid_axon.Em = SquidModel.EREST_ACT + 10.613
        self.squid_axon.initVm = SquidModel.EREST_ACT        
        self.squid_axon.specific_cm = 1.0 # uF/cm^2
        self.squid_axon.specific_gl =  0.3 # mmho/cm^2
        self.squid_axon.specific_ra = 0.030 # kohm-cm
        
        self.Na_channel = IonChannel('Na', self.squid_axon,
                                     self.specific_gNa,
                                     self.VNa,
                                     Xpower=3.0,
                                     Ypower=1.0)
        self.Na_channel.setupAlpha('X', SquidModel.Na_m_params,
                                   SquidModel.VDIVS,
                                   SquidModel.VMIN,
                                   SquidModel.VMAX)
        self.Na_channel.setupAlpha('Y', SquidModel.Na_h_params,
                                   SquidModel.VDIVS,
                                   SquidModel.VMIN,
                                   SquidModel.VMAX)
        self.K_channel = IonChannel('K', self.squid_axon,
                                    self.specific_gK,
                                    self.VK,
                                    Xpower=4.0)
        self.K_channel.setupAlpha('X', SquidModel.K_n_params,
                                  SquidModel.VDIVS,
                                  SquidModel.VMIN,
                                  SquidModel.VMAX)
        self.inject_delay = 5.0 # ms
        self.inject_dur = 40 # ms
        self.inject_amp = 0.1 # uA
        self.Vm_table = moose.Table('%s/Vm' % (self.path))
        moose.connect(self.Vm_table, 'requestData', self.squid_axon, 'get_Vm')
        if hasattr(self, 'K_channel'):
            self.gK_table = moose.Table('%s/gK' % (self.path))
            moose.connect(self.gK_table, 'requestData', self.K_channel, 'get_Gk')
        if hasattr(self, 'Na_channel'):
            self.gNa_table = moose.Table('%s/gNa' % (self.path))
            moose.connect(self.gNa_table, 'requestData', self.Na_channel, 'get_Gk')
        
    @property
    def VK(self):
        """Reversal potential of K+ channels"""
        return reversal_potential(self.temperature, self.K_out, self.K_in)
    
    @property
    def VNa(self):
        """Reversal potential of Na+ channels"""
        return reversal_potential(self.temperature, self.Na_out, self.Na_in)
    
    def run(self, runtime, simdt=1e-6):
        moose.setClock(0, simdt)
        moose.setClock(1, simdt)
        moose.setClock(2, simdt)
        moose.setClock(3, simdt)
        moose.useClock(0, '%s/#[TYPE=Compartment]' % (self.path), 'init')
        moose.useClock(1, '%s/#[TYPE=Compartment]' % (self.path), 'process')
        moose.useClock(2, '%s/#[TYPE=HHChannel]' % (self.squid_axon.path), 'process')
        moose.useClock(3, '%s/#[TYPE=Table]' % (self.path), 'process')
        moose.reinit()
        if (self.inject_delay > 0):
            moose.start(self.inject_delay)
        self.squid_axon.inject = self.inject_amp
        if (self.inject_dur > 0):
            moose.start(self.inject_dur)
        rest = runtime - self.inject_delay - self.inject_dur
        self.squid_axon.inject = 0.0
        if rest > 0:
            moose.start(rest)

    def save_data(self):
        self.Vm_table.xplot('Vm.dat', 'Vm')
        print 'Vm saved to Vm.dat'
        if hasattr(self, 'gK_table'):
            self.gK_table.xplot('gK.dat', 'gK')
            numpy.savetxt('K_alpha_n.dat', self.K_channel.alpha_m)
            numpy.savetxt('K_beta_n.dat', self.K_channel.beta_m)
            print 'K conductance saved to gK.dat'
        if hasattr(self, 'gNa_table'):
            self.gNa_table.xplot('gNa.dat', 'gNa')
            numpy.savetxt('Na_alpha_m.dat', self.Na_channel.alpha_m)
            numpy.savetxt('Na_beta_m.dat', self.Na_channel.beta_m)
            numpy.savetxt('Na_alpha_h.dat', self.Na_channel.alpha_h)
            numpy.savetxt('Na_beta_h.dat', self.Na_channel.beta_h)
            print 'Na conductance saved to gNa.dat'

import unittest

class SquidAxonTest(unittest.TestCase):
    def setUp(self):
        self.vrange = numpy.linspace(SquidModel.VMIN, SquidModel.VMAX, SquidModel.VDIVS+1)        
        self.model = SquidModel('testSquidAxon')
        
    def calc_alpha_beta(self, params, table='A'):        
        denominator = params[table+'_C'] + numpy.exp((self.vrange+params[table+'_D'])/params[table+'_F'])
        numerator = params[table+'_A'] + params[table+'_B'] * self.vrange
        y = numpy.zeros(len(self.vrange))
        singularities = numpy.nonzero(denominator == 0.0)[0]
        self.assertLessEqual(len(singularities), 1)
        if len(singularities) == 1:
            y[:singularities[0]] = numerator[:singularities[0]]/denominator[:singularities[0]]
            y[singularities[0]] = params[table+'_B'] * params[table+'_F']
            y[singularities[0]+1:] = numerator[singularities[0]+1:]/denominator[singularities[0]+1:]
        elif len(singularities) == 0:
            y[:] = numerator[:]/denominator[:]
        return y
        
    def test_Na_alpha_m(self):
        alpha_m = self.calc_alpha_beta(SquidModel.Na_m_params, 'A')
        difference = numpy.sqrt(numpy.mean((alpha_m - self.model.Na_channel.alpha_m)**2))
        pylab.title('Na_alpha_m')        
        pylab.plot(alpha_m, label='python')
        pylab.plot(self.model.Na_channel.alpha_m, label='moose')
        pylab.legend()
        pylab.show()
        self.assertLessEqual(difference, numpy.mean(alpha_m)*1e-6)
        
    def test_Na_beta_m(self):
        beta_m = self.calc_alpha_beta(SquidModel.Na_m_params, 'B')
        difference = numpy.sqrt(numpy.mean((beta_m - self.model.Na_channel.beta_m)**2))
        pylab.title('Na_beta_m')
        pylab.plot(beta_m, label='python')
        pylab.plot(self.model.Na_channel.beta_m, label='moose')
        pylab.legend()
        pylab.show()
        self.assertLessEqual(difference, numpy.mean(beta_m)*1e-6)

    def test_Na_alpha_h(self):
        alpha_h = self.calc_alpha_beta(SquidModel.Na_h_params, 'A')
        difference = numpy.sqrt(numpy.mean((alpha_h - self.model.Na_channel.alpha_h)**2))
        pylab.title('Na_alpha_h')
        pylab.plot(alpha_h, label='python')
        pylab.plot(self.model.Na_channel.alpha_h, label='moose')
        pylab.legend()
        pylab.show()
        self.assertLessEqual(difference, numpy.mean(alpha_h)*1e-6)

    def test_Na_beta_h(self):
        beta_h = self.calc_alpha_beta(SquidModel.Na_h_params, 'B')
        difference = numpy.sqrt(numpy.mean((beta_h - self.model.Na_channel.beta_h)**2))
        pylab.title('Na_beta_h')
        pylab.plot(beta_h, label='python')
        pylab.plot(self.model.Na_channel.beta_h, label='moose')
        pylab.legend()
        pylab.show()
        self.assertLessEqual(difference, numpy.mean(beta_h)*1e-6)

    def test_K_alpha_m(self):
        alpha_m = self.calc_alpha_beta(SquidModel.K_n_params, 'A')
        difference = numpy.sqrt(numpy.mean((alpha_m - self.model.K_channel.alpha_m)**2))
        pylab.title('K_alpha_n')
        pylab.plot(alpha_m, label='python')
        pylab.plot(self.model.K_channel.alpha_m, label='moose')
        pylab.legend()
        pylab.show()
        self.assertLessEqual(difference, numpy.mean(alpha_m)*1e-6)

    def test_K_beta_m(self):
        beta_m = self.calc_alpha_beta(SquidModel.K_n_params, 'B')
        difference = numpy.sqrt(numpy.mean((beta_m - self.model.K_channel.beta_m)**2))
        pylab.title('K_beta_n')
        pylab.plot(beta_m, label='python')
        pylab.plot(self.model.K_channel.beta_m, label='moose')
        pylab.legend()
        pylab.show()
        self.assertLessEqual(difference, numpy.mean(beta_m)*1e-6)
        
def test(runtime=100.0, simdt=1e-2):
    model = SquidModel('model')
    model.run(runtime, simdt)
    model.save_data()

if __name__ == '__main__':
    # unittest.main()
    test()
    
                                                                             
    
# 
# squid.py ends here
