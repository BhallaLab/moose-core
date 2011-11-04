# compartment.py --- 
# 
# Filename: compartment.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Apr 24 10:01:45 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 16:07:30 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 177
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 

# Code:

from math import pi

from cStringIO import StringIO
import moose
import config
from channel import ChannelBase
from cachans import *
from kchans import *
from nachans import *
from capool import *
from archan import AR

class MyCompartment(moose.Compartment):
    def __init__(self, *args):
        moose.Compartment.__init__(self, *args)
        self.channels = []
        self._xarea = None
        self._sarea = None
        self.raxial_list = []

    def setSpecificRm(self, RM):
        self.Rm = RM / self.sarea()

    def getSpecificRm(self):
        return self.Rm * self.sarea()

    def setSpecificRa(self, RA):
        self.Ra = RA * self.length / self.xarea()
        config.LOGGER.debug('%s %g %g %g' % (self.name, self.Ra, self.length, self.xarea()))

    def getSpecificRa(self):
        return self.Ra * self.xarea() / self.length

    def setSpecificCm(self, CM):
        self.Cm = CM * self.sarea()

    def getSpecificCm(self):
        return self.Cm / self.sarea()

    def xarea(self):
        if self._xarea is None:
            self._xarea = pi * self.diameter * self.diameter / 4.0
        return self._xarea

    def sarea(self):
        if self._sarea is None:
            self._sarea = pi * self.length * self.diameter
        return self._sarea

    def insertChannel(self, channel, specificGbar=None, Ek=None, shift=None):
        """Insert a channel setting its gbar as membrane_area *
        specificGbar and reversal potential to Ek.
        
        This method expects either a valid channel class name or an
        existing channel object. If specificGbar is given, the Gbar is
        set to specificGbar * surface-area of the compartment. If Ek
        is given, the channel's Ek is set to this value.
        """
        if type(channel) is type(''): # if it is a class name, create the channel as a child with the same name as the class name            
            chan_class = eval(channel)
            if shift:
                chan = chan_class(channel, self, shift=shift)
            else:
                chan = chan_class(channel, self)
        elif isinstance(channel, moose.HHChannel):
            chan = channel
        else:
            print "ERROR: unknown object passed as channel: ", channel
        if specificGbar is not None:
            chan.Gbar = specificGbar * self.sarea()
        if Ek is not None:
            chan.Ek = Ek
        self.channels.append(chan)
        self.connect("channel", chan, "channel")
        return chan

    def insertCaPool(self, phi, tau):
        """Insert a Ca+2 pool and connect it to the relevant channels.

        phi is the amount of Ca2+ in unit area. 

        NOTE that this function should be called only after all
        channels (Ca and Ca dependent K channels) have been
        initialized. You can call this function multiple times without
        harm, there is safeguard against multiple connections in
        CaPool class."""
        self.ca_pool = CaPool('CaPool', self)
        if phi > 0.0:
            self.ca_pool.B = phi / self.sarea()
        else:
            self.ca_pool.B = -phi
        self.ca_pool.tau = tau
        ca_channels = [ channel for channel in self.channels \
                            if isinstance(channel, CaL) or channel.name.startswith('CaL')]
        self.ca_pool.connectCaChannels(ca_channels)
        kca_channels = [ channel for channel in self.channels \
                            if isinstance(channel, KCaChannel) or channel.name.startswith('KC') or channel.name.startswith('KAHP') ]
        self.ca_pool.connectDepChannels(kca_channels)

    def insertRecorder(self, object_name, field_name, data_container): 
        """Creates a table for recording a field under data_container"""
        table = moose.Table(object_name, data_container)# Possible name conflict for multiple recorders on different compartments
        table.stepMode = 3
        self.connect(field_name, table, "inputRequest")
        return table

    def insertCaRecorder(self, object_name, data_container):
        """Creates a table for recording [Ca2+] under data_container"""
        ca_table = None
        ca_table_path = data_container.path + '/' + object_name
        if config.context.exists(ca_table_path):
            return moose.Table(ca_table_path)
        ca_conc_path = self.path + '/CaPool'
        if config.context.exists(ca_conc_path):
            ca_conc = moose.CaConc(ca_conc_path)
            ca_table = moose.Table(ca_table_path)
            ca_table.stepMode = 3
            if not ca_conc.connect('Ca', ca_table, 'inputRequest'):
                print 'Error connecting [Ca2+] on', ca_conc.path, 'to', ca_table.path

        return ca_table


    def insertPulseGen(self, name, parent,      \
                           baseLevel=0.0,       \
                           firstLevel=1e-10,    \
                           firstDelay=20e-3,    \
                           firstWidth=20e-3,    \
                           secondLevel=0.0,     \
                           secondDelay=1e10,    \
                           secondWidth=0.0):
        self.pulsegen = moose.PulseGen(name, parent)
        self.pulsegen.baseLevel = baseLevel
        self.pulsegen.firstLevel = firstLevel
        self.pulsegen.firstDelay = firstDelay
        self.pulsegen.firstWidth = firstWidth
        self.pulsegen.secondLevel = secondLevel
        self.pulsegen.secondDelay = secondDelay
        self.pulsegen.secondWidth = secondWidth
        self.pulsegen.connect('outputSrc', self, 'injectMsg')
        return self.pulsegen

    def makeSynapse(self, target, 
                    classname='SynChan', 
                    name='synapse', 
                    threshold=0.0, 
                    absRefract=0.0, 
                    Ek=0.0, 
                    Gbar=None, 
                    tau1=None, 
                    tau2=None,
                    weight=1.0,
                    delay=0.0,
                    Pr=1.0):
        """Make a synaptic connection from this compartment to target
        compartment and set the properties of the synaptic connection
        as specified in the parametrs.

        For NMDAChan, [Mg2+] has to be set separately. Also note that
        the weight and delay vectors are not available until reset is
        called. So these must be assigned afterwards.

        """
        classobj = eval('moose.' + classname)
        synapse = classobj(name, target)
        synapse.Ek = Ek # TODO set value according to original model
        synapse.Gbar = Gbar # TODO set value according to original model
        synapse.tau1 = tau1
        synapse.tau2 = tau2
        target.connect('channel', synapse, 'channel')
        spikegen = None
        spikegen = moose.SpikeGen('%s/spike' % (self.path))
        spikegen.threshold = threshold
        spikegen.absRefract = absRefract
        self.connect('VmSrc', spikegen, 'Vm')
        if not spikegen.connect('event', synapse, 'synapse'):
            raise Exception('Error creating connection: %s->%s' % (spikegen.path, synapse.path))
        # This is too much of log info. Hence commenting out.
        # else:
        #     config.LOGGER.debug('Connected %s->%s' % (spikegen.path, synapse.path))

        # We had an awkward situation here: the weight and delay
        # vectors were not updated until reset/setDelay/setWeight was
        # called.  So we had to use num_synapse (which should
        # actually have been incremented by 1 due to the connection)
        # instead of (num_synapse - 1).
        # 2010-03-29 After making a mandatory call to updateNumSynapse()
        # in getNumSynapses(), this is fixed. 
        num_synapses = synapse.numSynapses
        synapse.delay[num_synapses - 1] = delay
        synapse.weight[num_synapses - 1] = weight
        if config.stochastic:
            synapse.initPr[num_synapses - 1] = Pr
        return synapse
        

    def get_props(self):
        """Returns information about the compartment as a string
        similar to a line in genesis .p file"""
        s = StringIO()
        s.write(self.name)
        parent = moose.Neutral(self.parent)
        s.write(' ' + parent.name)
        s.write(' ' + str(self.length / 1e-6))
        s.write(' ' + str(self.diameter / 2e-6))
        s.write(' Em ' + str(self.Em))
        s.write(' CM ' + str(self.Cm / self.sarea()))
        s.write(' GM ' + str(1.0/(self.sarea() * self.Rm)))
        s.write(' RA ' + str(self.Ra * self.xarea() / self.length))
        for channel in self.channels:
            s.write(' ' + channel.name + ' ' + str(channel.Gbar / self.sarea()))
        if hasattr(self, 'ca_pool'):
            s.write(' caconc ' + str(self.ca_pool.tau))
        return s.getvalue()


def compare_compartment(left, right):
    """Compare if two compartments have same field values"""
#     return almost_equal(left.Em, right.Em) and \
#         almost_equal(left.Rm, right.Rm) and \
#         almost_equal(left.Cm, right.Cm) and \
#         almost_equal(left.Ra, right.Ra) and \
#         almost_equal(left.initVm, right.initVm)
    result = almost_equal(left.Em, right.Em)
    if not result:
        print left.path + ".Em = " + str(left.Em) + " <> " + right.path + ".Em = " + str(right.Em)
        return result
    result = almost_equal(left.Rm, right.Rm)
    if not result:
        print(left.path + ".Rm = " + str(left.Rm) + " <> " + right.path + ".Rm = " + str(right.Rm))
        return result

    result = almost_equal(left.Cm, right.Cm)
    if not result:
        print(left.path + ".Cm = " + str(left.Cm) + " <> " + right.path + ".Cm = " + str(right.Cm))
        return result
    result = almost_equal(left.Ra, right.Ra)
    if not result:
        print(left.path + ".Ra = " + str(left.Ra) + " <> " + right.path + ".Ra = " + str(right.Ra))
        return result

    result = almost_equal(left.initVm, right.initVm)
    if not result:
        print(left.path + ".initVm = " + str(left.initVm) + " <> " + right.path + ".initVm = " + str(right.initVm))
        return result

    result = almost_equal(left.Vm, right.Vm)
    if not result:
        print(left.path + ".Vm = " + str(left.Vm) + " <> " + right.path + ".Vm = " + str(right.Vm))
        return result

    return True
# ! compare_compartments

from trbutil import almost_equal
def compare_channel(left, right):
    """Compare two channels on same field values"""
#     result = almost_equal(left.Ek, right.Ek) and \
#         almost_equal(left.Gbar, right.Gbar) and \
#         almost_equal(left.Xpower, right.Xpower) and \
#         almost_equal(left.Ypower, right.Ypower) and \
#         almost_equal(left.Zpower, right.Zpower) and \
#         almost_equal(left.instant, right.instant) and \
#         almost_equal(left.Gk, right.Gk) and \
#         almost_equal(left.Ik, right.Ik)
    result = almost_equal(left.Ek, right.Ek)
    if not result:
        print(left.path + ".Ek = " + str(left.Ek) + " <> " + right.path + ".Ek = " + str(right.Ek))
        return result
    
    result = almost_equal(left.Gbar, right.Gbar)
    if not result:
        print(left.path + ".Gbar = " + str(left.Gbar) + " <> " + right.path + ".Gbar = " + str(right.Gbar))
        return result

    result = almost_equal(left.Xpower, right.Xpower)

    if not result:
        print(left.path + ".Xpower = " + str(left.Xpower) + " <> " + right.path + ".Xpower = " + str(right.Xpower))
        return result

    result = almost_equal(left.Ypower, right.Ypower)
    if not result:
        print(left.path + ".Ypower = " + str(left.Ypower) + " <> " + right.path + ".Ypower = " + str(right.Ypower))
        return result

    result = almost_equal(left.Zpower, right.Zpower)
    if not result:
        print(left.path + ".Zpower = " + str(left.Zpower) + " <> " + right.path + ".Zpower = " + str(right.Zpower))
        return result

    result = almost_equal(left.instant, right.instant)
    if not result:
        print(left.path + ".instant = " + str(left.instant) + " <> " + right.path + ".instant = " + str(right.instant))
        return result

    result = almost_equal(left.Gk, right.Gk)
    if not result:
        print(left.path + ".Gk = " + str(left.Gk) + " <> " + right.path + ".Gk = " + str(right.Gk))
        return result

    result = almost_equal(left.Ik, right.Ik)
    if not result:
        print(left.path + ".Ik = " + str(left.Ik) + " <> " + right.path + ".Ik = " + str(right.Ik))
        return result


    return True


def has_cycle(comp):
    comp._visited = True
    ret = False
    for item in comp.raxial_list:
        if hasattr(item, '_visited') and item._visited:
            config.LOGGER.warning('Cycle between: %s and %s' % (comp.path, 'and', item.path))
            return True
        ret = ret or has_cycle(item)
    return ret
        
# 
# compartment.py ends here
