#!/usr/bin/env python
#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU Lesser General Public License version 2.1
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

'''
This LIF network with Ca plasticity is based on:
David Higgins, Michael Graupner, Nicolas Brunel
    Memory Maintenance in Synapses with Calcium-Based
    Plasticity in the Presence of Background Activity
    PLOS Computational Biology, 2014.

Author: Aditya Gilra, NCBS, Bangalore, October, 2014.
'''

## import modules and functions to be used
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import moose

np.random.seed(100) # set seed for reproducibility of simulations
random.seed(100) # set seed for reproducibility of simulations

#############################################
# All parameters as per:
# David Higgins, Michael Graupner, Nicolas Brunel
#     Memory Maintenance in Synapses with Calcium-Based
#     Plasticity in the Presence of Background Activity
#     PLOS Computational Biology, 2014.
#############################################

#############################################
# Neuron model
#############################################

# equation: dv/dt = (1/taum)*(-(v-el)) + inp
# with spike when v>vt, reset to vr

el = -70e-3  #V        # Resting potential
vt = -50e-3  #V        # Spiking threshold
Rm = 20e6    #Ohm      # Only taum is needed, but LIF neuron accepts 
Cm = 1e-9    #F        # Rm and Cm and constructs taum=Rm*Cm
taum = Rm*Cm #s        # Membrane time constant is 20 ms
vr = -60e-3  #V        # Reset potential
inp = 24e-3/taum #V/s  # inp = Iinject/Cm to each neuron
                       # same as setting el=-41 mV and inp=0
Iinject = inp*Cm       # LIF neuron has injection current as param

#############################################
# Network parameters: numbers
#############################################

N = 1000          # Total number of neurons
fexc = 0.8        # Fraction of exc neurons
NE = int(fexc*N)  # Number of excitatory cells
NI = N-NE         # Number of inhibitory cells 

#############################################
# Simulation parameters
#############################################

simtime = 1.      #s # Simulation time
dt = 1e-3         #s # time step

#############################################
# Network parameters: synapses (not for ExcInhNetBase)
#############################################

## With each presynaptic spike in exc / inh neuron,
## J / -g*J is added to post-synaptic Vm -- delta-fn synapse
## Since LIF neuron used below is derived from Compartment class,
## conductance-based synapses (SynChan class) can also be used.

C = 50           # Number of incoming connections on each neuron (exc or inh)
fC = fexc         # fraction fC incoming connections are exc, rest inhibitory
J = 0.8e-3 #V     # exc strength is J (in V as we add to voltage)
                  # Critical J is ~ 0.45e-3 V in paper for N = 10000, C = 1000
                  # See what happens for J = 0.2e-3 V versus J = 0.8e-3 V
g = 5.0           # -gJ is the inh strength. For exc-inh balance g >~ f(1-f)=4
syndelay = 0.5e-3 + dt # s     # synaptic delay:
                               # 0 ms gives similar result contrary to Ostojic?!
refrT = 0.5e-3    # s     # absolute refractory time -- 0 ms gives similar result

#############################################
# Ca Plasticity parameters: synapses (not for ExcInhNetBase)
#############################################

CaPlasticity = True    # set it True or False to turn on/off plasticity
tauCa = 22.6936e-3      # s # Ca decay time scale
tauSyn = 346.3615       # s # synaptic plasticity time scale
## in vitro values in Higgins et al 2014, faster plasticity
CaPre = 0.56175         # mM
CaPost = 1.2964         # mM
## in vivo values in Higgins et al 2014, slower plasticity
#CaPre = 0.33705         # mM
#CaPost = 0.74378        # mM
delayD = 4.6098e-3      # s # CaPre is added to Ca after this delay
                        # proxy for rise-time of NMDA
thetaD = 1.0            # mM # depression threshold for Ca
thetaP = 1.3            # mM # potentiation threshold for Ca
gammaD = 331.909        # factor for depression term
gammaP = 725.085        # factor for potentiation term

eqWeight = 0.43         # initial synaptic weight
                        # gammaP/(gammaP+gammaD) = eq weight w/o noise
                        # see eqn (22), noiseSD also appears
                        # but doesn't work here, 
                        # weights away from 0.4 - 0.5 screw up the STDP rule!!

bistable = True        # if bistable is True, use bistable potential for weights
noisy = False          # use noisy weight updates given by noiseSD
noiseSD = 3.3501        # if noisy, use noiseSD (3.3501 from Higgins et al 2014)
#noiseSD = 0.1           # if bistable==False, use a smaller noise than in Higgins et al 2014

#############################################
# Exc-Inh network base class without connections
#############################################

class ExcInhNetBase:
    """Simulates and plots LIF neurons (exc and inh separate).
    Author: Aditya Gilra, NCBS, Bangalore, India, October 2014
    """
    
    def __init__(self,N=N,fexc=fexc,el=el,vt=vt,Rm=Rm,Cm=Cm,vr=vr,\
        refrT=refrT,Iinject=Iinject):
        """ Constructor of the class """
        
        self.N = N                 # Total number of neurons
        self.fexc = fexc           # Fraction of exc neurons
        self.NmaxExc = int(fexc*N) # max idx of exc neurons, rest inh

        self.el = el        # Resting potential
        self.vt = vt        # Spiking threshold
        self.taum = taum    # Membrane time constant
        self.vr = vr        # Reset potential
        self.refrT = refrT  # Absolute refractory period
        self.Rm = Rm        # Membrane resistance
        self.Cm = Cm        # Membrane capacitance
        self.Iinject = Iinject # constant input current
        
        self.simif = False  # whether the simulation is complete
        
        self._setup_network()

    def __str__(self):
         return "LIF network of %d neurons "\
             "having %d exc." % (self.N,self.NmaxExc)
    
    def _setup_network(self):
        """Sets up the network (_init_network is enough)"""
        self.network = moose.LIF( 'network', self.N );
        moose.le( '/network' )
        self.network.vec.Em = self.el
        self.network.vec.thresh = self.vt
        self.network.vec.refractoryPeriod = self.refrT
        self.network.vec.Rm = self.Rm
        self.network.vec.vReset = self.vr
        self.network.vec.Cm = self.Cm
        self.network.vec.inject = self.Iinject

    def _init_network(self,v0=el):
        """Initialises the network variables before simulation"""        
        self.network.vec.initVm = v0
        
    def simulate(self,simtime=simtime,dt=dt,plotif=False,**kwargs):
        
        self.dt = dt
        self.simtime = simtime
        self.T = np.ceil(simtime/dt)
        self.trange = np.arange(0,self.simtime,dt)   
        
        self._init_network(**kwargs)
        if plotif:
            self._init_plots()
        
        # moose simulation
        moose.useClock( 0, '/network/syns', 'process' )
        moose.useClock( 1, '/network', 'process' )
        moose.useClock( 2, '/plotSpikes', 'process' )
        moose.useClock( 3, '/plotVms', 'process' )
        if CaPlasticity:
            moose.useClock( 3, '/plotWeights', 'process' )
            moose.useClock( 3, '/plotCa', 'process' )
        moose.setClock( 0, dt )
        moose.setClock( 1, dt )
        moose.setClock( 2, dt )
        moose.setClock( 3, dt )
        moose.setClock( 9, dt )
        t1 = time.time()
        print 'reinit MOOSE -- takes a while ~20s.'
        moose.reinit()
        print 'reinit time t = ', time.time() - t1
        t1 = time.time()
        print 'starting'
        moose.start(self.simtime)
        print 'runtime, t = ', time.time() - t1

        if plotif:
            self._plot()

    def _init_plots(self):
        ## make a few tables to store a few Vm-s
        numVms = 10
        self.plots = moose.Table( '/plotVms', numVms )
        ## draw numVms out of N neurons
        nrnIdxs = random.sample(range(self.N),numVms)
        for i in range( numVms ):
            moose.connect( self.network.vec[nrnIdxs[i]], 'VmOut', \
                self.plots.vec[i], 'input')

        ## make self.N tables to store spikes of all neurons
        self.spikes = moose.Table( '/plotSpikes', self.N )
        moose.connect( self.network, 'spikeOut', \
            self.spikes, 'input', 'OneToOne' )

        ## make 2 tables to store spikes of all exc and all inh neurons
        self.spikesExc = moose.Table( '/plotSpikesAllExc' )
        for i in range(self.NmaxExc):
            moose.connect( self.network.vec[i], 'spikeOut', \
                self.spikesExc, 'input' )
        self.spikesInh = moose.Table( '/plotSpikesAllInh' )
        for i in range(self.NmaxExc,self.N):
            moose.connect( self.network.vec[i], 'spikeOut', \
                self.spikesInh, 'input' )

    def _plot(self):
        """ plots the spike raster for the simulated net"""
        
        plt.figure()
        for i in range(0,self.NmaxExc):
            if i==0: label = 'Exc. spike trains'
            else: label = ''
            spikes = self.spikes.vec[i].vector
            plt.plot(spikes,[i]*len(spikes),\
                'b.',marker=',',label=label)
        for i in range(self.NmaxExc,self.N):
            if i==self.NmaxExc: label = 'Inh. spike trains'
            else: label = ''
            spikes = self.spikes.vec[i].vector
            plt.plot(spikes,[i]*len(spikes),\
                'r.',marker=',',label=label)           
        plt.xlabel('Time [ms]')
        plt.ylabel('Neuron number [#]')
        plt.xlim([0,self.simtime])
        plt.title("%s" % self, fontsize=14,fontweight='bold')
        plt.legend(loc='upper left')

#############################################
# Exc-Inh network class with Ca plasticity based connections
# (inherits from ExcInhNetBase)
#############################################

class ExcInhNet(ExcInhNetBase):
    """ Recurrent network simulation """
    
    def __init__(self,J=J,incC=C,fC=fC,scaleI=g,syndelay=syndelay,**kwargs):
        """Overloads base (parent) class"""
        self.J = J              # exc connection weight
        self.incC = incC        # number of incoming connections per neuron
        self.fC = fC            # fraction of exc incoming connections
        self.excC = int(fC*incC)# number of exc incoming connections
        self.scaleI = scaleI    # inh weight is scaleI*J
        self.syndelay = syndelay# synaptic delay

        # call the parent class constructor
        ExcInhNetBase.__init__(self,**kwargs) 
    
    def __str__(self):
         return "LIF network of %d neurons "\
             "of which %d are exc." % (self.N,self.NmaxExc) 
 
    def _init_network(self,**args):
        ExcInhNetBase._init_network(self,**args)
        
    def _init_plots(self):
        ExcInhNetBase._init_plots(self)
        self.recN = 5 # number of neurons for which to record weights and Ca
        if CaPlasticity:
            ## make self.N tables to store weight of 2 incoming synapses
            ## for a post-synaptic neuron: one exc, one inh synapse
            self.weights = moose.Table( '/plotWeights', self.excC*self.recN )
            for i in range(self.recN):      # range(self.N) is too large
                for j in range(self.excC):
                    moose.connect( self.weights.vec[self.excC*i+j], 'requestOut',
                        self.synsEE.vec[i].synapse[j], 'getWeight')            
            self.CaTables = moose.Table( '/plotCa', self.recN )
            for i in range(self.recN):      # range(self.N) is too large
                moose.connect( self.CaTables.vec[i], 'requestOut',
                        self.synsEE.vec[i], 'getCa')            

    def _setup_network(self):
        ## Set up the neurons without connections
        ExcInhNetBase._setup_network(self)  

        ## Now, add in the connections...
        ## Each LIF neuron has one incoming SynHandler,
        ##  which collects the activation from all presynaptic neurons
        ## Each pre-synaptic spike cause Vm of post-neuron to rise by
        ##  synaptic weight in one time step i.e. delta-fn synapse.
        ## Since LIF neuron is derived from Compartment class,
        ## conductance-based synapses (SynChan class) can also be used.
        ## E to E synapses can be plastic
        if CaPlasticity:
            self.synsEE = moose.GraupnerBrunel2012CaPlasticitySynHandler( \
                '/network/synsEE', self.NmaxExc )
        else:
            self.synsEE = moose.SimpleSynHandler( '/network/synsEE', self.NmaxExc )
        ## I to E synapses are not plastic
        self.synsIE = moose.SimpleSynHandler( '/network/synsIE', self.NmaxExc )
        ## all synapses to I neurons are not plastic
        self.synsI = moose.SimpleSynHandler( '/network/synsI', self.N-self.NmaxExc )
        ## connect all SynHandlers to their respective neurons
        for i in range(self.NmaxExc):
            moose.connect( self.synsEE.vec[i], 'activationOut', \
                self.network.vec[i], 'activation' )
            moose.connect( self.synsIE.vec[i], 'activationOut', \
                self.network.vec[i], 'activation' )
        for i in range(self.NmaxExc,self.N):
            moose.connect( self.synsEE.vec[i-self.NmaxExc], 'activationOut', \
                self.network.vec[i], 'activation' )

        ## Connections from some Exc/Inh neurons to each Exc neuron
        for i in range(0,self.NmaxExc):
            ## each neuron has excC number of EE synapses
            self.synsEE.vec[i].numSynapses = self.excC
            self.synsIE.vec[i].numSynapses = self.incC-self.excC

            ## set parameters for the Ca Plasticity SynHandler
            ## in the post-synaptic neuron
            if CaPlasticity:
                connectExcId = moose.connect( self.network.vec[i], \
                    'spikeOut', self.synsEE.vec[i], 'addPostSpike')
                self.synsEE.vec[i].CaInit = 0.0
                self.synsEE.vec[i].tauCa = tauCa
                self.synsEE.vec[i].tauSyn = tauSyn
                self.synsEE.vec[i].CaPre = CaPre
                self.synsEE.vec[i].CaPost = CaPost
                self.synsEE.vec[i].delayD = delayD
                self.synsEE.vec[i].thetaD = thetaD
                self.synsEE.vec[i].thetaP = thetaP
                self.synsEE.vec[i].gammaD = gammaD
                self.synsEE.vec[i].gammaP = gammaP
                self.synsEE.vec[i].weightMax = 1.0   # bounds on the weight
                self.synsEE.vec[i].weightMin = 0.0
                self.synsEE.vec[i].weightScale = \
                        self.J/eqWeight   # weight is eqWeight
                                          # weightScale = J/eqWeight
                                          # weight*weightScale=J is activation
                self.synsEE.vec[i].noisy = noisy
                self.synsEE.vec[i].noiseSD = noiseSD
                self.synsEE.vec[i].bistable = bistable

            ## Connections from some Exc neurons to each Exc neuron
            ## draw excC number of neuron indices out of NmaxExc neurons
            preIdxs = random.sample(range(self.NmaxExc),self.excC)
            ## connect these presynaptically to i-th post-synaptic neuron
            for synnum,preIdx in enumerate(preIdxs):
                synij = self.synsEE.vec[i].synapse[synnum]
                connectExcId = moose.connect( self.network.vec[preIdx], \
                    'spikeOut', synij, 'addSpike')
                synij.delay = syndelay
                synij.weight = eqWeight

            ## Connections from some Inh neurons to each Exc neuron
            ## draw inhC=incC-excC number of neuron indices out of inhibitory neurons
            preIdxs = random.sample(range(self.NmaxExc,self.N),self.incC-self.excC)
            ## connect these presynaptically to i-th post-synaptic neuron
            for synnum,preIdx in enumerate(preIdxs):
                synij = self.synsIE.vec[i].synapse[synnum]
                connectInhId = moose.connect( self.network.vec[preIdx], \
                    'spikeOut', synij, 'addSpike')
                synij.delay = syndelay
                synij.weight = -self.scaleI

        ## Connections from some Exc/Inh neurons to each Inh neuron
        for i in range(self.N-self.NmaxExc):
            ## each neuron has incC number of synapses
            self.synsI.vec[i].numSynapses = self.incC

            ## draw excC number of neuron indices out of NmaxExc neurons
            preIdxs = random.sample(range(self.NmaxExc),self.excC)
            ## connect these presynaptically to i-th post-synaptic neuron
            for synnum,preIdx in enumerate(preIdxs):
                synij = self.synsI.vec[i].synapse[synnum]
                connectExcId = moose.connect( self.network.vec[preIdx], \
                    'spikeOut', synij, 'addSpike')
                synij.delay = syndelay
                synij.weight = 1.0

            ## draw inhC=incC-excC number of neuron indices out of inhibitory neurons
            preIdxs = random.sample(range(self.NmaxExc,self.N),self.incC-self.excC)
            ## connect these presynaptically to i-th post-synaptic neuron
            for synnum,preIdx in enumerate(preIdxs):
                synij = self.synsI.vec[i].synapse[ self.excC + synnum ]
                connectInhId = moose.connect( self.network.vec[preIdx], \
                    'spikeOut', synij, 'addSpike')
                synij.delay = syndelay
                synij.weight = -self.scaleI

#############################################
# Analysis functions
#############################################

def rate_from_spiketrain(spiketimes,fulltime,dt,tau=50e-3):
    """
    Returns a rate series of spiketimes convolved with a Gaussian kernel;
    all times must be in SI units.
    """
    sigma = tau/2.
    ## normalized Gaussian kernel, integral with dt is normed to 1
    ## to count as 1 spike smeared over a finite interval
    norm_factor = 1./(np.sqrt(2.*np.pi)*sigma)
    gauss_kernel = np.array([norm_factor*np.exp(-x**2/(2.*sigma**2))\
        for x in np.arange(-5.*sigma,5.*sigma+dt,dt)])
    kernel_len = len(gauss_kernel)
    ## need to accommodate half kernel_len on either side of fulltime
    rate_full = np.zeros(int(fulltime/dt)+kernel_len)
    for spiketime in spiketimes:
        idx = int(spiketime/dt)
        rate_full[idx:idx+kernel_len] += gauss_kernel
    ## only the middle fulltime part of the rate series
    ## This is already in Hz,
    ## since should have multiplied by dt for above convolution
    ## and divided by dt to get a rate, so effectively not doing either.
    return rate_full[kernel_len/2:kernel_len/2+int(fulltime/dt)]

#############################################
# Make plots
#############################################

def extra_plots(net):
    ## extra plots apart from the spike rasters
    ## individual neuron Vm-s
    plt.figure()
    plt.plot(net.trange,net.plots.vec[0].vector[0:len(net.trange)])
    plt.plot(net.trange,net.plots.vec[1].vector[0:len(net.trange)])
    plt.plot(net.trange,net.plots.vec[2].vector[0:len(net.trange)])
    plt.xlabel('time (s)')
    plt.ylabel('Vm (V)')
    plt.title("Vm-s of 3 LIF neurons (spike = reset).")

    timeseries = net.trange
    ## individual neuron firing rates
    fig = plt.figure()
    plt.subplot(221)
    num_to_plot = 10
    #rates = []
    for nrni in range(num_to_plot):
        rate = rate_from_spiketrain(\
            net.spikes.vec[nrni].vector,simtime,dt)
        plt.plot(timeseries,rate)
    plt.title("Rates of "+str(num_to_plot)+" exc nrns")
    plt.ylabel("Hz")
    plt.ylim(0,100)
    plt.subplot(222)
    for nrni in range(num_to_plot):
        rate = rate_from_spiketrain(\
            net.spikes.vec[net.NmaxExc+nrni].vector,simtime,dt)
        plt.plot(timeseries,rate)
    plt.title("Rates of "+str(num_to_plot)+" inh nrns")
    plt.ylim(0,100)

    ## population firing rates
    plt.subplot(223)
    rate = rate_from_spiketrain(net.spikesExc.vector,simtime,dt)\
        /float(net.NmaxExc) # per neuron
    plt.plot(timeseries,rate)
    plt.ylim(0,100)
    plt.title("Exc population rate")
    plt.ylabel("Hz")
    plt.xlabel("Time (s)")
    plt.subplot(224)
    rate = rate_from_spiketrain(net.spikesInh.vector,simtime,dt)\
        /float(net.N-net.NmaxExc) # per neuron    
    plt.plot(timeseries,rate)
    plt.ylim(0,100)
    plt.title("Inh population rate")
    plt.xlabel("Time (s)")

    fig.tight_layout()

    ## Ca plasticity: weight vs time plots
    if CaPlasticity:
        ## Ca versus time in post-synaptic neurons
        plt.figure()
        for i in range(net.recN):      # range(net.N) is too large
                plt.plot(timeseries,\
                    net.CaTables.vec[i].vector[:len(timeseries)])
        plt.title("Evolution of Ca in some neurons")
        plt.xlabel("Time (s)")
        plt.ylabel("Ca (mM)")

        plt.figure()
        for i in range(net.recN):      # range(net.N) is too large
            for j in range(net.excC):
                plt.plot(timeseries,\
                    net.weights.vec[net.excC*i+j].vector[:len(timeseries)])
        plt.title("Evolution of some weights")
        plt.xlabel("Time (s)")
        plt.ylabel("Weight (V)")

        ## all EE weights are used for a histogram
        weights = [ net.synsEE.vec[i].synapse[j].weight \
                    for i in range(net.NmaxExc) for j in range(net.excC) ]
        plt.figure()
        plt.hist(weights, bins=100)

if __name__=='__main__':
    ## ExcInhNetBase has unconnected neurons,
    ## ExcInhNet connects them
    ## Instantiate either ExcInhNetBase or ExcInhNet below
    #net = ExcInhNetBase(N=N)
    net = ExcInhNet(N=N)
    print net
    ## Important to distribute the initial Vm-s
    ## else weak coupling gives periodic synchronous firing
    net.simulate(simtime,plotif=True,\
        v0=np.random.uniform(el-20e-3,vt,size=N))

    extra_plots(net)
    plt.show()
    
