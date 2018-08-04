# -*- coding: utf-8 -*-
# This script is modified version of GraupnerBrunel2012 model by Aditya Gilra.
# Modification is following:
#  - Added global seed.
#  - Removed some messages.
#  - Added assertion.
#  
# NOTE: This script is used for testing random number generators on various
# platform. This should not be used in any tutorial or scientific demo.

import  matplotlib as mpl
mpl.use('Agg')
import moose
print( 'Using moose from %s' % moose.__file__ )
import numpy as np

moose.seed( 10 )

def main():
    """
    Simulate a pseudo-STDP protocol and plot the STDP kernel
    that emerges from Ca plasticity of Graupner and Brunel 2012.

    Author: Aditya Gilra, NCBS, Bangalore, October, 2014.
    """

    # ###########################################
    # Neuron models
    # ###########################################

    ## Leaky integrate and fire neuron
    Vrest = -65e-3 # V      # resting potential
    Vt_base = -45e-3 # V    # threshold
    Vreset = -55e-3 # V     # in current steps, Vreset is same as pedestal
    R = 1e8 # Ohm
    tau = 10e-3 # s
    refrT = 2e-3 # s

    # ###########################################
    # Initialize neuron group
    # ###########################################

    ## two neurons: index 0 will be presynaptic, 1 will be postsynaptic
    network = moose.LIF( 'network', 2 );
    moose.le( '/network' )
    network.vec.Em = Vrest
    network.vec.thresh = Vt_base
    network.vec.refractoryPeriod = refrT
    network.vec.Rm = R
    network.vec.vReset = Vreset
    network.vec.Cm = tau/R
    network.vec.inject = 0.
    network.vec.initVm = Vrest

    tauCa = 20e-3     
    tauSyn = 150.0    
    CaPre = 1.0       
    CaPost = 2.0      
    delayD = 13.7e-3  
    thetaD = 1.0      
    thetaP = 1.3      
    gammaD = 200.0    
    gammaP = 321.808  
    J = 5e-3 # V      
    weight = 0.5      
    bistable = True   

    syn = moose.GraupnerBrunel2012CaPlasticitySynHandler( '/network/syn' )
    syn.numSynapses = 1   
    moose.connect( syn, 'activationOut', network.vec[1], 'activation' )

    # synapse from presynaptic neuron
    moose.connect( network.vec[0],'spikeOut', syn.synapse[0], 'addSpike')

    # post-synaptic spikes also needed for STDP
    moose.connect( network.vec[1], 'spikeOut', syn, 'addPostSpike')

    syn.synapse[0].delay = 0.0
    syn.synapse[0].weight = weight
    syn.CaInit = 0.0
    syn.tauCa = tauCa
    syn.tauSyn = tauSyn
    syn.CaPre = CaPre
    syn.CaPost = CaPost
    syn.delayD = delayD
    syn.thetaD = thetaD
    syn.thetaP = thetaP
    syn.gammaD = gammaD
    syn.gammaP = gammaP
    syn.weightScale = J        
    syn.weightMax = 1.0 
    syn.weightMin = 0.

    syn.noisy = True
    syn.noiseSD = 1.3333
    syn.bistable = bistable

    # ###########################################
    # Setting up tables
    # ###########################################

    Vms = moose.Table( '/plotVms', 2 )
    moose.connect( network, 'VmOut', Vms, 'input', 'OneToOne')
    spikes = moose.Table( '/plotSpikes', 2 )
    moose.connect( network, 'spikeOut', spikes, 'input', 'OneToOne')
    CaTable = moose.Table( '/plotCa', 1 )
    moose.connect( CaTable, 'requestOut', syn, 'getCa')
    WtTable = moose.Table( '/plotWeight', 1 )
    moose.connect( WtTable, 'requestOut', syn.synapse[0], 'getWeight')

    dt = 1e-3 
    moose.useClock( 0, '/network/syn', 'process' )
    moose.useClock( 1, '/network', 'process' )
    moose.useClock( 2, '/plotSpikes', 'process' )
    moose.useClock( 3, '/plotVms', 'process' )
    moose.useClock( 3, '/plotCa', 'process' )
    moose.useClock( 3, '/plotWeight', 'process' )
    moose.setClock( 0, dt )
    moose.setClock( 1, dt )
    moose.setClock( 2, dt )
    moose.setClock( 3, dt )
    moose.setClock( 9, dt )
    moose.reinit()

    # function to make the aPlus and aMinus settle to equilibrium values
    settletime = 10e-3 # s
    def reset_settle():
        """ Call this between every pre-post pair
        to reset the neurons and make them settle to rest.
        """
        syn.synapse[0].weight = weight
        syn.Ca = 0.0
        moose.start(settletime)
        # Ca gets a jump at pre-spike+delayD
        # So this event can occur during settletime
        # So set Ca and weight once more after settletime
        syn.synapse[0].weight = weight
        syn.Ca = 0.0

    # function to inject a sharp current pulse to make neuron spike
    # immediately at a given time step
    def make_neuron_spike(nrnidx,I=1e-7,duration=1e-3):
        """ Inject a brief current pulse to
        make a neuron spike
        """
        network.vec[nrnidx].inject = I
        moose.start(duration)
        network.vec[nrnidx].inject = 0.

    dwlist_neg = []
    ddt = 10e-3 # s
    # since CaPlasticitySynHandler is event based
    # multiple pairs are needed for Ca to be registered above threshold
    # Values from Fig 2, last line of legend
    numpairs = 60           # number of spike parts per deltat
    t_between_pairs = 1.0   # time between each spike pair
    t_extent = 100e-3       # s  # STDP kernel extent,
                            # t_extent > t_between_pairs/2 inverts pre-post pairing!
    # dt = tpost - tpre
    # negative dt corresponds to post before pre
    print('-----------------------------------------------')
    for deltat in np.arange(t_extent,0.0,-ddt):
        reset_settle()
        for i in range(numpairs):
            # post neuron spike
            make_neuron_spike(1)
            moose.start(deltat)
            # pre neuron spike after deltat
            make_neuron_spike(0)
            moose.start(t_between_pairs)  # weight changes after pre-spike+delayD
                                          # must run for at least delayD after pre-spike
        dw = ( syn.synapse[0].weight - weight ) / weight
        print(('post before pre, dt = %1.3f s, dw/w = %1.3f'%(-deltat,dw)))
        dwlist_neg.append(dw)

    print('-----------------------------------------------')
    # positive dt corresponds to pre before post
    dwlist_pos = []
    for deltat in np.arange(ddt,t_extent+ddt,ddt):
        reset_settle()
        for i in range(numpairs):
            # pre neuron spike
            make_neuron_spike(0)
            moose.start(deltat)
            # post neuron spike after deltat
            make_neuron_spike(1)
            moose.start(t_between_pairs)
        dw = ( syn.synapse[0].weight - weight ) / weight
        print(('pre before post, dt = %1.3f s, dw/w = %1.3f'%(deltat,dw)))
        dwlist_pos.append(dw)

    Vmseries0 = Vms.vec[0].vector
    numsteps = len(Vmseries0)

    for t in spikes.vec[0].vector:
        Vmseries0[int(t/dt)-1] = 30e-3 # V

    Vmseries1 = Vms.vec[1].vector

    for t in spikes.vec[1].vector:
        Vmseries1[int(t/dt)-1] = 30e-3 # V

    timeseries = np.linspace(0.,200*numsteps*dt,numsteps)

    # STDP curve
    up, sp = np.mean( dwlist_pos ), np.std( dwlist_pos )
    un, sn = np.mean( dwlist_neg ), np.std( dwlist_neg )

    assert dwlist_pos == [0.009009761382687831, -0.028935548805074318, 0.026640291140274774,
            0.15096694961231227, 0.05169877176909221, -0.07561175582012891,
            -0.19621671306246136, -0.17264272470496844, -0.10031348546737717,
            0.07087873025966163]
    assert dwlist_neg == [-0.10549061671120019, -0.02712419419104095, 0.1126929558996459,
            0.015845230672240973, -0.17258572998325195, 0.15302349815061378,
            -0.22361180919667656, -0.03996270824811321, -0.20346690396689837,
            -0.24545719865254234]

    got = (up, sp)
    expNew = (-0.026452572369598148, 0.1045010909950147)
    assert np.isclose(got, expNew).all(), 'Expected: %s, Got: %s' % (str(expNew), str(got))

if __name__ == '__main__':
    main()
