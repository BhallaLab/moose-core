# -*- coding: utf-8 -*-
# test_synchan.py ---

import moose
print( 'Using moose from %s' % moose.__file__ )

def make_synapse(path):
    """Create a synapse with two time constants. Connect a spikegen to the
    synapse. Create a pulsegen to drive the spikegen."""
    syn = moose.SynChan(path)
    syn.tau1 = 5.0 # ms
    syn.tau2 = 1.0 # ms
    syn.Gk = 1.0 # mS
    syn.Ek = 0.0

    # IN new implementation, there is SimpleSynHandler class which takes cares
    # of multiple synapses.
    synH = moose.SimpleSynHandler( '%s/SynHandler' % path)
    ss = synH.synapse.vec

    synH.synapse.num = 1
    synH.synapse.delay = 1.0
    synH.synapse.weight = 1.0
    moose.connect(synH, 'activationOut', syn, 'activation')
    print('Synapses:', synH.synapse.num, 'w=', synH.synapse[0].weight )

    spikegen = moose.SpikeGen('%s/spike' % (syn.parent.path))
    spikegen.edgeTriggered = False # Make it fire continuously when input is high
    spikegen.refractT = 10.0 # With this setting it will fire at 1 s / 10 ms = 100 Hz
    spikegen.threshold = 0.5
    # This will send alternatind -1 and +1 to SpikeGen to make it fire
    spike_stim = moose.PulseGen('%s/spike_stim' % (syn.parent.path))
    spike_stim.delay[0] = 1.0
    spike_stim.level[0] = 1.0
    spike_stim.width[0] = 100.0
    moose.connect(spike_stim, 'output', spikegen, 'Vm')
    print(synH.synapse, synH.synapse.vec)
    m = moose.connect(spikegen, 'spikeOut', synH.synapse.vec, 'addSpike', 'Sparse')
    m.setRandomConnectivity(1.0, 1)
    m = moose.connect(spikegen, 'spikeOut', synH.synapse[0], 'addSpike') # this causes segfault
    print('Constructed synapses')
    return syn, spikegen

def test_synchan():
    model = moose.Neutral('/model')
    syn, spikegen = make_synapse('/model/synchan')
    moose.setClock(0, 0.01)
    moose.useClock(0, '/model/##', 'process')
    moose.reinit()
    moose.start(100)

if __name__ == '__main__':
    test_synchan()
