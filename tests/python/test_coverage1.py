import os
import moose
import copy
import numpy as np
import sys
import itertools
import rdesigneur as rd
import csv 

sdir_ = os.path.dirname(__file__)

params = {
    'diffusionLength':0.5e-6,  # Diffusion characteristic length, used as voxel length too.
    'dendDiameter': 1.0e-6, # Diameter of section of dendrite in model
    'dendLength': 0.5e-6,    # Length of section of dendrite in model
    'diffConstCa':100e-12,  # Diffusion constant of Ca
    'stimAmplitude': 0.008, # Ca Stimulus amplitude, mM
    'baseCa':2.5e-4,        # Base Ca level, mM.
    'BAPCa':0.002,          # Dend Ca spike amplitude
    'BAPwidth':0.1,         # Dend Ca spike width.
    'blankVoxelsAtEnd':50,  # of voxels to leave blank at end of cylinder
    'preStimTime':10.0,     # Time to run before turning on stimulus.
    'postStimTime':100.0,    # Time to run after stimulus.
    'stimWidth': 2,       # Duration of Ca influx for each stimulus.
    'spineSpacing':0.5e-6,  # Spacing between spines.
    'diffConstMAPK': 5e-12, # Diffusion constant for MAPK
    'diffConstPP': 2e-12,   # Diff constant for MAPK-activated phosphatase
    'CaActivateRafKf': 6e6, # 1/sec/mM^2: rate for activation of Raf by Ca
    'cellModel':'PassiveSoma',  # Cell morphology script
    'chemModel': os.path.join(sdir_, 'NN_mapk14.g'),  # Chem model definition
    'seqDt': 3.0,           # Time interval between successive inputs in seq
    'seqDx': 3.0e-6,        # Distance between successive inputs in seq.
    'seqLength' : 5,   # Sequence Length 
    'seed': 12345,          # Seed for random number generator
    'sequence': '01234',    # Sequence of spines, spaced by seqDx microns, 
                            # activated every seqDt seconds
    'fnumber': 0,           # identifier for run
    'basalCaConc' : 5e-6    # Basal Ca conc
}

def makePassiveSoma( name, length, diameter ):
    elecid = moose.Neuron( '/library/' + name )
    dend = moose.Compartment( elecid.path + '/soma' )
    dend.diameter = diameter
    dend.length = length
    dend.x = length
    return elecid

def setDiffConst( element, paramName ):
    e = moose.element( '/library/chem/kinetics/DEND/' + element )
    e.diffConst = params[ paramName ]

# stimQ is { time:[index, CaConc] }
# Expected Ca concs are BaseCa = 5e-6 and stimulated Ca = 0.005. 
def runStimulus( stimQ ):
    #  print( stimQ )
    Ca_input = moose.vec('/model/chem/psd/Ca_input')
    #  print(Ca_input)
    Ca_input.concInit = 5e-6
    moose.vec( '/model/chem/dend/DEND/Ca_input' ).concInit = 5e-6
    moose.reinit()

    clock = moose.element( '/clock' )
    for t in sorted( stimQ ):
        currt = clock.currentTime
        if ( t > currt ):
            moose.start( t - currt )
            #  print("Ran for ", t-currt)
            for entry in stimQ[t]:
                index, conc = entry
                #The below line will work in Python3 if int(index) is used
                print(index)
                Ca_input[index].concInit = conc 

    moose.start(params['postStimTime'] )
    #  print "Finished stimulus run at t = ", clock.currentTime

def runAndPrint( seq):
    runStimulus( seq )
    mapk = moose.vec( '/model/graphs/plot0' )
    binEdges = np.linspace(0, 1.6, 17)
    np.histogram([np.sum(i.vector) for i in mapk], bins=binEdges)
    tot = np.sum( np.array([i.vector for i in mapk] ) )
    cam_ca4 = moose.vec( '/model/graphs/plot1' )
    print ("tot : ", tot)
    print(seq,  tot)
    return(tot)

def panelEFspatialSeq(fname ):
    #  print ("Starting Panel EF")
    moose.seed( int(params['seed']) )
    rdes = rd.rdesigneur(
        useGssa = False,
        turnOffElec = True,
        chemPlotDt = 0.02,
        diffusionLength = params['diffusionLength'],
        spineProto = [['makePassiveSpine()', 'spine']],
        spineDistrib = [['spine', '#', str(params['spineSpacing']),'-1e-7','1.4','0','0', 'p*6.28e7']], # Spiral, uniform spacing.
        cellProto = [['cell', 'soma']],
        chemProto = [[params['chemModel'], 'chem']],
        chemDistrib = [['chem', 'soma', 'install', '1' ]],
        plotList = [
            ['soma', '1', 'dend/DEND/P_MAPK', 'conc', '[dend P_MAPK]'],
            ['soma', '1', 'dend/DEND/CaM/CaM', 'conc', '[dend CaM]'],
            ['#head#', '1', 'psd/Ca', 'conc', '[PSD Ca]'],
            ['#head#', '1', 'spine/Ca', 'conc', '[spine Ca]'],
            ['soma', '1', 'dend/DEND/Ca', 'conc', '[dend Ca]'],
            ],
        #  moogList = [['#', '11', '.', 'conc', '[Ca]']]
    )
    # Assign parameters to the prototype model.
    setDiffConst( 'Ca', 'diffConstCa' )
    setDiffConst( '../../compartment_1/Ca', 'diffConstCa' )
    setDiffConst( '../../compartment_2/Ca', 'diffConstCa' )
    setDiffConst( 'P_MAPK', 'diffConstMAPK' )
    setDiffConst( 'MAPK', 'diffConstMAPK' )
    setDiffConst( 'reg_phosphatase', 'diffConstPP' )
    setDiffConst( 'inact_phosphatase', 'diffConstPP' )
    moose.element( '/library/chem/kinetics/DEND/Ca_activate_Raf' ).Kf = params['CaActivateRafKf']
    #  print ("Set up rdesigneur")

    rdes.buildModel()
    #  print ("MODEL BUILT")

    ################################################################
    # Run and print the stimulus
    # stimQ is { time:[index, CaConc], ... }
    ampl = params['stimAmplitude']
    basalAmpl = params['basalCaConc']
    seq1 = { 10: [(10, ampl)], 12:[(10, basalAmpl)], 14:[(13,ampl)],
            16:[(13,basalAmpl)], 18: [(16, ampl)], 20: [(16, basalAmpl)], 
            22:[(19,ampl)], 24:[(19,basalAmpl)], 26:[(22, ampl)],
            28:[(22, basalAmpl)]
            }
    seq2 = { 10: [(10, ampl)], 12:[(10, basalAmpl)], 14:[(19,ampl)],
            16:[(19,basalAmpl)], 18: [(13, ampl)], 20: [(13, basalAmpl)], 
            22:[(22,ampl)], 24:[(22,basalAmpl)], 26:[(16, ampl)],
            28:[(16, basalAmpl)]
            }
    M = params['seqLength'] 
    times = np.arange(int(params['preStimTime']),
            int(params['preStimTime']+params['seqDt']*M), int(params['seqDt']))
    spineSpacing = int(params['seqDx'] / params['spineSpacing'])
    #  print("spine Spavcing",spineSpacing)
    locs = np.arange(params['blankVoxelsAtEnd'], 
            params['blankVoxelsAtEnd']+spineSpacing*M, spineSpacing)
    #  print("times", times)
    #  print("locs", locs)
    
    seqList = []
    for p, pattern in enumerate(itertools.permutations(locs, M)):
        #  print(pattern)
        seq = {}
        for i, loc in enumerate(pattern):
            if times[i] in seq:
                seq[times[i]].append([loc, ampl])
            else:
                seq[times[i]] = [[loc, ampl]]
            if times[i]+params['stimWidth'] in seq:
                seq[times[i+params['stimWidth']]].append([
                        loc, basalAmpl])
            else:
                seq[times[i]+params['stimWidth']] = [[
                        loc, basalAmpl]]

        if p==0:
            referenceSeq = copy.deepcopy(seq)
            seqList.append(referenceSeq)

        #  for ectPattern in product(times, range(M)):
        '''
        ectTime = (times[0] + times[-1]) / 2.0
        ectLoc = int((locs[0] + locs[-1]) / 2.0)
        if ectTime in seq:
            if seq[ectTime][0][0] == ectLoc:
                seq[ectTime][0][1] += ampl
            else:
                seq[ectTime].append([ectLoc, ampl])
        else:
            seq[ectTime] = [[ectLoc, ampl]]
        if ectTime+params['stimWidth'] in seq:
            if seq[ectTime+params['stimWidth']][0][0] == ectLoc:
                seq[ectTime+params['stimWidth']][0][1] += basalAmpl
            else:
                seq[ectTime+params['stimWidth']].append([ectLoc, basalAmpl])
        else:
            seq[ectTime+params['stimWidth']] = [[ectLoc, basalAmpl]]
        '''
        seqList.append(seq)
    #  runAndPrint(referenceSeq)
    #  print(seqList)
    #  p = Pool(16)
    #  p.map(runAndPrint, seqList)
    with open(fname, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for seq in [seqList[0]]:
            writer.writerow([seq,runAndPrint( seq)])
            rdes.display()
    #  print(seqList[0])
    #  runAndPrint( seqList[0] )
    #  runAndPrint( seqList[-1] )
    #  runAndPrint( seq2 )

    #  rdes.displayMoogli(0.5, 40, 0)
    moose.delete( '/model' )

def main():
    global params
    moose.Neutral( '/library' )
    for ii in range( len( sys.argv ) ):
        if sys.argv[ii][:2] == '--':
            argName = sys.argv[ii][2:]
            if argName in params:
                params[argName] = float( sys.argv[ii+1] )
                #  print(argName, params[argName])
                if argName == 'sequence':
                    params[argName] = sys.argv[ii+1] # Leave it as a str.

    moose.seed( int(params['seed']) )
    params['dendLength'] = (2*params['blankVoxelsAtEnd'] + 1) * \
            params['diffusionLength'] + \
            params['seqDx'] * (params['seqLength'] - 1)
    print(params['dendLength'])
    makePassiveSoma( 'cell', params['dendLength'], params['dendDiameter'] )
    filename = './detailed-model-ectopic-seqDx-%.6f-seqDt-%.2f-stimAmp-%.3f.csv' %(
            params['seqDx'], params['seqDt'], params['stimAmplitude'])
    panelEFspatialSeq(filename);

if __name__ == '__main__':
    main()
