# -*- coding: utf-8 -*-
"""test_muparser.py:

"""

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import os
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy as np
import pylab
import matplotlib.pyplot as plt
import moose
import abstrModelEqns9 as ame
import rdesigneur as rd


def singleCompt( name, params ):
    mod = moose.copy( '/library/' + name + '/' + name, '/model' )
    A = moose.element( mod.path + '/A' )
    Z = moose.element( mod.path + '/Z' )
    Z.nInit = 1
    Ca = moose.element( mod.path + '/Ca' )
    CaStim = moose.element( Ca.path + '/CaStim' )
    runtime = params['preStimTime'] + params['postStimTime']
    steptime = 50

    CaStim.expr += ' + x2 * (t > 100+' + str( runtime ) + ' ) * ( t < 100+' + str( runtime + steptime ) +  ' )'
    print(CaStim.expr)
    tab = moose.Table2( '/model/' + name + '/Atab' )
    #for i in range( 10, 19 ):
        #moose.setClock( i, 0.01 )
    ampl = moose.element( mod.path + '/ampl' )
    phase = moose.element( mod.path + '/phase' )
    moose.connect( tab, 'requestOut', A, 'getN' )
    ampl.nInit = params['stimAmplitude'] * 1
    phase.nInit = params['preStimTime']

    ksolve = moose.Ksolve( mod.path + '/ksolve' )
    stoich = moose.Stoich( mod.path + '/stoich' )
    stoich.compartment = mod
    stoich.ksolve = ksolve
    stoich.path = mod.path + '/##'
    #runtime += 2 * steptime

    moose.reinit()
    #runtime = 150.0
    runtime += 100 + steptime*2
    moose.start( runtime )
    t = np.arange( 0, runtime + 1e-9, tab.dt )
    return name, t, tab.vector
    #pylab.plot( t, tab.vector, label='[A] (mM)' )

    #pylab.show()

def plotBoilerplate( panelTitle, plotPos, xlabel = ''):
    plotPos += 4
    ax = plt.subplot( 7,4,plotPos )
    #ax.xaxis.set_ticks( i[1] )
    #ax.locator_params(
    ax.spines['top'].set_visible( False )
    ax.spines['right'].set_visible( False )
    ax.tick_params( direction = 'out' )
    if (((plotPos -1)/4) % 2) == 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel( xlabel )
    for tick in ax.xaxis.get_major_ticks():
        tick.tick2On = False
    for tick in ax.yaxis.get_major_ticks():
        tick.tick2On = False

    if (plotPos % 4) == 1:
        plt.ylabel( 'A (a.u.)', fontsize = 14 )
        # alternate way of doing this separately.
        #plt.yaxis.label.size_size(16)
        #plt.title( 'B' )
        ax.text( -0.3, 1, panelTitle, fontsize = 18, weight = 'bold',
        transform=ax.transAxes )
    return ax

def eqnPlot( index, label, adot, adot2, bdot ):
    ax = plt.subplot( 7,4,index)
    ax.axis( 'off' )
    ax.text( 0.15, 0.6, label, fontsize = 12, weight='bold' )
    ax.text( 0.0, 0.3, adot, fontsize = 12 )
    ax.text( 0.0, 0.1, adot2, fontsize = 12 )
    ax.text( 0.0, -0.4, bdot, fontsize = 12 )
    return ax

def plotPanelA():
    ax = eqnPlot( 1, "Neg. FB", r"$A'=-0.1A-0.2AB+10Ca$",
            "",
            r"$B'=0.2A-0.1B$")

    ax.text( -0.3,0.6, 'A', fontsize = 18, weight = 'bold', transform=ax.transAxes )

    ax = eqnPlot( 2, "Neg. FF", r"$A'=-0.1A-0.01AB+$",
            r"$10Ca/(1+40B^2)$",
            r"$B'=2Ca-0.05B$")

    ax = eqnPlot( 3, "FHN", r"$A'=5(A-2-(A-2)^3/3-$",
            r"$B+0.8+Ca)$",
            r"$2.5B'=A-2+0.7-0.8(B-0.8)$")

    ax = eqnPlot( 4, "Switch", r"$A'=0.1-5A+5A^2-A^3+$",
            r"$10Ca.A/(1+A+10B)-5AB$",
            r"$B'=0.01A^2-0.01B$")

def plotPanelB():
    ax = plt.subplot( 7,4,5)
    ax.axis( 'off' )
    ax.text( -0.3,0.6, 'B', fontsize = 18, weight = 'bold', transform=ax.transAxes )


def plotPanelC():
    panelC = []
    panelCticks = []
    panelC.append( singleCompt( 'negFB', ame.makeNegFB( [] ) ) )
    panelC.append( singleCompt( 'negFF', ame.makeNegFF( [] ) ) )
    panelC.append( singleCompt( 'fhn', ame.makeFHN( [] ) ) )
    panelC.append( singleCompt( 'bis', ame.makeBis( [] ) ) )

    panelCticks.append( np.arange( 0, 15.00001, 5 ) )
    panelCticks.append( np.arange( 0, 1.50001, 0.5 ) )
    panelCticks.append( np.arange( 0, 5.00002, 1 ) )
    panelCticks.append( np.arange( 0, 5.00002, 1 ) )
    moose.delete( '/model' )
    for i in zip( panelC, panelCticks, list(range( len( panelC ))) ):
        plotPos = i[2] + 5
        ax = plotBoilerplate( 'C', plotPos, 'Time (s)' )
        plt.plot( i[0][1], i[0][2] )
        doty = i[1][-1] * 0.95
        plt.plot((150, 200), (doty, doty), '-')
        plt.plot((10,), (doty,), 'ro')
        xmax = ax.get_xlim()[1]
        #ax.xaxis.set_ticks( np.arange( 0, xmax, 50 ) )
        ax.xaxis.set_ticks( np.arange( 0, 250.001, 50 ) )
        ax.yaxis.set_ticks( i[1] )

def runPanelDEFG( name, dist, seqDt, numSpine, seq, stimAmpl ):
    preStim = 10.0
    blanks = 20
    rdes = rd.rdesigneur(
        useGssa = False,
        turnOffElec = True,
        chemPlotDt = 0.1,
        #diffusionLength = params['diffusionLength'],
        diffusionLength = 1e-6,
        cellProto = [['cell', 'soma']],
        chemProto = [['dend', name]],
        chemDistrib = [['dend', 'soma', 'install', '1' ]],
        plotList = [['soma', '1', 'dend' + '/A', 'n', '# of A']],
    )
    rdes.buildModel()
    A = moose.vec( '/model/chem/dend/A' )
    Z = moose.vec( '/model/chem/dend/Z' )
    print(moose.element( '/model/chem/dend/A/Adot' ).expr)
    print(moose.element( '/model/chem/dend/B/Bdot' ).expr)
    print(moose.element( '/model/chem/dend/Ca/CaStim' ).expr)
    phase = moose.vec( '/model/chem/dend/phase' )
    ampl = moose.vec( '/model/chem/dend/ampl' )
    vel = moose.vec( '/model/chem/dend/vel' )
    vel.nInit = 1e-6 * seqDt
    ampl.nInit = stimAmpl
    stride = int( dist ) / numSpine
    phase.nInit = 10000
    Z.nInit = 0
    for j in range( numSpine ):
        k = int( blanks + j * stride )
        Z[k].nInit = 1
        phase[k].nInit = preStim + seq[j] * seqDt

    moose.reinit()
    runtime = 50
    snapshot = preStim + seqDt * (numSpine - 0.8)
    print(snapshot)
    #snapshot = 26
    moose.start( snapshot )
    avec = moose.vec( '/model/chem/dend/A' ).n
    moose.start( runtime - snapshot )
    tvec = []
    for i in range( 5 ):
        tab = moose.element( '/model/graphs/plot0[' + str( blanks + i * stride ) + ']' )
        dt = tab.dt
        tvec.append( tab.vector )
    moose.delete( '/model' )
    return dt, tvec, avec

def makePassiveSoma( name, length, diameter ):
    elecid = moose.Neuron( '/library/' + name )
    dend = moose.Compartment( elecid.path + '/soma' )
    dend.diameter = diameter
    dend.length = length
    dend.x = length
    return elecid

def plotOnePanel( tLabel, dt, tplot, numSyn, plotRange, tick ):
    t = np.arange( 0, len( tplot[0] ), 1.0 ) * dt
    ax = plotBoilerplate( tLabel, 1 + start )
    for i in range( 5 ):
        plt.plot( t, tplot[i] )
    ax.yaxis.set_ticks( np.arange( 0, plotRange, tick ) )


def plotPanelDEFG( seq, row ):
    makePassiveSoma( 'cell', 100e-6, 10e-6 )
    start = (row -1) * 4
    tLabel = chr( ord( 'B' ) + row - 1 )
    xLabel = chr( ord( 'D' ) + row - 1 )
    xplot = []

    ############################################################

    dt, tplot, avec = runPanelDEFG( 'negFB', 5.0, 2.0, 5, seq, 1.0 )
    xplot.append( avec )
    t = np.arange( 0, len( tplot[0] ), 1.0 ) * dt
    ax = plotBoilerplate( tLabel, 1 + start, 'Time (s)')
    for i in range( 5 ):
        plt.plot( t, tplot[i] )
    ax.yaxis.set_ticks( np.arange( 0, 10.00001, 5.0 ) )

    dt, tplot, avec = runPanelDEFG( 'negFF', 10.0, 1.0, 5, seq, 1.0 )
    xplot.append( avec )
    t = np.arange( 0, len( tplot[0] ), 1.0 ) * dt
    ax = plotBoilerplate( tLabel, 2 + start, 'Time (s)')
    for i in range( 5 ):
        plt.plot( t, tplot[i] )
    ax.yaxis.set_ticks( np.arange( 0, 1.50001, 0.5 ) )
    #dt, tplot, avec = runPanelDEFG( 'fhn', 15.0, 3.0, 5, seq, 0.4 )
    dt, tplot, avec = runPanelDEFG( 'fhn', 5.0, 1.5, 5, seq, 0.4 )
    xplot.append( avec )
    #plotOnePanel( dt, 'B', tplot, 5, 1.5, 0.5 )
    t = np.arange( 0, len( tplot[0] ), 1.0 ) * dt
    #ax = plotBoilerplate( tLabel, 1 + start )
    ax = plotBoilerplate( tLabel, 3 + start, 'Time (s)')
    for i in range( 5 ):
        plt.plot( t, tplot[i] )
    yl = ax.get_ylim()[1]
    ax.yaxis.set_ticks( np.arange( 0, 4.0001, 2.0 ) )

    #dt, tplot, avec = runPanelDEFG( 'bis', 5.0, 4.0, 5, seq, 1.0 )
    dt, tplot, avec = runPanelDEFG( 'bis', 15.0, 2.0, 5, seq, 1.0 )
    xplot.append( avec )
    t = np.arange( 0, len( tplot[0] ), 1.0 ) * dt
    ax = plotBoilerplate( tLabel, 4 + start, 'Time (s)' )
    for i in range( 5 ):
        plt.plot( t, tplot[i] )
    yl = ax.get_ylim()[1]
    ax.yaxis.set_ticks( np.arange( 0, 4.0001, 2.0 ) )

##################################################################
    for i in zip( list(range(4)), (10, 1.5, 4.0, 4.0 ), (5, 0.5, 2, 2) ):
        ax = plotBoilerplate( xLabel, 9 + start + i[0], r'Position ($\mu$m)' )
        plt.plot( xplot[i[0]][:50] )
        ax.yaxis.set_ticks( np.arange( 0, i[1] * 1.0000001, i[2] ) )

##################################################################

if __name__ == '__main__':
    moose.Neutral( '/library' )
    moose.Neutral( '/model' )
    fig = plt.figure( figsize = (12,12), facecolor='white' )
    fig.subplots_adjust( left = 0.18 )
    plotPanelA()
    plotPanelB()
    plotPanelC()
    plotPanelDEFG( [0,1,2,3,4], 3 )
    plotPanelDEFG( [4,1,0,3,2], 4 )
    plt.tight_layout()
    outfile ='%s.png' % sys.argv[0]
    plt.savefig( outfile )
    print( '[INFO] Saved to %s' % outfile )
