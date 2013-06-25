# HsolveInstability.py --- 

# Commentary: 
# 
# A toy compartmental neuronal + chemical model that causes bad things
# to happen to the hsolver, as of 28 May 2013. Hopefully this will
# become irrelevant soon.
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 

# Code:

import sys
sys.path.append('../../python')
import os
os.environ['NUMPTHREADS'] = '1'
import math

import moose

EREST_ACT = -70e-3

# Gate equations have the form:
#
# y(x) = (A + B * x) / (C + exp((x + D) / F))
# 
# where x is membrane voltage and y is the rate constant for gate
# closing or opening

Na_m_params = [1e5 * (25e-3 + EREST_ACT),   # 'A_A':
                -1e5,                       # 'A_B':
                -1.0,                       # 'A_C':
                -25e-3 - EREST_ACT,         # 'A_D':
               -10e-3,                      # 'A_F':
                4e3,                     # 'B_A':
                0.0,                        # 'B_B':
                0.0,                        # 'B_C':
                0.0 - EREST_ACT,            # 'B_D':
                18e-3                       # 'B_F':    
               ]
Na_h_params = [ 70.0,                        # 'A_A':
                0.0,                       # 'A_B':
                0.0,                       # 'A_C':
                0.0 - EREST_ACT,           # 'A_D':
                0.02,                     # 'A_F':
                1000.0,                       # 'B_A':
                0.0,                       # 'B_B':
                1.0,                       # 'B_C':
                -30e-3 - EREST_ACT,        # 'B_D':
                -0.01                    # 'B_F':       
                ]        
K_n_params = [ 1e4 * (10e-3 + EREST_ACT),   #  'A_A':
               -1e4,                      #  'A_B':
               -1.0,                       #  'A_C':
               -10e-3 - EREST_ACT,         #  'A_D':
               -10e-3,                     #  'A_F':
               0.125e3,                   #  'B_A':
               0.0,                        #  'B_B':
               0.0,                        #  'B_C':
               0.0 - EREST_ACT,            #  'B_D':
               80e-3                       #  'B_F':  
               ]
VMIN = -30e-3 + EREST_ACT
VMAX = 120e-3 + EREST_ACT
VDIVS = 3000

def createSquid():
    """Create a single compartment squid model."""
    parent = moose.Neutral ('/n' )
    compt = moose.SymCompartment( '/n/compt' )
    Em = EREST_ACT + 10.613e-3
    compt.Em = Em
    compt.initVm = EREST_ACT
    compt.Cm = 7.85e-9 * 0.5
    compt.Rm = 4.2e5 * 5.0
    compt.Ra = 7639.44e3
    nachan = moose.HHChannel( '/n/compt/Na' )
    nachan.Xpower = 3
    xGate = moose.HHGate(nachan.path + '/gateX')    
    xGate.setupAlpha(Na_m_params + [VDIVS, VMIN, VMAX])
    xGate.useInterpolation = 1
    nachan.Ypower = 1
    yGate = moose.HHGate(nachan.path + '/gateY')
    yGate.setupAlpha(Na_h_params + [VDIVS, VMIN, VMAX])
    yGate.useInterpolation = 1
    nachan.Gbar = 0.942e-3
    nachan.Ek = 115e-3+EREST_ACT
    moose.connect(nachan, 'channel', compt, 'channel', 'OneToOne')

    kchan = moose.HHChannel( '/n/compt/K' )
    kchan.Xpower = 4.0
    xGate = moose.HHGate(kchan.path + '/gateX')    
    xGate.setupAlpha(K_n_params + [VDIVS, VMIN, VMAX])
    xGate.useInterpolation = 1
    kchan.Gbar = 0.2836e-3
    kchan.Ek = -12e-3+EREST_ACT
    moose.connect(kchan, 'channel', compt, 'channel', 'OneToOne')
    return compt

def createSpine( parentCompt, parentObj, index, frac, length, dia, theta ):
    """Create spine of specified dimensions and index"""
    RA = 1.0
    RM = 1.0
    CM = 0.01
    shaftDia = dia / 5.0
    sname = 'shaft' + str(index)
    hname = 'head' + str(index)
    shaft = moose.SymCompartment( parentObj.path + '/' + sname )
    moose.connect( parentCompt, 'CONNECTSPHERE', shaft, 'CONNECTSPHERE', 'Single' )
    x = parentCompt.x0 + frac * ( parentCompt.x - parentCompt.x0 )
    y = parentCompt.y0 + frac * ( parentCompt.y - parentCompt.y0 )
    z = parentCompt.z0 + frac * ( parentCompt.z - parentCompt.z0 )
    shaft.x0 = x
    shaft.y0 = y
    shaft.z0 = z
    sy = y + length * math.cos( theta * math.pi / 180.0 )
    sz = z + length * math.sin( theta * math.pi / 180.0 )
    shaft.x = x
    shaft.y = sy
    shaft.z = sz
    shaft.diameter = dia / 2.0
    shaft.length = length
    xa = math.pi * shaftDia * shaftDia / 4
    circumference = math.pi * shaftDia
    shaft.Ra = RA * length / xa
    shaft.Rm = RM / ( length * circumference )
    shaft.Cm = CM * length * circumference
    shaft.Em = EREST_ACT
    shaft.initVm = EREST_ACT

    head = moose.SymCompartment( parentObj.path + '/' + hname )
    moose.connect( shaft, 'CONNECTHEAD', head, 'CONNECTTAIL', 'Single' )
    head.x0 = x
    head.y0 = sy
    head.z0 = sz
    hy = sy + length * math.cos ( theta * math.pi / 180.0 )
    hz = sz + length * math.sin ( theta * math.pi / 180.0 )
    head.x = x
    head.y = hy
    head.z = hz
    head.diameter = dia
    head.length = length
    xa = math.pi * dia * dia / 4.0
    circumference = math.pi * dia
    head.Ra = RA * length / xa
    head.Rm = RM / ( length * circumference )
    head.Cm = CM * length * circumference
    head.Em = EREST_ACT
    head.initVm = EREST_ACT
    #print head.Rm, head.Ra, head.Cm, head.diameter, head.length
    #print shaft.Rm, shaft.Ra, shaft.Cm, shaft.diameter, shaft.length
    return head

def createSpineWithReceptor( compt, cell, index, frac ):
    FaradayConst = 96485.3415    # s A / mol
    spineLength = 5.0e-6
    spineDia = 4.0e-6
    head = createSpine( compt, cell, index, frac, spineLength, spineDia, 0.0 )
    gluR = moose.SynChan( head.path + '/gluR' )
    gluR.tau1 = 4e-3
    gluR.tau2 = 4e-3
    gluR.Gbar = 1e-6
    gluR.Ek= 10.0e-3
    moose.connect( head, 'channel', gluR, 'channel', 'Single' )

    caPool = moose.CaConc( head.path + '/ca' )
    caPool.CaBasal = 1e-4       # 0.1 micromolar
    caPool.tau = 0.01
    B = 1.0 / ( FaradayConst * spineLength * spineDia * spineDia *math.pi/4)
    B = B / 20.0                # scaling factor for Ca buffering
    caPool.B = B
    moose.connect( gluR, 'IkOut', caPool, 'current', 'Single' )

    return gluR

def addPlot( objpath, field, plot ):
    assert moose.exists( objpath )
    tab = moose.Table( '/graphs/' + plot )
    obj = moose.element( objpath )
    moose.connect( tab, 'requestData', obj, field )
    return tab

def makeElecPlots():
    graphs = moose.Neutral( '/graphs' )
    elec = moose.Neutral( '/graphs/elec' )
    addPlot( '/model/elec/compt', 'get_Vm', 'elec/dendVm' )
    #addPlot( '/n/compt/Na', 'get_Gbar', 'elec/NaGbar' )
    #addPlot( '/n/compt/K', 'get_Gbar', 'elec/KGbar' )
    #addPlot( '/n/compt/Na', 'get_Ik', 'elec/NaIk' )
    #addPlot( '/n/compt/K', 'get_Ik', 'elec/KIk' )
    #addPlot( '/n/compt/Na', 'get_Ek', 'elec/NaEk' )
    #addPlot( '/n/compt/K', 'get_Ek', 'elec/KEk' )
    addPlot( '/model/elec/head0', 'get_Vm', 'elec/head0Vm' )
    addPlot( '/model/elec/head2', 'get_Vm', 'elec/head2Vm' )
    #addPlot( '/n/head2', 'get_Im', 'elec/head2Im' )
    addPlot( '/model/elec/head2/ca', 'get_Ca', 'elec/head2Ca' )
    addPlot( '/model/elec/head0/gluR', 'get_Ik', 'elec/head0Ik' )
    addPlot( '/model/elec/head2/gluR', 'get_Ik', 'elec/head2Ik' )
    #addPlot( '/n/head0/gluR', 'get_Gk', 'elec/head0Gk' )
    #addPlot( '/n/head2/gluR', 'get_Gk', 'elec/head2Gk' )
    #addPlot( '/n/head2/gluR', 'get_Gbar', 'elec/head2Gbar' )

def dumpPlots( fname ):
    if ( os.path.exists( fname ) ):
        os.remove( fname )
    for x in moose.wildcardFind( '/graphs/##[ISA=Table]' ):
        moose.element( x[0] ).xplot( fname, x[0].name )

def makeSpinyCompt():
    comptLength = 100e-6
    comptDia = 4e-6
    numSpines = 5
    compt = createSquid()
    compt.inject = 0
    compt.x0 = 0
    compt.y0 = 0
    compt.z0 = 0
    compt.x = comptLength
    compt.y = 0
    compt.z = 0
    compt.length = comptLength
    compt.diameter = comptDia
    #kchan = moose.element( '/n/compt/K' )
    #kchan.Gbar = 0.2e-3
    synInput = moose.SpikeGen( '/n/compt/synInput' )
    synInput.refractT = 47e-3
    synInput.threshold = -1.0
    synInput.edgeTriggered = 0
    synInput.Vm( 0 )
    cell = moose.element( '/n' )
    for i in range( numSpines ):
        r = createSpineWithReceptor( compt, cell, i, i/float(numSpines) )
        r.synapse.num = 1
        syn = moose.element( r.path + '/synapse' )
        moose.connect( synInput, 'event', syn, 'addSpike', 'Single' )
        syn.weight = 0.2
        syn.delay = i * 1.0e-4

def createPool( compt, name, concInit ):
    meshEntries = moose.element( compt.path + '/mesh' )
    pool = moose.Pool( compt.path + '/' + name )
    moose.connect( pool, 'mesh', meshEntries, 'mesh', 'Single' )
    pool.concInit = concInit
    pool.diffConst = 1e-12
    return pool


def createChemModel( neuroCompt, spineCompt, psdCompt ):
    # Stuff in spine + psd
    psdGluR = createPool( psdCompt, 'psdGluR', 1 )
    headCa = createPool( spineCompt, 'Ca', 1e-4 )
    headGluR = createPool( spineCompt, 'headGluR', 2 )
    toPsd = createPool( spineCompt, 'toPsd', 0 )
    toPsdInact = createPool( spineCompt, 'toPsdInact', 1e-3 )
    turnOnPsd = moose.Reac( spineCompt.path +  '/turnOnPsd' )
    moose.connect( turnOnPsd, 'sub', headCa, 'reac', 'OneToOne' )
    moose.connect( turnOnPsd, 'sub', toPsdInact, 'reac', 'OneToOne' )
    moose.connect( turnOnPsd, 'prd', toPsd, 'reac', 'OneToOne' )
    turnOnPsd.Kf = 1e3
    turnOnPsd.Kb = 1
    toPsdEnz = moose.Enz( toPsd.path + '/enz' )
    toPsdEnzCplx = moose.Pool( toPsdEnz.path + '/cplx' )
    mesh = moose.element( spineCompt.path + '/mesh' )
    moose.connect( toPsdEnzCplx, 'mesh', mesh, 'mesh' )
    toPsdEnzCplx.concInit = 0
    moose.connect( toPsdEnz, 'enz', toPsd, 'reac', 'OneToOne' )
    moose.connect( toPsdEnz, 'sub', headGluR, 'reac', 'OneToOne' )
    moose.connect( toPsdEnz, 'prd', psdGluR, 'reac', 'OneToOne' )
    moose.connect( toPsdEnz, 'cplx', toPsdEnzCplx, 'reac', 'OneToOne' )
    toPsdEnz.Km = 1.0e-3
    toPsdEnz.kcat = 10.0
    fromPsd = moose.Reac( psdCompt.path + '/fromPsd' )
    moose.connect( fromPsd, 'sub', psdGluR, 'reac', 'OneToOne' )
    moose.connect( fromPsd, 'prd', headGluR, 'reac', 'OneToOne' )
    fromPsd.Kf = 0.5
    fromPsd.Kb = 0.0
    # Stuff in dendrite
    dendCa = createPool( neuroCompt, 'Ca', 1e-4 )
    bufCa = moose.Pool( neuroCompt.path + '/bufCa' )
    mesh = moose.element( neuroCompt.path + '/mesh' )
    moose.connect( mesh, 'mesh', bufCa, 'mesh', 'Single' )
    bufCa.concInit = 1e-4
    pumpCa = moose.Reac( neuroCompt.path + '/pumpCa' )
    moose.connect( pumpCa, 'sub', dendCa, 'reac', 'OneToOne' )
    moose.connect( pumpCa, 'prd', bufCa, 'reac', 'OneToOne' )
    pumpCa.Kf = 1
    pumpCa.Kb = 1
    dendKinaseInact = createPool( neuroCompt, 'inact_kinase', 1e-4 )
    dendKinase = createPool( neuroCompt, 'Ca.kinase', 0.0 )
    dendTurnOnKinase = moose.Reac( neuroCompt.path + '/turnOnKinase' )
    moose.connect( dendTurnOnKinase, 'sub', dendCa, 'reac' )
    moose.connect( dendTurnOnKinase, 'sub', dendKinaseInact, 'reac' )
    moose.connect( dendTurnOnKinase, 'prd', dendKinase, 'reac' )
    dendTurnOnKinase.Kf = 50000
    dendTurnOnKinase.Kb = 1
    dendKinaseEnz = moose.Enz( dendKinase.path + '/enz' )
    dendKinaseEnzCplx = moose.Pool( dendKinase.path + '/enz/cplx' )
    moose.connect( dendKinaseEnzCplx, 'mesh', mesh, 'mesh' )
    kChan = createPool( neuroCompt, 'kChan', 1e-3 )
    kChan_p = createPool( neuroCompt, 'kChan_p', 0.0 )
    moose.connect( dendKinaseEnz, 'enz', dendKinase, 'reac', 'OneToOne' )
    moose.connect( dendKinaseEnz, 'sub', kChan, 'reac', 'OneToOne' )
    moose.connect( dendKinaseEnz, 'prd', kChan_p, 'reac', 'OneToOne' )
    moose.connect( dendKinaseEnz, 'cplx', dendKinaseEnzCplx, 'reac', 'OneToOne' )
    dendKinaseEnz.Km = 1e-4
    dendKinaseEnz.kcat = 20
    dendPhosphatase = moose.Reac( neuroCompt.path + '/phosphatase' )
    moose.connect( dendPhosphatase, 'sub', kChan_p, 'reac' )
    moose.connect( dendPhosphatase, 'prd', kChan, 'reac' )
    dendPhosphatase.Kf = 1
    dendPhosphatase.Kb = 0.0


# Just for printf debugging
def printMolVecs( title ):
    nCa = moose.ematrix( '/model/chem/neuroMesh/Ca' )
    sCa = moose.ematrix( '/model/chem/spineMesh/Ca' )
    sR = moose.ematrix( '/model/chem/spineMesh/headGluR' )
    pR = moose.ematrix( '/model/chem/psdMesh/psdGluR' )
    print title
    print 'sizes: nCa, sCa, sR, pR = ', len(nCa), len(sCa), len(sR), len(pR) 
    #print 'nCa=', nCa.conc, ', sCa=', sCa.conc, ', sR=', sR.n, ', pR=', pR.n
    print 'nCaConcInit=', nCa.concInit, ', sCa=', sCa.concInit
    #print 'sRnInit=', sR.nInit, ', pR=', pR.nInit
    print 'sRconcInit=', sR.concInit, ', pR=', pR.concInit

    #print 'nCaSize=', nCa.size, ', sCa=', sCa.size, ', sR=', sR.n, ', pR=', pR.n

def makeChemInCubeMesh():
    dendSide = 10.8e-6
    spineSide = 6.8e-6
    psdSide = 8.565e-7
    parent = moose.Neutral ('/n' )
    neuroMesh = moose.CubeMesh( '/n/neuroMesh' )
    spineMesh = moose.CubeMesh( '/n/spineMesh' )
    psdMesh = moose.CubeMesh( '/n/psdMesh' )
    coords = [dendSide] * 9
    coords[0] = 0
    coords[1] = 0
    coords[2] = 0
    neuroMesh.coords = coords
    neuroMesh.preserveNumEntries = 1

    coords = [spineSide] * 9
    coords[0] = dendSide
    coords[1] = 0
    coords[2] = 0
    coords[3] = spineSide + dendSide
    spineMesh.coords = coords
    spineMesh.preserveNumEntries = 1

    coords = [psdSide] * 9
    coords[0] = dendSide + spineSide
    coords[1] = 0
    coords[2] = 0
    coords[3] = psdSide + spineSide + dendSide
    psdMesh.coords = coords
    psdMesh.preserveNumEntries = 1

    createChemModel( neuroMesh, spineMesh, psdMesh )
    dendCa = moose.element( '/n/neuroMesh/Ca' )
    assert dendCa.size == dendSide * dendSide * dendSide
    spineCa = moose.element( '/n/spineMesh/Ca' )
    assert spineCa.size == spineSide * spineSide * spineSide
    psdGluR = moose.element( '/n/psdMesh/psdGluR' )
    assert psdGluR.size == psdSide * psdSide * psdSide
    dendKinaseEnzCplx = moose.element( '/n/neuroMesh/Ca.kinase/enz/cplx' )
    assert dendKinaseEnzCplx.size == dendSide * dendSide * dendSide

    return parent

def makeCubeMultiscale():
    makeSpinyCompt()
    synInput = moose.element( '/n/compt/synInput' )
    synInput.refractT = 47e-3
    #Na = moose.element( '/n/compt/Na' )
    #Na.Gbar *= 1.5
    makeChemInCubeMesh()
    # set up diffusion between compts.
    headCa = moose.element( '/n/spineMesh/Ca' )
    dendCa = moose.element( '/n/neuroMesh/Ca' )
    diffReac = moose.Reac( '/n/spineMesh/diff' )
    moose.connect( diffReac, 'sub', headCa, 'reac' )
    moose.connect( diffReac, 'prd', dendCa, 'reac' )
    diffReac.Kf = 1
    diffReac.Kb = headCa.size / dendCa.size
    # set up adaptors
    adaptCa = moose.Adaptor( '/n/adaptCa' )
    elecCa = moose.element( '/n/head2/ca' )
    moose.connect( elecCa, 'concOut', adaptCa, 'input', 'OneToAll' )
    moose.connect( adaptCa, 'outputSrc', headCa, 'set_conc' )
    adaptCa.outputOffset = 0.0001    # 100 nM offset in chem.
    adaptCa.scale = 0.05             # 0.06 to 0.003 mM

    adaptGluR = moose.Adaptor( '/n/adaptGluR' )
    chemR = moose.element( '/n/psdMesh/psdGluR' )
    elec1R = moose.element( '/n/head1/gluR' )
    elec2R = moose.element( '/n/head2/gluR' )
    elec3R = moose.element( '/n/head3/gluR' )
    moose.connect( adaptGluR, 'requestField', chemR, 'get_n', 'OneToAll' )
    moose.connect( adaptGluR, 'outputSrc', elec1R, 'set_Gbar', 'OneToAll' )
    moose.connect( adaptGluR, 'outputSrc', elec2R, 'set_Gbar', 'OneToAll' )
    moose.connect( adaptGluR, 'outputSrc', elec3R, 'set_Gbar', 'OneToAll' )
    adaptGluR.scale = 1e-6 / 100     # from n to pS

    adaptK = moose.Adaptor( '/n/adaptK' )
    chemK = moose.element( '/n/neuroMesh/kChan' )
    elecK = moose.element( '/n/compt/K' )
    moose.connect( adaptK, 'requestField', chemK, 'get_conc', 'OneToAll' )
    moose.connect( adaptK, 'outputSrc', elecK, 'set_Gbar', 'OneToAll' )
    adaptK.scale = 0.3               # from mM to Siemens

def makeNeuroMeshModel():
    diffLength = 10e-6
    makeSpinyCompt()
    elec = moose.element( '/n' )
    elec.name = 'elec'
    model = moose.Neutral( '/model' )
    moose.move( elec, model )
    synInput = moose.element( '/model/elec/compt/synInput' )
    synInput.refractT = 47e-3

    chem = moose.Neutral( '/model/chem' )
    neuroCompt = moose.NeuroMesh( '/model/chem/neuroMesh' )
    neuroCompt.separateSpines = 1
    neuroCompt.diffLength = diffLength
    neuroCompt.geometryPolicy = 'cylinder'
    spineCompt = moose.SpineMesh( '/model/chem/spineMesh' )
    moose.connect( neuroCompt, 'spineListOut', spineCompt, 'spineList', 'OneToOne' )
    psdCompt = moose.PsdMesh( '/model/chem/psdMesh' )
    moose.connect( neuroCompt, 'psdListOut', psdCompt, 'psdList', 'OneToOne' )

    createChemModel( neuroCompt, spineCompt, psdCompt )

    # Put in the solvers, see how they fare.
    nmksolve = moose.GslStoich( '/model/chem/neuroMesh/ksolve' )
    print 'setting up nmksolve'
    nmksolve.path = '/model/chem/neuroMesh/##'
    print 'finished setting up nmksolve'
    nmksolve.compartment = moose.element( '/model/chem/neuroMesh' )
    nmksolve.method = 'rk5'
    nm = moose.element( '/model/chem/neuroMesh/mesh' )
    moose.connect( nm, 'remesh', nmksolve, 'remesh' )
    print "neuron: nv=", nmksolve.numLocalVoxels, ", nav=", nmksolve.numAllVoxels, nmksolve.numVarPools, nmksolve.numAllPools
    #
    printMolVecs( 'before addJunction neuron-spine' )

    print 'setting up smksolve'
    smksolve = moose.GslStoich( '/model/chem/spineMesh/ksolve' )
    smksolve.path = '/model/chem/spineMesh/##'
    smksolve.compartment = moose.element( '/model/chem/spineMesh' )
    smksolve.method = 'rk5'
    sm = moose.element( '/model/chem/spineMesh/mesh' )
    moose.connect( sm, 'remesh', smksolve, 'remesh' )
    print "spine: nv=", smksolve.numLocalVoxels, ", nav=", smksolve.numAllVoxels, smksolve.numVarPools, smksolve.numAllPools
    #
    print 'setting up pmksolve'
    pmksolve = moose.GslStoich( '/model/chem/psdMesh/ksolve' )
    pmksolve.path = '/model/chem/psdMesh/##'
    pmksolve.compartment = moose.element( '/model/chem/psdMesh' )
    pmksolve.method = 'rk5'
    pm = moose.element( '/model/chem/psdMesh/mesh' )
    moose.connect( pm, 'remesh', pmksolve, 'remesh' )

    print 'Assigning the cell model'
    # Now to set up the model.
    neuroCompt.cell = elec
    ns = neuroCompt.numSegments
    #assert( ns == 11 ) # dend, 5x (shaft+head)
    ndc = neuroCompt.numDiffCompts
    assert( ndc == 10 )
    ndc = neuroCompt.mesh.num
    assert( ndc == 10 )
    sdc = spineCompt.mesh.num
    assert( sdc == 5 )
    pdc = psdCompt.mesh.num
    assert( pdc == 5 )
    #
    #
    printMolVecs( 'before addJunction neuron-spine' )
    smksolve.addJunction( nmksolve )
    printMolVecs( 'after addJunction neuron-spine' )
    print "spine: nv=", smksolve.numLocalVoxels, ", nav=", smksolve.numAllVoxels, smksolve.numVarPools, smksolve.numAllPools
    pmksolve.addJunction( smksolve )
    printMolVecs( 'after addJunction spine-psd' )
    print "psd: nv=", pmksolve.numLocalVoxels, ", nav=", pmksolve.numAllVoxels, pmksolve.numVarPools, pmksolve.numAllPools
    # Have to pass a message between the various solvers.
    foo = moose.ematrix( '/model/chem/spineMesh/headGluR' )
    printMolVecs( 'before model setup ' )

    # oddly, numLocalFields does not work.
    ca = moose.element( '/model/chem/neuroMesh/Ca' )
    assert( ca.lastDimension == ndc )

    moose.ematrix( '/model/chem/spineMesh/headGluR' ).nInit = 100
    moose.ematrix( '/model/chem/psdMesh/psdGluR' ).nInit = 0
    #moose.ematrix( '/model/chem/psdMesh/psdGluR' ).nInit = 200
    printMolVecs( 'after setup, before addJunction' )

    #print ca
    #print "ns=", ns, ", ndc = ", ndc, ", sdc = ", sdc, ", pdc = ", pdc
    #print "nca=", ca.localNumField, ",lastDim = ", ca.lastDimension

    # set up adaptors
    """
    adaptCa = moose.Adaptor( '/model/chem/neuroMesh/adaptCa' )
    chemCa = moose.element( '/model/chem/neuroMesh/Ca' )
    elecCa = moose.element( '/model/elec/head2/ca' )
    moose.connect( elecCa, 'concOut', adaptCa, 'input', 'OneToOne' )
    moose.connect( adaptCa, 'outputSrc', chemCa, 'set_conc', 'OneToOne' )
    adaptCa.outputOffset = 0.0001    # 100 nM offset in chem.
    adaptCa.scale = 0.05             # 0.06 to 0.003 mM

    adaptGluR = moose.Adaptor( '/model/chem/psdMesh/adaptGluR' )
    chemR = moose.element( '/model/chem/psdMesh/psdGluR' )
    elec0R = moose.element( '/model/elec/head0/gluR' )
    elec1R = moose.element( '/model/elec/head1/gluR' )
    elec2R = moose.element( '/model/elec/head2/gluR' )
    elec3R = moose.element( '/model/elec/head3/gluR' )
    elec4R = moose.element( '/model/elec/head4/gluR' )
    moose.connect( adaptGluR, 'requestField', chemR, 'get_n', 'OneToAll' )
    moose.connect( adaptGluR, 'outputSrc', elec0R, 'set_Gbar', 'OneToAll' )
    moose.connect( adaptGluR, 'outputSrc', elec1R, 'set_Gbar', 'OneToAll' )
    moose.connect( adaptGluR, 'outputSrc', elec2R, 'set_Gbar', 'OneToAll' )
    moose.connect( adaptGluR, 'outputSrc', elec3R, 'set_Gbar', 'OneToAll' )
    moose.connect( adaptGluR, 'outputSrc', elec4R, 'set_Gbar', 'OneToAll' )
    adaptGluR.scale = 1e-6 / 100     # from n to pS
    """

    """
    adaptK = moose.Adaptor( '/n/adaptK' )
    chemK = moose.element( '/n/neuroMesh/kChan' )
    elecK = moose.element( '/n/compt/K' )
    moose.connect( adaptK, 'requestField', chemK, 'get_conc', 'OneToAll' )
    moose.connect( adaptK, 'outputSrc', elecK, 'set_Gbar', 'OneToAll' )
    adaptK.scale = 0.3               # from mM to Siemens
    """


def makeChemPlots():
    graphs = moose.Neutral( '/graphs' )
    addPlot( '/model/chem/spineMesh/Ca[0]', 'get_conc', 'spine0Ca' )
    addPlot( '/model/chem/spineMesh/Ca[2]', 'get_conc', 'spine2Ca' )
    addPlot( '/model/chem/neuroMesh/Ca[0]', 'get_conc', 'dend0Ca' )
    addPlot( '/model/chem/neuroMesh/Ca[3]', 'get_conc', 'dend3Ca' )
    addPlot( '/model/chem/neuroMesh/Ca[6]', 'get_conc', 'dend6Ca' )
    addPlot( '/model/chem/neuroMesh/Ca[9]', 'get_conc', 'dend9Ca' )
    addPlot( '/model/chem/neuroMesh/kChan_p[4]', 'get_conc', 'kChan_p' )
    addPlot( '/model/chem/neuroMesh/kChan[4]', 'get_conc', 'kChan' )
    addPlot( '/model/chem/neuroMesh/Ca.kinase[4]', 'get_conc', 'dendKinase' )
    addPlot( '/model/chem/spineMesh/toPsd[0]', 'get_conc', 'toPsd' )
    addPlot( '/model/chem/spineMesh/toPsd[2]', 'get_conc', 'toPsd' )
    addPlot( '/model/chem/spineMesh/toPsd[4]', 'get_conc', 'toPsd' )
    #addPlot( '/n/neuroMesh/Ca', 'get_conc', 'dendCa' )
    #addPlot( '/n/neuroMesh/inact_kinase', 'get_conc', 'inactDendKinase' )
    #addPlot( '/n/psdMesh/psdGluR', 'get_n', 'psdGluR' )

def testChemAlone():
    nid = makeChemInCubeMesh()
    moose.le( '/n' )
    makeChemPlots()
    moose.setClock( 5, 1e-2 )
    moose.setClock( 6, 1e-2 )
    moose.setClock( 7, 1.0 )
    moose.setClock( 8, 1.0 )
    moose.setClock( 9, 1.0 )
    moose.useClock( 5, '/n/##', 'init' )
    moose.useClock( 6, '/n/##', 'process' )
    #moose.useClock( 7, '/graphs/#', 'process' )
    moose.useClock( 8, '/graphs/#', 'process' )
    moose.reinit()
    moose.start( 100 )
    dumpPlots( 'chem.plot' )
    # Make ksolver and rerun.
    ksolve = moose.GslStoich( '/n/solver' )
    ksolve.path = '/n/##'
    ksolve.method = 'rk5'
    moose.useClock( 5, '/n/solver', 'process' )
    moose.setClock( 5, 1 )
    moose.setClock( 6, 1 )
    moose.reinit()
    moose.start( 100 )
    dumpPlots( 'kchem.plot' )

def testElecAlone():
    makeSpinyCompt()
    makeElecPlots()
    head2 = moose.element( '/n/head2' )
    kchan = moose.element( '/n/compt/K' )
    moose.setClock( 0, 2e-6 )
    moose.setClock( 1, 2e-6 )
    moose.setClock( 2, 2e-6 )
    moose.setClock( 8, 0.1e-3 )
    #moose.useClock( 0, '/n/#', 'init' )
    #moose.useClock( 1, '/n/#', 'process' )
    #moose.useClock( 2, '/n/#/#', 'process' )
    #print moose.wildcardFind( '/n/##[ISA=SpikeGen]' )
    moose.useClock( 0, '/n/##[ISA=Compartment]', 'init' )
    moose.useClock( 1, '/n/##[ISA=Compartment]', 'process' )
    moose.useClock( 2, '/n/##[ISA=ChanBase],/n/##[ISA=SynBase],/n/##[ISA=CaConc],/n/##[ISA=SpikeGen]','process')
    moose.useClock( 8, '/graphs/elec/#', 'process' )
    moose.reinit()
    moose.start( 0.1 )
    dumpPlots( 'instab.plot' )
    # make Hsolver and rerun
    hsolve = moose.HSolve( '/n/hsolve' )
    moose.useClock( 1, '/n/hsolve', 'process' )
    moose.setClock( 0, 2e-5 )
    moose.setClock( 1, 2e-5 )
    moose.setClock( 2, 2e-5 )
    hsolve.dt = 2e-5
    hsolve.target = '/n/compt'
    moose.reinit()
    #print kchan, ', Gbar = ', kchan.Gbar
    #kchan.Gbar = 0.1e-3
    #print 'Gbar = ', kchan.Gbar
    moose.start( 0.11 )
    dumpPlots( 'h_instab.plot' )

def testCubeMultiscale( useSolver ):
    elecDt = 10e-6
    chemDt = 1e-4
    plotDt = 5e-4
    plotName = 'symcm.plot'
    if ( useSolver ):
        elecDt = 50e-6
        chemDt = 2e-3
        plotName = 'solve_cm.plot'

    makeCubeMultiscale()
    makeChemPlots()
    makeElecPlots()
    moose.setClock( 0, elecDt )
    moose.setClock( 1, elecDt )
    moose.setClock( 2, elecDt )
    moose.setClock( 5, chemDt )
    moose.setClock( 6, chemDt )
    moose.setClock( 7, plotDt )
    moose.setClock( 8, plotDt )
    moose.useClock( 1, '/n/##[ISA=SpikeGen]', 'process' )
    moose.useClock( 2, '/n/##[ISA=SynBase]','process')
    moose.useClock( 6, '/n/##[ISA=Adaptor]', 'process' )
    moose.useClock( 7, '/graphs/#', 'process' )
    moose.useClock( 8, '/graphs/elec/#', 'process' )
    moose.useClock( 0, '/n/##[ISA=Compartment]', 'init' )
    moose.useClock( 1, '/n/##[ISA=Compartment]', 'process' )
    moose.useClock( 2, '/n/##[ISA=ChanBase],/n/##[ISA=SynBase],/n/##[ISA=CaConc]','process')
    moose.useClock( 5, '/n/##[ISA=PoolBase],/n/##[ISA=ReacBase],/n/##[ISA=EnzBase]', 'process' )
    if ( useSolver ):
        # Put in the solvers, see how they fare.
        ksolve = moose.GslStoich( '/n/ksolve' )
        ksolve.path = '/n/##'
        ksolve.method = 'rk5'
        moose.useClock( 5, '/n/ksolve', 'process' )
        hsolve = moose.HSolve( '/n/hsolve' )
        moose.useClock( 1, '/n/hsolve', 'process' )
        hsolve.dt = elecDt
        hsolve.target = '/n/compt'
    moose.reinit()
    moose.start( 1 )
    dumpPlots( plotName )

def testNeuroMeshMultiscale():
    elecDt = 50e-6
    chemDt = 2e-3
    plotDt = 5e-4
    plotName = 'nm.plot'

    makeNeuroMeshModel()
    chemCa = moose.element( '/model/chem/neuroMesh/Ca[0]' )
    chemCa.concInit = 0.001
    makeChemPlots()
    makeElecPlots()
    moose.setClock( 0, elecDt )
    moose.setClock( 1, elecDt )
    moose.setClock( 2, elecDt )
    moose.setClock( 5, chemDt )
    moose.setClock( 6, chemDt )
    moose.setClock( 7, plotDt )
    moose.setClock( 8, plotDt )
    moose.useClock( 0, '/model/elec/##[ISA=Compartment]', 'init' )
    moose.useClock( 1, '/model/elec/##[ISA=SpikeGen]', 'process' )
    moose.useClock( 2, '/model/elec/##[ISA=ChanBase],/n/##[ISA=SynBase],/n/##[ISA=CaConc]','process')
    moose.useClock( 5, '/model/chem/##[ISA=PoolBase],/n/##[ISA=ReacBase],/n/##[ISA=EnzBase]', 'process' )
    moose.useClock( 6, '/model/chem/##[ISA=Adaptor]', 'process' )
    moose.useClock( 7, '/graphs/#', 'process' )
    moose.useClock( 8, '/graphs/elec/#', 'process' )
    moose.useClock( 5, '/model/chem/#Mesh/ksolve', 'init' )
    moose.useClock( 6, '/model/chem/#Mesh/ksolve', 'process' )
    hsolve = moose.HSolve( '/model/elec/hsolve' )
    moose.useClock( 1, '/model/elec/hsolve', 'process' )
    hsolve.dt = elecDt
    hsolve.target = '/model/elec/compt'
    moose.reinit()
    printMolVecs( 'after reinit' )

    moose.start( 1.0 )
    dumpPlots( plotName )


def main():
    #testCubeMultiscale( 1 )
    #testElecAlone()
    testNeuroMeshMultiscale()

if __name__ == '__main__':
    main()

# 
# HsolveInstability.py ends here
