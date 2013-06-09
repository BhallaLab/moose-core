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

def create_squid():
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

def create_spine( parentCompt, parentObj, index, frac, length, dia, theta ):
    """Create spine of specified dimensions and index"""
    RA = 1.0
    RM = 1.0
    CM = 0.01
    shaftDia = dia / 5.0
    sname = 'shaft' + str(index)
    hname = 'head' + str(index)
    shaft = moose.SymCompartment( parentObj.path + '/' + sname )
    moose.connect( parentCompt, 'raxial2', shaft, 'raxial1', 'Single' )
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
    moose.connect( shaft, 'raxial2', head, 'raxial1', 'Single' )
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

def create_spine_with_receptor( compt, cell, index, frac ):
    FaradayConst = 96485.3415    # s A / mol
    spineLength = 5.0e-6
    spineDia = 4.0e-6
    head = create_spine( compt, cell, index, frac, spineLength, spineDia, 0.0 )
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

def add_plot( objpath, field, plot ):
    assert moose.exists( objpath )
    tab = moose.Table( '/graphs/' + plot )
    obj = moose.element( objpath )
    moose.connect( tab, 'requestData', obj, field )
    return tab

def make_elec_plots():
    graphs = moose.Neutral( '/graphs' )
    elec = moose.Neutral( '/graphs/elec' )
    add_plot( '/n/compt', 'get_Vm', 'elec/dendVm' )
    #add_plot( '/n/compt/Na', 'get_Gbar', 'elec/NaGbar' )
    #add_plot( '/n/compt/K', 'get_Gbar', 'elec/KGbar' )
    #add_plot( '/n/compt/Na', 'get_Ik', 'elec/NaIk' )
    #add_plot( '/n/compt/K', 'get_Ik', 'elec/KIk' )
    #add_plot( '/n/compt/Na', 'get_Ek', 'elec/NaEk' )
    #add_plot( '/n/compt/K', 'get_Ek', 'elec/KEk' )
    add_plot( '/n/head0', 'get_Vm', 'elec/head0Vm' )
    add_plot( '/n/head2', 'get_Vm', 'elec/head2Vm' )
    #add_plot( '/n/head2', 'get_Im', 'elec/head2Im' )
    add_plot( '/n/head2/ca', 'get_Ca', 'elec/head2Ca' )
    add_plot( '/n/head0/gluR', 'get_Ik', 'elec/head0Ik' )
    add_plot( '/n/head2/gluR', 'get_Ik', 'elec/head2Ik' )
    #add_plot( '/n/head0/gluR', 'get_Gk', 'elec/head0Gk' )
    #add_plot( '/n/head2/gluR', 'get_Gk', 'elec/head2Gk' )
    #add_plot( '/n/head2/gluR', 'get_Gbar', 'elec/head2Gbar' )

def dump_plots( fname ):
    if ( os.path.exists( fname ) ):
        os.remove( fname )
    for x in moose.wildcardFind( '/graphs/##[ISA=Table]' ):
        moose.element( x[0] ).xplot( fname, x[0].name )

def make_spiny_compt():
    comptLength = 100e-6
    comptDia = 4e-6
    numSpines = 5
    compt = create_squid()
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
        r = create_spine_with_receptor( compt, cell, i, i/float(numSpines) )
        r.synapse.num = 1
        syn = moose.element( r.path + '/synapse' )
        moose.connect( synInput, 'event', syn, 'addSpike', 'Single' )
        syn.weight = 0.2
        syn.delay = i * 1.0e-4

def create_pool( compt, name, concInit ):
    meshEntries = moose.element( compt.path + '/mesh' )
    pool = moose.Pool( compt.path + '/' + name )
    moose.connect( pool, 'mesh', meshEntries, 'mesh', 'Single' )
    pool.concInit = concInit
    return pool


def create_chem_model( nid, neuroCompt, spineCompt, psdCompt ):
    # Stuff in spine + psd
    psdGluR = create_pool( psdCompt, 'psdGluR', 0 )
    headCa = create_pool( spineCompt, 'Ca', 1e-4 )
    headGluR = create_pool( spineCompt, 'headGluR', 0 )
    headGluR.nInit = 100
    toPsd = create_pool( spineCompt, 'toPsd', 0 )
    toPsdInact = create_pool( spineCompt, 'toPsdInact', 1e-3 )
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
    dendCa = create_pool( neuroCompt, 'Ca', 1e-4 )
    bufCa = moose.BufPool( neuroCompt.path + '/bufCa' )
    mesh = moose.element( neuroCompt.path + '/mesh' )
    moose.connect( mesh, 'mesh', bufCa, 'mesh', 'Single' )
    bufCa.concInit = 1e-4
    pumpCa = moose.Reac( neuroCompt.path + '/pumpCa' )
    moose.connect( pumpCa, 'sub', dendCa, 'reac', 'OneToOne' )
    moose.connect( pumpCa, 'prd', bufCa, 'reac', 'OneToOne' )
    pumpCa.Kf = 1
    pumpCa.Kb = 1
    dendKinaseInact = create_pool( neuroCompt, 'inact_kinase', 1e-4 )
    dendKinase = create_pool( neuroCompt, 'Ca.kinase', 0.0 )
    dendTurnOnKinase = moose.Reac( neuroCompt.path + '/turnOnKinase' )
    moose.connect( dendTurnOnKinase, 'sub', dendCa, 'reac' )
    moose.connect( dendTurnOnKinase, 'sub', dendKinaseInact, 'reac' )
    moose.connect( dendTurnOnKinase, 'prd', dendKinase, 'reac' )
    dendTurnOnKinase.Kf = 50000
    dendTurnOnKinase.Kb = 1
    dendKinaseEnz = moose.Enz( dendKinase.path + '/enz' )
    dendKinaseEnzCplx = moose.Pool( dendKinase.path + '/enz/cplx' )
    moose.connect( dendKinaseEnzCplx, 'mesh', mesh, 'mesh' )
    kChan = create_pool( neuroCompt, 'kChan', 1e-3 )
    kChan_p = create_pool( neuroCompt, 'kChan_p', 0.0 )
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


def make_chem_in_cube_mesh():
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

    create_chem_model( parent, neuroMesh, spineMesh, psdMesh )
    dendCa = moose.element( '/n/neuroMesh/Ca' )
    assert dendCa.size == dendSide * dendSide * dendSide
    spineCa = moose.element( '/n/spineMesh/Ca' )
    assert spineCa.size == spineSide * spineSide * spineSide
    psdGluR = moose.element( '/n/psdMesh/psdGluR' )
    assert psdGluR.size == psdSide * psdSide * psdSide
    dendKinaseEnzCplx = moose.element( '/n/neuroMesh/Ca.kinase/enz/cplx' )
    assert dendKinaseEnzCplx.size == dendSide * dendSide * dendSide

    return parent

def make_cube_multiscale():
    make_spiny_compt()
    synInput = moose.element( '/n/compt/synInput' )
    synInput.refractT = 47e-3
    #Na = moose.element( '/n/compt/Na' )
    #Na.Gbar *= 1.5
    make_chem_in_cube_mesh()
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


def make_chem_plots():
    graphs = moose.Neutral( '/graphs' )
    add_plot( '/n/spineMesh/Ca', 'get_conc', 'spineCa' )
    add_plot( '/n/neuroMesh/Ca', 'get_conc', 'dendCa' )
    add_plot( '/n/neuroMesh/kChan_p', 'get_conc', 'kChan_p' )
    add_plot( '/n/neuroMesh/kChan', 'get_conc', 'kChan' )
    add_plot( '/n/neuroMesh/Ca.kinase', 'get_conc', 'dendKinase' )
    add_plot( '/n/spineMesh/toPsd', 'get_conc', 'toPsd' )
    #add_plot( '/n/neuroMesh/Ca', 'get_conc', 'dendCa' )
    #add_plot( '/n/neuroMesh/inact_kinase', 'get_conc', 'inactDendKinase' )
    #add_plot( '/n/psdMesh/psdGluR', 'get_n', 'psdGluR' )

def test_chem_alone():
    nid = make_chem_in_cube_mesh()
    moose.le( '/n' )
    make_chem_plots()
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
    dump_plots( 'chem.plot' )
    # Make ksolver and rerun.
    ksolve = moose.GslStoich( '/n/solver' )
    ksolve.path = '/n/##'
    ksolve.method = 'rk5'
    moose.useClock( 5, '/n/solver', 'process' )
    moose.setClock( 5, 1 )
    moose.setClock( 6, 1 )
    moose.reinit()
    moose.start( 100 )
    dump_plots( 'kchem.plot' )

def test_elec_alone():
    make_spiny_compt()
    make_elec_plots()
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
    dump_plots( 'instab.plot' )
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
    dump_plots( 'h_instab.plot' )

def test_cube_multiscale( useSolver ):
    elecDt = 10e-6
    chemDt = 1e-4
    plotDt = 5e-4
    plotName = 'symcm.plot'
    if ( useSolver ):
        elecDt = 50e-6
        chemDt = 2e-3
        plotName = 'solve_cm.plot'

    make_cube_multiscale()
    make_chem_plots()
    make_elec_plots()
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
    dump_plots( plotName )


def main():
    test_cube_multiscale( 1 )
    #test_elec_alone()

if __name__ == '__main__':
    main()

# 
# HsolveInstability.py ends here
