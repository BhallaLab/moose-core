"""./camkii_in_moose.py

See http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1069645/

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import params as p
import moose
import moose.utils as mu
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from camkii import CaMKII


class Args: pass 
args = Args()

print( '[INFO] Using moose from %s' % moose.__file__ )

# global data-structures
model_path_ = '/model'
compt_path_ = None
molecules_ = {}
tables_ = {}
compt_ = None

N = 6.023e23

try:
    plt.style.use('presentation')
except Exception as e:
    pass

def conc_to_n( conc ):
    val = conc * float( args['volume'] ) * N 
    assert val > 0.0
    return val

def inv_conc_to_N( conc ):
    return conc / (float(args['volume']) * N )

def init( ):
    global compt_
    global args
    moose.Neutral( model_path_ )
    compt_ = moose.CubeMesh( compt_path_ )
    compt_.nx = 10
    compt_.ny = 10
    compt_.nz = 10
    print('[INFO] Volume of PSD: %s' % args['volume'])
    compt_.volume = float( args['volume'] )
    for st in range(7):
        camkii = CaMKII( st )
        molecules_[camkii.name] = camkii.create_moose_pool(compartment = compt_path_)
        add_table( molecules_[camkii.name], 'N' )

    for c in [ 'I1', 'CaMKII*', 'PP1', 'PP1*', 'PP1_active', 'I1P',
            'INITCamKII', 'turnover' ]:
        if c in [ 'I1', 'PP1', 'INITCamKII']:
            molecules_[c] = moose.BufPool( '%s/%s' % (compt_path_, c) )
        else:
            molecules_[c] = moose.Pool( '%s/%s' % (compt_path_, c ) )
        add_table( molecules_[c], 'N' )

    print('[INFO] CaMKII = %s' % (2*args['camkii']))
    molecules_['x0y0'].nInit = 2 * int(args['camkii'])
    molecules_['CaMKII*'].nInit = 0
    molecules_['INITCamKII'].nInit = 2 * int(args['camkii'])
    print('[INFO] Total PP1 %s' % args['pp1'])
    molecules_['PP1'].nInit = int(args['pp1'])

    # TODO: Much of it only make sense if stochastic solver is used.
    molecules_['I1'].concInit =  p.conc_i1_free
    pp1CamKIICplx = moose.Pool( '%s/camkii_pp1_cplx' % compt_path_ )
    pp1CamKIICplx.concInit = 0
    molecules_['pp1_camkii_cplx'] = pp1CamKIICplx

def add_table( moose_elem, field):
    tablePath = '%s/table-%s' % (moose_elem.path, field)
    if moose.exists( tablePath ):
        return None
    t = moose.Table2( tablePath )
    moose.connect( t, 'requestOut', moose_elem, 'get'+field[0].upper()+field[1:])
    tables_[ '%s.%s' % (moose_elem.name, field.upper() )] = t

def ca_input( ):
    """Input calcium pulse

    When using stochastic solver, any event less than 100 or so seconds would
    not work.
    """
    global args
    # Input calcium
    ca = moose.BufPool( '%s/ca' % compt_path_ )
    ca.concInit = args['ca_low']
    print("[INFO] Setting up input calcium : init = %s" % args['ca_low'] )
    add_table( ca, 'conc' )
    molecules_['ca'] = ca

    # mu.info("Baselevel ca conc = %s" % p.resting_ca_conc)
    concFunc = moose.Function( "%s/cafunc" % ca.path )
    if args['protocol'] == 'ltp':
        mu.info('Using LTP protocol')
        expr = '(t > 1000 && t < 1005)?{0}:{1}'.format(
                args['ca_high'], args['ca_low']
                )
    elif args['protocol'] == 'basal':
        mu.info('Using basal protocol. Base level with small fluctualtions')
        expr = '(sin(13*t) > 0.95)?{0}:{1}'.format(args['ca_low'], args['ca_high'])

    else:
        mu.error( 'Protocol %s is not supported yet' % args['protocol'] )
        quit()

    concFunc.expr = expr
    moose.connect( concFunc, 'valueOut', ca, 'setConc' )
    mu.info("Using function %s to update Ca conc == " % concFunc.expr )

def attach_one_more_pp1( camkii ):
    # Add a single molecule of PP1 to camkii.
    frm = CaMKII(camkii[0], camkii[1])
    frmPool = frm.create_moose_pool( compt_path_ )
    molecules_[frmPool.name] = frmPool
    to = CaMKII(camkii[0], camkii[1] + 1 )
    toPool = to.create_moose_pool( compt_path_ )
    molecules_[toPool.name] = toPool
    r = moose.Reac( '%s/attach_pp1' % (frmPool.path))
    moose.connect( r, 'sub', frmPool, 'reac')
    moose.connect( r, 'sub', molecules_['PP1_active'], 'reac')
    moose.connect( r, 'prd', toPool, 'reac' )
    r.numKf = p.k_2 / conc_to_n(p.K_M)
    r.numKb = 0
    # mu.info("Phosphorylation: %s = %s => %s" % (frmPool, r.Kf, toPool))
    return to

def dephosphorylate( camkii ):
    pp1ToRemove = camkii.nPP1 
    # mu.info("Total PP1s to remove %s" % pp1ToRemove)
    for npp1 in range(pp1ToRemove):
        c = CaMKII( camkii.nP - npp1, camkii.nPP1 - npp1 )
        frmPool = c.create_moose_pool( compt_path_ )
        to = CaMKII( camkii.nP - npp1 - 1, camkii.nPP1 -  npp1 - 1)
        toPool = to.create_moose_pool( compt_path_ )
        # mu.info( "Dephosphorylation: %s --> %s" % (frmPool.name, toPool.name))
        r = moose.Reac( '%s/detach_pp1' % frmPool.path )
        moose.connect( r, 'sub', frmPool, 'reac' )
        moose.connect( r, 'prd', toPool, 'reac' )
        moose.connect( r, 'prd', molecules_['PP1'], 'reac' )
        r.numKf = p.k_2
        r.numKb = 0.0
        add_table( frmPool, 'N' )
        add_table( toPool, 'N' )
        ## And the turnover leads to.
        # add_turnover( frmPool )

def deactivation_by_pp1( nP ):
    print("[INFO] CaMKII with %s subunits phosphorylated" % nP )
    nPP1s = range( 0, nP ) 
    for nPP1 in nPP1s:
        camkii = CaMKII(nP, nPP1) 
        molecules_[camkii.name] = camkii.create_moose_pool( compt_path_ )
    statesCaMKIIWithPP1 = [ (nPP1+1, x) for x in nPP1s ]
    camkiiWithPP1s = map(attach_one_more_pp1, statesCaMKIIWithPP1)
    for camkii in camkiiWithPP1s:
        # Each of these camkii with PP1 attached also have dephosphorylation
        # step.
        dephosphorylate( camkii )

def i1_to_i1p( ):
    global compt_path_
    f = moose.Function( '%s/i1_to_i1p' % molecules_['I1P'].path )
    f.x.num = 2
    f.mode = 2
    f.expr = 'x1*(1+(x0/{kh2})^3)/(x0/{kh2})^3'.format(kh2=conc_to_n(p.K_H2))
    moose.connect(  molecules_['ca'], 'nOut', f.x[0], 'input' )
    moose.connect(   molecules_['I1'], 'nOut', f.x[1], 'input' )
    moose.connect( f, 'valueOut',  molecules_['I1P'], 'setN' )
    mu.info('I1P = %s' % ( f.expr ))

def pp1_activation( ):
    """Here we setup the reactions PP1
    """
    global compt_path_ 

    i1_to_i1p( )

    # Now a fraction of PP1 is active PP1. The value of fraction depends on the
    # value of I1P. I can not set it up like a reaction I1P + PP1 <----> I1P.PP1
    # because I1P is a BufPool and all of PP1 will turn into complex. I need a
    # Function to calculate the fraction of PP1 free from I1P.
    # I1P and PP1 associates and dissociates.

    f = moose.Function( '%s/active_pp1' % molecules_['PP1*'].path )
    f.x.num = 2
    f.expr = 'x0/(1+(x1*{k3}/{k4}))'.format(k3=inv_conc_to_N(p.k_3), k4=p.k_4) 
    moose.connect(  molecules_['PP1'], 'nOut', f.x[0], 'input' )
    moose.connect(  molecules_['I1P'], 'nOut', f.x[1], 'input' )
    moose.connect( f, 'valueOut', molecules_['PP1*'], 'setN' )
    mu.info('[PP1*] = %s ' % f.expr )

    # Let's have a simple reaction PP1* <---1, 1 ---> PP1_active. This way we
    # PP1_active can have step jump ot 1. PP1* is not a chemical pool per se
    # since it has fractional numbers.
    r = moose.Reac( '%s/pp1_start_pp1_active' % compt_path_)
    moose.connect( r, 'sub', molecules_['PP1*'], 'reac' )
    moose.connect( r, 'prd', molecules_['PP1_active'], 'reac')
    r.Kf = 1.0
    r.Kb = 1.0

    # PP1.Sp + I1P <-- -- --> PP1.Sp.I1P
    r = moose.Reac( '%s/reac_pp1sp_i1p' % compt_path_)
    r.Kf = inv_conc_to_N( p.k_3 )
    r.Kb = p.k_4
    mu.info('PP1.Sp  <- %s, %s -> PP1.Sp.I1P' % ( r.Kb, r.Kf ))
    moose.connect( r, 'sub', molecules_['pp1_camkii_cplx'], 'reac')
    moose.connect( r, 'sub', molecules_['PP1_active'], 'reac')
    cplx = moose.Pool( '%s/PP1CaMKII1P' % compt_path_ )
    cplx.nInit = 0
    molecules_['pp1_camkii_i1p_cplx']  = cplx
    moose.connect( r, 'prd', molecules_['pp1_camkii_i1p_cplx'], 'reac')

def camkii_activation_deactivation( ):
    """All reaction involved in activating CaMKII"""
        
    # CaMKII to x1y0 is slow. 
    r0 = moose.Reac( '%s/reac_0to1' % compt_path_ )
    moose.connect( r0, 'sub', molecules_['x0y0'], 'reac' )
    moose.connect( r0, 'prd', molecules_['x1y0'], 'reac' )
    # The rate expression of this reaction depends on Ca++ 
    f = moose.Function( '%s/func_kf' % r0.path  )
    f.expr = '{k1}*(x0/{kh1})^6/(1 + (x0/{kh1})^3)^2'.format(
            k1 = p.k_1
            , kh1= conc_to_n(p.K_H1)
            )
    f.x.num = 1
    moose.connect( molecules_['ca'], 'nOut', f.x[0], 'input' )
    moose.connect( f, 'valueOut', r0, 'setNumKf' )
    print("[INFO] Connecting %s to reaction %s" % ( molecules_['ca'], r0))
    r0.Kb = 0
       
    # First reaction which is slow phosphorylate only 1 subunit of holoenzyme
    # Phosphorylation of CaMKII occurs in various stages: 
    # x0y0 -> x1y0 -> x2y0 -> x3y0 -> x4y0 -> x5y0 -> CaMKII*
    for pp in range(1, 6):
        frm, to = 'x%sy0' % pp, 'x%sy0' % (pp + 1)
        # mu.info("Setting step %s -> %s" % (frm, to))
        r = moose.Reac( '%s/reac_%s_%s' % (compt_path_, frm, to))
        moose.connect( r, 'sub', molecules_[frm], 'reac')
        moose.connect( r, 'prd', molecules_[to], 'reac')
        # And its rate is controlled by another function
        f = moose.Function( '%s/funcKf' % r.path  )
        f.expr = '{k1}*(x0/{kh1})^3/(1 + (x0/{kh1})^3)'.format(
                k1 = p.k_1, kh1= conc_to_n(p.K_H1)
                )
        f.x.num = 1
        moose.connect(  molecules_['ca'], 'nOut', f.x[0], 'input' )
        moose.connect( f, 'valueOut', r, 'setNumKf' )
        r.Kb = 0

    for nP in range(1, 7):
        deactivation_by_pp1( nP )

    #  Count all subunits which are phosphorylated.
    f = moose.Function( '%s/func_sum_all' % compt_path_)
    f.expr = '(x0+2*x1+3*x2+4*x3+5*x4+6*x5)/6'
    f.x.num = len( f.expr.split('+') )
    for i in range( f.x.num ):
        moose.connect( molecules_['x%sy0' % (i+1)], 'nOut', f.x[i], 'input')
    moose.connect( f, 'valueOut', molecules_['CaMKII*'], 'setN' )


def add_turnover( camkii ):
    # Added turnover of CaMKII 
    r = moose.Reac( '%s/turnover_reac' % camkii.path )
    moose.connect( r, 'sub', camkii, 'reac' )
    # NOTE: Not sure if PP1_active is produced or PP1.
    moose.connect( r, 'prd', molecules_['PP1'], 'reac' )
    # TODO: verify Kf or numKf ?
    r.numKf =  p.turnover_rate_holoenzyme
    r.Kb = 0.0
    mu.info('PP1.Sp = {0} => PP1 + x0y0'.format(r.Kf))

def make_model( ):
    camkii_activation_deactivation( )
    pp1_activation( )

def setup_solvers( stochastic ):
    global args
    stoich = moose.Stoich( "%s/stoich" % compt_path_)
    if stochastic:
        print("[INFO] Setting up Stochastic solver")
        s = moose.Gsolve('%s/gsolve' % compt_path_)
        #s.useClockedUpdate = True
    else:
        s = moose.Ksolve('%s/ksolve' % compt_path_)

    stoich.ksolve = s
    stoich.compartment = moose.element(compt_.path)
    stoich.path = '%s/##' % compt_path_

    print('compt_path = %s' %compt_path_)
    print('stoich = %s' %stoich)
    #print('stoich.dataIndex = %s' %stoich.dataIndex)
    print('stoich.path = %s' %stoich.path)

def add_more_plots( ):
    global tables_
    vec = tables_['CaMKII*.N'].vector / (12 * args['camkii'] )  
    class Table: pass
    table = Table()
    table.vector = vec
    tables_['CaMKII.frac'] = table


def is_close( val, ref, rtol):
    if abs(val - ref)/float(ref) <= rtol:
        return True
    return False

def assert_close( val, ref, rtol):
    if is_close( val, ref, rtol ):
        return True
    mu.log('FAILED'
            , "Expected %s, got %s, relative tolerance: %s" % (ref, val, rtol)
            )
    return False

def get_params_from_file( ):
    parameters = []
    for v in vars(p):
        if '__' in v:
            continue
        else:
            parameters.append( '%s=%s' % (v, vars(p)[v]))
    return ",".join(parameters)

def process_data( ):
    global tables_
    global args
    time = moose.Clock('/clock').currentTime 
    camkii = tables_['CaMKII*.N'].vector
    camkii6 = tables_['x6y0.N'].vector
    camkiiFrac = tables_['CaMKII.frac'].vector
    timeVec = np.linspace(0, time, len(camkii))
    plt.subplot(3, 1, 1)
    # plt.plot( timeVec, camkiiFrac, label = 'CaMKII*' )
    plt.plot( timeVec, tables_['x0y0.N'].vector, label = 'x0y0', alpha = 0.3)
    plt.plot( timeVec, tables_['x6y0.N'].vector, label = 'x6y0', alpha = 0.3)
    plt.plot( timeVec, camkii, '.')
    plt.xlabel( 'Time (seconds)' )
    plt.ylabel('#Phosphorylated subunits')
    plt.legend(loc='best', framealpha=0.4)
    plt.subplot( 3, 1, 2)
#    plt.hist( camkii, bins = int(max(camkii)) )
    plt.xlabel( '#Phosphorylated subunits' )
    plt.ylabel( 'Count' )
    plt.subplot(3, 1, 3)
    plt.plot( timeVec, tables_['ca.CONC'].vector )
    title = ''
    for k in [ 'camkii', 'pp1', 'volume' ]:
        title += '%s:%s, ' % (str(args[k]), k)
    plt.suptitle( title, fontsize = 8 )

    filename = title.translate(None, ', ')
    filename += '%s_ca_low,%s_ca_high' % (args['ca_low'], args['ca_high'])
    outfile = '_images/params_%s.png' % filename
    plt.tight_layout( )
    plt.savefig( outfile )
    print('[INFO] Done saving figure to %s' % outfile)
    plt.savefig('output.png')
    
def main( compt_name = 'switch1', **kwargs ):
    global args
    global compt_path_
    compt_path_ = '%s/%s' % (model_path_, compt_name )
    args = kwargs
    init( )
    ca_input( )
    moose.seed(0)
    make_model()
    setup_solvers( stochastic = True )
    moose.setClock(18, 20) # Table2 
    moose.reinit( )
    print('[INFO] Running for %s' % args['simtime'])
    t1 = time.time()
    moose.start( args['simtime'] )
    print( '[INFO] Total time taken %s' % (time.time() - t1 ) )
    add_more_plots( )
    comment = get_params_from_file()
    mu.saveRecords( tables_ , outfile = args['outfile'], comment = comment )
    process_data()

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = 'CaMKII/PP1 switch. Most parameters are in params.py file.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--ca_high', '-ch'
        , required = False
        , default = p.ca_pulse
        , type = float
        , help = 'Calcium pulse (high)'
        )

    parser.add_argument('--ca_low', '-cl'
        , required = False
        , default = p.resting_ca_conc
        , type = float
        , help = 'Calcium pulse (low)'
        )

    parser.add_argument('--simtime', '-st'
        , required = False
        , default = p.run_time
        , type = int
        , help = 'Run time for simulation.'
        )

    parser.add_argument('--outfile', '-o'
        , required = False
        , default = '%s.dat' % sys.argv[0]
        , help = 'Outfile to save the data in (csv) file.'
        )

    parser.add_argument('--camkii', '-ck'
        , required = False
        , default = p.N_CaMK
        , type = int
        , help = 'No of CaMKII molecules'
        )

    parser.add_argument('--pp1', '-pp'
        , required = False
        , default = p.N_PP1
        , help = 'No of PP1 molecules.'
        )

    parser.add_argument('--volume', '-vol'
        , required = False
        , type = str
        , default = str(p.volume)
        , help = 'Volume in m^3'
        )

    parser.add_argument('--protocol', '-p'
        , required = False
        , type = str
        , default = 'basal'
        , help = 'Protocol to use [ltp/basal]'
        )

    parser.parse_args( namespace = args )
    main( 'camkii_pp1_0', **vars(args) )
