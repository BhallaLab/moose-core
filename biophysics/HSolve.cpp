/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment.
 **           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#include "moose.h"
#include "../element/Wildcard.h"
#include "HSolveStructure.h"
#include "NeuroHub.h"
#include "NeuroScanBase.h"
#include "NeuroScan.h"
#include "HSolveBase.h"
#include "HSolve.h"
#include "ThisFinfo.h"

const Cinfo* initHSolveCinfo()
{
	static Finfo* processShared[] =
	{
            new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                           RFCAST( &HSolve::processFunc ) ),
            new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                           RFCAST( &HSolve::reinitFunc ) ),
	};
        
    static Finfo* process = new SharedFinfo( "process", processShared,
                                             sizeof( processShared ) / sizeof( Finfo* ) );
        
    /** Shared message to NeuroScan:
     *  (Src) hubCreate: Sends solver element.
     *                   Solver delegates hub creation to the scan element.
     *  (Src) readModel: Sends 'seed' compartment, dt.
     *                   This message is a request to read in the model.
     *                   The seed compartment is the starting point for
     *                   reading in the model.
     */
    static Finfo* scanShared[] =
	{
            new SrcFinfo( "hubCreate",
                          Ftype1< Element* >::global() ),
            new SrcFinfo( "readModel",
                          Ftype2< Element*, double >::global() ),
	};
	
    static Finfo* hsolveFinfos[] = 
	{
            //////////////////////////////////////////////////////////////////
            // Field definitions
            //////////////////////////////////////////////////////////////////
            new ValueFinfo( "path", ValueFtype1< string >::global(),
                            GFCAST( &HSolve::getPath ),
                            RFCAST( &HSolve::setPath )
		),
            new ValueFinfo( "NDiv", ValueFtype1< int >::global(),
                            GFCAST( &HSolve::getNDiv ),
                            RFCAST( &HSolve::setNDiv )
		),
            new ValueFinfo( "VLo", ValueFtype1< double >::global(),
                            GFCAST( &HSolve::getVLo ),
                            RFCAST( &HSolve::setVLo )
		),
            new ValueFinfo( "VHi", ValueFtype1< double >::global(),
                            GFCAST( &HSolve::getVHi ),
                            RFCAST( &HSolve::setVHi )
		),
		
            //////////////////////////////////////////////////////////////////
            // SharedFinfo definitions
            //////////////////////////////////////////////////////////////////
            new SharedFinfo( "scan", scanShared,
                             sizeof( scanShared ) / sizeof( Finfo* ) ),
            process,
                
		
            //////////////////////////////////////////////////////////////////
            // MsgSrc definitions
            //////////////////////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////
            // DestFinfo definitions
            //////////////////////////////////////////////////////////////////
            new DestFinfo( "postCreate", Ftype0::global(),
                           &HSolve::postCreateFunc ),
            new DestFinfo( "scanTicks", Ftype0::global(),
                           &HSolve::scanTicksFunc ),
	};
    // Schedule it to clock tick 2, stage 0
    static SchedInfo schedInfo[] = {{ process, 2, 0 }};
        
        
                    
    static Cinfo hsolveCinfo(
        "HSolve",
        "Niraj Dudani, 2007, NCBS",
        "HSolve: Hines solver, for solving branching neuron models.",
        initNeutralCinfo(),
        hsolveFinfos,
        sizeof( hsolveFinfos ) / sizeof( Finfo* ),
        ValueFtype1< HSolve >::global(),
        schedInfo,
        1
	);

    return &hsolveCinfo;
}

static const Cinfo* hsolveCinfo = initHSolveCinfo();

static const unsigned int scanSlot =
initHSolveCinfo()->getSlotIndex( "scan" );
static const unsigned int hubCreateSlot =
initHSolveCinfo()->getSlotIndex( "scan.hubCreate" );
static const unsigned int readModelSlot =
initHSolveCinfo()->getSlotIndex( "scan.readModel" );

static const Finfo* scanNDivFinfo =
initNeuroScanCinfo()->findFinfo( "NDiv" );
static const Finfo* scanVLoFinfo =
initNeuroScanCinfo()->findFinfo( "VLo" );
static const Finfo* scanVHiFinfo =
initNeuroScanCinfo()->findFinfo( "VHi" );
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void HSolve::setPath( const Conn& c, string path )
{
    static_cast< HSolve* >( c.data() )->
        innerSetPath( c.targetElement(), path );
}

void HSolve::innerSetPath( Element* e, const string& path )
{
    vector< Element* > elist;
    int nFound = simpleWildcardFind( path, elist );
    if ( nFound == 0 )
        cerr << "Error: Invalid Path.\n";
    else if ( nFound > 1 )
        cerr << "Error: Path should point to exactly 1 object.\n";
    else {
        Element* seed = elist[ 0 ];
        const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >(
            seed->getThisFinfo() );
		
        /* Check here if element is being solved already */
		
        if( !tf->cinfo()->isA( Cinfo::find( "Compartment" ) ) )
            cerr << "Error: Path should point to object of type Compartment.\n";
        else {
            path_ = path;
            seed_ = seed;
        }
    }
}

string HSolve::getPath( const Element* e )
{
    return static_cast< const HSolve* >( e->data() )->path_;
}

/** Lookup table specifics (NDiv, VLo, VHi) actually are fields on NeuroScan.
 *  Here we provide port-holes to access the same.
 */
void HSolve::setNDiv( const Conn& c, int NDiv )
{
    HSolve* solve = static_cast< HSolve* >( c.data() );
    set< int >( solve->scanElm_, scanNDivFinfo, NDiv );
}

int HSolve::getNDiv( const Element* e )
{
    int NDiv;
    HSolve* solve = static_cast< HSolve* >( e->data() );
    get< int >( solve->scanElm_, scanNDivFinfo, NDiv );
    return NDiv;
}

/** Lookup table specifics (NDiv, VLo, VHi) actually are fields on NeuroScan.
 *  Here we provide port-holes to access the same.
 */
void HSolve::setVLo( const Conn& c, double VLo )
{
    HSolve* solve = static_cast< HSolve* >( c.data() );
    set< double >( solve->scanElm_, scanVLoFinfo, VLo );
}

double HSolve::getVLo( const Element* e )
{
    double VLo;
    HSolve* solve = static_cast< HSolve* >( e->data() );
    get< double >( solve->scanElm_, scanVLoFinfo, VLo );
    return VLo;
}

/** Lookup table specifics (NDiv, VLo, VHi) actually are fields on NeuroScan.
 *  Here we provide port-holes to access the same.
 */
void HSolve::setVHi( const Conn& c, double VHi )
{
    HSolve* solve = static_cast< HSolve* >( c.data() );
    set< double >( solve->scanElm_, scanVHiFinfo, VHi );
}

double HSolve::getVHi( const Element* e )
{
    double VHi;
    HSolve* solve = static_cast< HSolve* >( e->data() );
    get< double >( solve->scanElm_, scanVHiFinfo, VHi );
    return VHi;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HSolve::processFunc( const Conn&c, ProcInfo p )
{
    static_cast< HSolve* >( c.data() )->
        innerProcessFunc();
}

void HSolve::innerProcessFunc( )
{
    if ( seed_ == 0 )
        return;
    step();
}

void HSolve::reinitFunc( const Conn& c, ProcInfo p )
{
    static_cast< HSolve* >( c.data() )->
        innerReinitFunc( c.targetElement(), p );
}

void HSolve::innerReinitFunc( Element* e, const ProcInfo& p )
{
    if ( seed_ == 0 )
        return;
    unsigned int scanConn =
        e->connSrcBegin( readModelSlot ) -
        e->lookupConn( 0 );
    sendTo2< Element*, double >( e, readModelSlot, scanConn,
                                 seed_, p->dt_ );
}

/** Scan element is created as child, which in turn
 *  creates hub as its sibling.
 */
void HSolve::postCreateFunc( const Conn& c )
{
    static_cast< HSolve* >( c.data() )->
        innerPostCreateFunc( c.targetElement() );
}

void HSolve::innerPostCreateFunc( Element* e )
{
    // Scan element's data field is owned by its parent HSolve
    // structure, so we set it's noDelFlag to 1.
    scanElm_ = initNeuroScanCinfo()->create( 
        Id::scratchId(), "scan",
        static_cast< void* >( &scanData_ ), 1 );
    e->findFinfo( "childSrc" )->
        add( e, scanElm_, scanElm_->findFinfo( "child" ) );
	
    // Setting up shared msg between solver and scanner.
    e->findFinfo( "scan" )->
        add( e, scanElm_, scanElm_->findFinfo( "scan" ) );
	
    unsigned int scanConn =
        e->connSrcBegin( scanSlot ) -
        e->lookupConn( 0 );
    sendTo1< Element* >( e, hubCreateSlot, scanConn, e );
}

void HSolve::scanTicksFunc( const Conn& c )
{
    static_cast< HSolve* >( c.data() )->innerScanTicksFunc( );
}

void HSolve::innerScanTicksFunc( )
{
    static Id t0( "/sched/cj/t0" );
    static const Finfo* processFinfo = t0()->findFinfo( "process" );
    assert( !t0.bad() );
    assert( processFinfo != 0 );
	
    vector< Conn > list;
    vector< Conn >::iterator i;
	
    processFinfo->outgoingConns( t0(), list );
    for ( i = list.begin(); i != list.end(); i++ ) {
        Element* el = i->targetElement();
        if( el->cinfo()->isA( Cinfo::find( "Compartment" ) ) ) {
            seed_ = el;
            break;
        }
    }
}
