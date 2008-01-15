/*******************************************************************
 * File:            init.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-09-25 15:38:08
 ********************************************************************/
#include "init.h"
#include <utility/utility.h>
#include <scheduling/ClockJob.h>

using namespace std;

// Most of this code is a compilation from ClockJob.cpp and main.cpp
// written by Upi and Niraj.
int mooseInit()
{
    static bool initialized = false;
    if(initialized)
    {
        return 0;
    }
    else
    {
        initialized = true;
    }
    
#ifdef CRL_MPI
    const Cinfo* c = Cinfo::find( "ParShell" );
#else
    const Cinfo* c = Cinfo::find( "Shell" );
#endif

    assert ( c != 0 );
    const Finfo* childSrc = Element::root()->findFinfo( "childSrc" );
    assert ( childSrc != 0 );
    	/// \todo Check if this can be replaced with the Neutral::create
    Element* shell = c->create( Id( 1 ), "shell" );
    assert( shell != 0 );
    bool ret = childSrc->add( Element::root(), shell, 
                              shell->findFinfo( "child" ) );
    assert( ret );

#ifdef USE_MPI
    
	MPI::Init( argc, argv );
	unsigned int totalnodes = MPI::COMM_WORLD.Get_size();
	mynode = MPI::COMM_WORLD.Get_rank();
	Id::manager().setNodes( mynode, totalnodes );

	Element* postmasters =
			Neutral::create( "Neutral", "postmasters", Element::root(), Id::scratchId());
	vector< Element* > post;
	post.reserve( totalnodes );
	for ( unsigned int i = 0; i < totalnodes; i++ ) {
		char name[10];
		if ( i != mynode ) {
			sprintf( name, "node%d", i );
			Element* p = Neutral::create(
					"PostMaster", name, postmasters, Id::scratchId() );
			assert( p != 0 );
			set< unsigned int >( p, "remoteNode", i );
			post.push_back( p );
		}
	}
	Id::manager().setPostMasters( post );
	// Perhaps we will soon want to also connect up the clock ticks.
	// How do we handle different timesteps?
#else	
//	Neutral::create( "Shell", "shell", Element::root() );
#endif
    /**
     * Here we set up a bunch of predefined objects, that
     * exist simultaneously on each node.
     */
    Element* sched =
        Neutral::create( "Neutral", "sched", Element::root(), Id::scratchId() );
    // This one handles the simulation clocks
    Element* cj =
        Neutral::create( "ClockJob", "cj", sched, Id::scratchId() );
    
    Neutral::create( "Neutral", "library", Element::root(), Id::scratchId() );
    Neutral::create( "Neutral", "proto", Element::root(), Id::scratchId() );
    Element* solvers = 
            Neutral::create( "Neutral", "solvers", Element::root(), Id::scratchId() );
    // These two should really be solver managers because there are
    // a lot of decisions to be made about how the simulation is best
    // solved. For now let the Shell deal with it.
    Neutral::create( "Neutral", "chem", solvers, Id::scratchId() );
    Neutral::create( "Neutral", "neuronal", solvers, Id::scratchId() );

    
#ifdef USE_MPI
	// This one handles parser and postmaster scheduling.
	Element* pj =
			Neutral::create( "ClockJob", "pj", sched, Id::scratchId() );
	Element* t0 =
			Neutral::create( "ParTick", "t0", cj, Id::scratchId() );
	Element* pt0 =
			Neutral::create( "ParTick", "t0", pj, Id::scratchId() );
#else

    // Not really honouring AUTOSCHEDULE setting -
    // May need only t0 for AUTOSCHEDULE=false
    // But creating a few extra clock ticks does not hurt as much as
    // not allowing user to change the clock settings
    Neutral::create( "Tick", "t0", cj, Id::scratchId() );
    Neutral::create( "Tick", "t1", cj, Id::scratchId() );
#endif
    return 0;    
}

void setupDefaultSchedule( 
	Element* t0, Element* t1, Element* cj)
{
	set< double >( t0, "dt", 1e-2 );
	set< double >( t1, "dt", 1e-2 );
	set< int >( t1, "stage", 1 );
	set( cj, "resched" );
	set( cj, "reinit" );
}
