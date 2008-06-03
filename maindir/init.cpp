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

	/**
	 * This function puts the FuncVecs in order and must be called
	 * after static initialization but before any messaging
	 */
	FuncVec::sortFuncVec();
    
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
	bool ret = Eref::root().add( "childSrc", shell, "child" );
    assert( ret );

    /**
     * Here we set up a bunch of predefined objects, that
     * exist simultaneously on each node.
     */
    Element* sched =
        Neutral::create( "Neutral", "sched", Element::root()->id(), Id::scratchId() );
    // This one handles the simulation clocks
    Element* cj =
        Neutral::create( "ClockJob", "cj", sched->id(), Id::scratchId() );
    
    Neutral::create( "Neutral", "library", Element::root()->id(), Id::scratchId() );
    Neutral::create( "Neutral", "proto", Element::root()->id(), Id::scratchId() );
    Element* solvers = 
            Neutral::create( "Neutral", "solvers", Element::root()->id(), Id::scratchId() );
    // These two should really be solver managers because there are
    // a lot of decisions to be made about how the simulation is best
    // solved. For now let the Shell deal with it.
    Neutral::create( "Neutral", "chem", solvers->id(), Id::scratchId() );
    Neutral::create( "Neutral", "neuronal", solvers->id(), Id::scratchId() );

    // Not really honouring AUTOSCHEDULE setting -
    // May need only t0 for AUTOSCHEDULE=false
    // But creating a few extra clock ticks does not hurt as much as
    // not allowing user to change the clock settings
    Neutral::create( "Tick", "t0", cj->id(), Id::scratchId() );
    Neutral::create( "Tick", "t1", cj->id(), Id::scratchId() );
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
