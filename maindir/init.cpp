/*******************************************************************
 * File:            init.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-09-25 15:38:08
 ********************************************************************/
#include "init.h"
#include <utility/Configuration.h>
#include <scheduling/ClockJob.h>
using namespace std;

// Most of this code is a compilation from ClockJob.cpp and main.cpp
// written by Upi and Niraj.
int mooseInit(std::string configFile)
{
    static Configuration conf(configFile);    
    const Cinfo* c = Cinfo::find( "Shell" );
    assert ( c != 0 );
    const Finfo* childSrc = Element::root()->findFinfo( "childSrc" );
    assert ( childSrc != 0 );
    Element* shell = c->create( Id( 1 ), "shell" );
    assert( shell != 0 );
    bool ret = childSrc->add( Element::root(), shell, 
                              shell->findFinfo( "child" ) );
    assert( ret );

#ifdef USE_MPI
// 	Element* shell =
// 			Neutral::create( "Shell", "shell", Element::root() );
	MPI::Init( argc, argv );
	unsigned int totalnodes = MPI::COMM_WORLD.Get_size();
	mynode = MPI::COMM_WORLD.Get_rank();
	Id::manager().setNodes( mynode, totalnodes );

	Element* postmasters =
			Neutral::create( "Neutral", "postmasters", Element::root());
	vector< Element* > post;
	post.reserve( totalnodes );
	for ( unsigned int i = 0; i < totalnodes; i++ ) {
		char name[10];
		if ( i != mynode ) {
			sprintf( name, "node%d", i );
			Element* p = Neutral::create(
					"PostMaster", name, postmasters );
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
        Neutral::create( "Neutral", "sched", Element::root() );
    // This one handles the simulation clocks
    Element* cj =
        Neutral::create( "ClockJob", "cj", sched );
    // TODO: Put the solver creation here

    
    Neutral::create( "Neutral", "library", Element::root() );
    Neutral::create( "Neutral", "proto", Element::root() );
    Element* solvers = 
            Neutral::create( "Neutral", "solvers", Element::root() );
    // These two should really be solver managers because there are
    // a lot of decisions to be made about how the simulation is best
    // solved. For now let the Shell deal with it.
    Neutral::create( "Neutral", "chem", solvers );
    Neutral::create( "Neutral", "neuronal", solvers );
        
    if ( conf.properties[Configuration::CREATESOLVER].compare("true") == 0 )
    {
        cout << "Creating solvers...." << endl;
        
        initSolvers(cj);        
    }
    else 
    {
        cout << "No solvers created" << endl;
    }
    
#ifdef USE_MPI
	// This one handles parser and postmaster scheduling.
	Element* pj =
			Neutral::create( "ClockJob", "pj", sched );
	Element* t0 =
			Neutral::create( "ParTick", "t0", cj );
	Element* pt0 =
			Neutral::create( "ParTick", "t0", pj );
#else

    // Not really honouring AUTOSCHEDULE setting -
    // May need only t0 for AUTOSCHEDULE=false
    // But creating a few extra clock ticks does not hurt as much as
    // not allowing user to change the clock settings
    Element* t0 = Neutral::create( "Tick", "t0", cj );
    Element* t1 = Neutral::create( "Tick", "t1", cj );
    Element* t2 = Neutral::create( "Tick", "t2", cj );
    Element* t3 = Neutral::create( "Tick", "t3", cj );
    Element* t4 = Neutral::create( "Tick", "t4", cj );
    Element* t5 = Neutral::create( "Tick", "t5", cj );
//    setupDefaultSchedule(t0,t1,t2,t3,t4,t5,cj); // cannot put here - dt causes failure in SchedTest
#endif
    return 0;    
}

// This is insufficient - we have to create the solvers at reset time - see discussion with Niraj and Raamesh
void initSolvers(Element *clockJob)
{
    vector<Id> childList = Neutral::getChildList(clockJob);
    
    static const Cinfo* tickCinfo = Cinfo::find( "Tick" );
    assert ( tickCinfo != 0 );
    static const Finfo* procFinfo = tickCinfo->findFinfo( "process" );
    assert ( procFinfo != 0 );
    static Id ksolvers( "/solvers/chem" );
    static Id nsolvers( "/solvers/neuronal" );
    // This is a really dumb first pass on solver assignment,
    // puts all chem stuff onto a single solver
	
    if ( childList.size() != 6 ) // This happens during unit tests.
        return;
    // assert( childList.size() == 6 );

    if ( procFinfo->numOutgoing( childList[2]() ) > 0 && 
         procFinfo->numOutgoing( childList[3]() ) > 0 ) {
        // Set up kinetic solver
        Id ksolve( "/solvers/chem/stoich" );
        if ( !ksolve.bad() ) { // alter existing ksolve
            set( ksolve(), "scanTicks" );
        } else { // make a whole new set.
            Element* ki = Neutral::create( "GslIntegrator", "integ",
                                           ksolvers() );
            if ( ki == 0 ) { // GslIntegrator class does not exist
                cout << "ClockJob::checkSolvers: Warning: GslIntegrator does not exist.\nReverting to Exp Euler method\n";
                return;
            }
            Element* ks = Neutral::create( "Stoich", "stoich", ksolvers() );
            Element* kh = Neutral::create( "KineticHub", "hub", ksolvers());
            ks->findFinfo( "hub" )->add( ks, kh, kh->findFinfo( "hub" ) );
            ks->findFinfo( "gsl" )->add( ks, ki, ki->findFinfo( "gsl" ) );
            set< string >( ki, "method", "rk5" );
            set< double >( ki, "relativeAccuracy", 1.0e-5 );
            set< double >( ki, "absoluteAccuracy", 1.0e-5 );
            set( ks, "scanTicks" );
            childList[4]()->findFinfo( "process" )->add( 
                childList[4](), ki, ki->findFinfo( "process" ) );
        }
    }
    
    // Similar stuff here for hsolver.
    // childList[0] is clock tick 0
    if ( procFinfo->numOutgoing( childList[0]() ) > 0 ) {
        // Set up neuronal solver
        Id nsolve( "/solvers/neuronal/integ" );
        if ( !nsolve.bad() )
        { // alter existing nsolve
            set( nsolve(), "scanTicks" );
        }
        else
        { // make a whole new set.
            Element* ni = Neutral::create( "HSolve", "integ",
                                           nsolvers() );
            set( ni, "scanTicks" );
 
            childList[4]()->findFinfo( "process" )->add( childList[4](), ni, ni->findFinfo( "process" ) );
        }
    }
}
