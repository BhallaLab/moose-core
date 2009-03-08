/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
**             and Niraj Dudani and Johannes Hjorth, KTH, Stockholm
**
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <mpi.h>
#include <music.hh>
#include "maindir/MuMPI.h"
#include "Music.h"
#include "element/Neutral.h"

MUSIC::Setup* Music::setup_ = 0;
MUSIC::Runtime* Music::runtime_ = 0;
//!!!
double Music::dt_;
double Music::stopTime_; 

const Cinfo* initMusicCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo(
			"process", Ftype1< ProcInfo >::global(),
			RFCAST( &Music::processFunc ) ),
		new DestFinfo(
			"reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &Music::reinitFunc ) ),
	};
	
	static Finfo* process = new SharedFinfo(
		"process",
		processShared,
		sizeof( processShared ) / sizeof( Finfo* ),
		" This is a shared message to receive Process messages from the scheduler objects." );
	
	static Finfo* musicFinfos[] =
	{
	//////////////////////////////////////////////////////////////////
	// Field definitions
	//////////////////////////////////////////////////////////////////
        new ValueFinfo( "rank",
                        ValueFtype1< int >::global(),
                        GFCAST( &Music::getRank ),
                        &dummyFunc 
                        ),
        new ValueFinfo( "size",
                        ValueFtype1< int >::global(),
                        GFCAST( &Music::getSize ),
                        &dummyFunc 
                        ),
        new ValueFinfo( "stoptime",
                        ValueFtype1< double >::global(),
                        GFCAST( &Music::getStopTime ),
                        &dummyFunc 
                        ),

	//////////////////////////////////////////////////////////////////
	// SharedFinfos
	//////////////////////////////////////////////////////////////////
		process,
	//////////////////////////////////////////////////////////////////
	// Dest Finfos.
	//////////////////////////////////////////////////////////////////
		new DestFinfo(
			"reinitialize",
			Ftype0::global(),
			RFCAST( &Music::reinitializeFunc ) ),
		new DestFinfo(
			"finalize",
			Ftype0::global(),
			RFCAST( &Music::finalizeFunc ) ),
		new DestFinfo(
			"addPort", 
			Ftype3< string, string, string >::global(),
			RFCAST( &Music::addPort ) ),
	};
	
	//~ static SchedInfo schedInfo[] = { { process, 0, 1 } };
	
	static string doc[] =
	{
		"Name", "Music",
		"Author", "Niraj Dudani and Johannes Hjorth",
		"Description", "Moose Music object for communciation with the MUSIC API",
	};

	static Cinfo musicCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		musicFinfos,
		sizeof( musicFinfos ) / sizeof( Finfo* ),
		ValueFtype1< Music >::global()
		//~ schedInfo, 1
	);
	
	return &musicCinfo;
}

static const Cinfo* musicCinfo = initMusicCinfo();

//////////////////////////////////////////////////////////////////
// Field access functions
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// Message dest functions.
//////////////////////////////////////////////////////////////////
void Music::innerProcessFunc( const Conn* c, ProcInfo p ) 
{
  //~ cerr << "TICK! " << p->currTime_ << endl;
  //~ cerr << "Music time: " << runtime_->time() << endl;
  
  if ( p->currTime_ <= stopTime_ )
  	runtime_->tick();
}

void Music::processFunc( const Conn* c, ProcInfo p ) 
{
	static_cast < Music* > (c->data() )->innerProcessFunc( c, p );
}
  
void Music::reinitFunc( const Conn* c, ProcInfo p ) 
{
  static_cast < Music* > (c->data() )->innerReinitFunc(c->target(),p);
}

void Music::innerReinitFunc( Eref e, ProcInfo p ) 
{
  if(setup_) {
    
	dt_ = p->dt_;
    runtime_ = new MUSIC::Runtime(setup_, p->dt_ );
    setup_ = 0;
  }

}

void Music::reinitializeFunc( const Conn* c ) 
{
  static_cast < Music* > (c->data() )->innerReinitializeFunc();
}

void Music::innerReinitializeFunc( ) 
{
  if(setup_) {
    //~ runtime_ = new MUSIC::runtime(setup_, 0.01 );
    //~ runtime_ = new MUSIC::runtime(setup_, dt_ );
    //~ setup_ = 0;
	// !!!
	//~ cerr << "dt_ hardcoded" << endl;
  }
}

MPI::Intracomm Music::setup( int& argc, char**& argv )
{
  setup_ = new MUSIC::Setup( argc, argv );

  // Store the MUSIC stop time
  setup_->config ("stoptime", &stopTime_);

  return setup_->communicator();
}

void Music::finalizeFunc( const Conn* c )
{
	static_cast< Music* >( c->data() )->innerFinalizeFunc(c->target());
}

void Music::innerFinalizeFunc( Eref e )
{
	// cerr << "Music time: " << runtime_->time() << endl;
	if ( runtime_ ) {
		runtime_->finalize();
		delete runtime_;
	}
	runtime_ = 0;
}

void Music::addPort (
	const Conn* c,
	string direction,
	string type,
	string name )
{
  static_cast < Music* > ( c->data() )->innerAddPort(c->target(), 
                                                     direction, type, name );
}

void Music::innerAddPort (
	Eref e,
	string direction,
	string type,
	string name ) 
{
  if(direction == "in" && type == "event") {

    // Create the event input port
    Element* port = 
      Neutral::create("InputEventPort", name, e.id(), Id::scratchId() );
	port->id().setGlobal();

    // Publish the event input port to music
    MUSIC::EventInputPort* mPort = setup_->publishEventInput(name);

    unsigned int width = mPort->width();
    
    unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
    unsigned int myRank = MuMPI::INTRA_COMM().Get_rank();

    // Calculate base offset and width for our process
    // last node gets any extra channels left.
    unsigned int avgWidth = width / numNodes;
    unsigned int myWidth = (myRank < numNodes-1) ? 
      avgWidth : width - avgWidth*(numNodes-1);

    unsigned int myOffset = myRank * avgWidth;

    set< unsigned int, unsigned int, MUSIC::EventInputPort* >(
		port,"initialise", myWidth, myOffset, mPort );
    
    // Map the input from MUSIC to data channels local to this process
    // is done in InputEventPort

  }
  else if(direction == "out" && type == "event"){
     // Create the event output port
    Element* port = 
      Neutral::create("OutputEventPort", name, e.id(), Id::scratchId() );
	port->id().setGlobal();

    // Publish the event output port to music
    MUSIC::EventOutputPort* mPort = setup_->publishEventOutput(name);
    unsigned int width = mPort->width();
    
    unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
    unsigned int myRank = MuMPI::INTRA_COMM().Get_rank();

    // Calculate base offset and width for our process
    // last node gets any extra channels left.
    unsigned int avgWidth = width / numNodes;
    unsigned int myWidth = (myRank < numNodes-1) ? 
      avgWidth : width - avgWidth*(numNodes-1);

    unsigned int myOffset = myRank * avgWidth;

    set< unsigned int, unsigned int, MUSIC::EventOutputPort* >(
    	port,"initialise", myWidth, myOffset, mPort );
    
    // Map the output from MUSIC to data channels local to this process
    // is done in OutputEventPort

  }
  else {
    cerr << "Music::innerAddPort: " << direction << " " << type 
         << " Not supported yet";

  }
}


int Music::getRank( Eref e ) {
  return MuMPI::INTRA_COMM().Get_rank();
}

int Music::getSize( Eref e) {
  return MuMPI::INTRA_COMM().Get_size();
}

double Music::getStopTime( Eref e) {
  return stopTime_;
}
