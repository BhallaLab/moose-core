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

MUSIC::setup* Music::setup_ = 0;
MUSIC::runtime* Music::runtime_ = 0;
//!!!
double Music::dt_;

const Cinfo* initMusicCinfo()
{
	/**
	 * This is a shared message to receive Process messages from
	 * the scheduler objects.
	 */
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
		sizeof( processShared ) / sizeof( Finfo* ) );
	
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
	
//	static SchedInfo schedInfo[] = { { process, 0, 1 } };
	
	static Cinfo musicCinfo(
		"Music",
		"Niraj Dudani and Johannes Hjorth",
		"Moose Music object for communciation with the MUSIC API",
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
  runtime_->tick();
  //~ cerr << "TICK! " << p->currTime_ << endl;
  //~ cerr << "Music time: " << runtime_->time() << endl;
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
//    runtime_ = new MUSIC::runtime(setup_, p->dt_ );
   // setup_ = 0;
  }

}

void Music::reinitializeFunc( const Conn* c ) 
{
  static_cast < Music* > (c->data() )->innerReinitializeFunc();
}

void Music::innerReinitializeFunc( ) 
{
  if(setup_) {
    runtime_ = new MUSIC::runtime(setup_, 0.01 );
    //~ runtime_ = new MUSIC::runtime(setup_, dt_ );
    setup_ = 0;
	// !!!
	cerr << "dt_ hardcoded" << endl;
  }
}

MPI::Intracomm Music::setup( int& argc, char**& argv )
{
  setup_ = new MUSIC::setup( argc, argv );

  return setup_->communicator();
}

void Music::finalizeFunc( const Conn* c )
{
	static_cast< Music* >( c->data() )->innerFinalizeFunc(c->target());
}

void Music::innerFinalizeFunc( Eref e )
{
	delete runtime_;
	runtime_ = 0;
}

void Music::addPort (
	const Conn* c,
	string name,
	string direction,
	string type ) 
{
  static_cast < Music* > ( c->data() )->innerAddPort(c->target(), 
                                                     name, direction, type );
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

    // Publish the event input port to music
    MUSIC::event_input_port* mPort = setup_->publish_event_input(name);
    unsigned int width = mPort->width();
    
    unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
    unsigned int myRank = MuMPI::INTRA_COMM().Get_rank();

    // Calculate base offset and width for our process
    // last node gets any extra channels left.
    unsigned int avgWidth = width / numNodes;
    unsigned int myWidth = (myRank < numNodes-1) ? 
      avgWidth : width - avgWidth*(numNodes-1);

    unsigned int myOffset = myRank * avgWidth;

    set< unsigned int >(port,"initialise", myWidth, myOffset, mPort);
    
    // Map the input from MUSIC to data channels local to this process
    // is done in InputEventPort

  }
  else if(direction == "out" && type == "event"){
     // Create the event output port
    Element* port = 
      Neutral::create("OutputEventPort", name, e.id(), Id::scratchId() );

    // Publish the event output port to music
    MUSIC::event_output_port* mPort = setup_->publish_event_output(name);
    unsigned int width = mPort->width();
    
    unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
    unsigned int myRank = MuMPI::INTRA_COMM().Get_rank();

    // Calculate base offset and width for our process
    // last node gets any extra channels left.
    unsigned int avgWidth = width / numNodes;
    unsigned int myWidth = (myRank < numNodes-1) ? 
      avgWidth : width - avgWidth*(numNodes-1);

    unsigned int myOffset = myRank * avgWidth;

    set< unsigned int >(port,"initialise", myWidth, myOffset, mPort);
    
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
