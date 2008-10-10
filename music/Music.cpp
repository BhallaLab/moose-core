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
	//////////////////////////////////////////////////////////////////
	// SharedFinfos
	//////////////////////////////////////////////////////////////////
		process,
	//////////////////////////////////////////////////////////////////
	// Dest Finfos.
	//////////////////////////////////////////////////////////////////
		new DestFinfo(
			"setup",
			Ftype1< MUSIC::setup* >::global(),
			RFCAST( &Music::setupFunc ) ),
		new DestFinfo(
			"finalize",
			Ftype0::global(),
			RFCAST( &Music::finalizeFunc ) ),
		new DestFinfo(
			"addPort", 
			Ftype3< string, string, string >::global(),
			RFCAST( &Music::addPort ) ),
	};
	
	static SchedInfo schedInfo[] = { { process, 0, 1 } };
	
	static Cinfo musicCinfo(
		"Music",
		"Niraj Dudani and Johannes Hjorth",
		"Moose Music object for communciation with the MUSIC API",
		initNeutralCinfo(),
		musicFinfos,
		sizeof( musicFinfos ) / sizeof( Finfo* ),
		ValueFtype1< Music >::global(),
		schedInfo, 1
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

	cerr << "Music::innerProcessFunc not implemented yet" << endl;
}

void Music::processFunc( const Conn* c, ProcInfo p ) 
{
	static_cast < Music* > (c->data() )->innerProcessFunc( c, p );
}
  
void Music::reinitFunc( const Conn* c, ProcInfo p ) 
{
	;
}

void Music::setupFunc( const Conn* c, MUSIC::setup* setup )
{
	static_cast< Music* >( c->data() )->
		innerSetupFunc( c->target(), setup );
}

void Music::innerSetupFunc( Eref e, MUSIC::setup* setup )
{
	setup_ = setup;
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

  Element* port = 
    Neutral::create("InputEventPort", name, e.id(), Id::scratchId() );

  MUSIC::cout_output_port* out = setup_->publish_event_input(name);
  int width = out->width();
    
  set< unsigned int >(port,"width", width);

  }
  else {
    cerr << "Music::innerAddPort: " << direction " " << type 
         << " Not supported yet";

  }
}
