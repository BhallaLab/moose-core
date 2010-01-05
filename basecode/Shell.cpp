/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <pthread.h>
#include "header.h"
#include "Shell.h"
#include "Dinfo.h"

// Want to separate out this search path into the Makefile options
#include "../scheduling/Tick.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"

ProcInfo Shell::p_;

const Cinfo* Shell::initCinfo()
{
	/*
	static Finfo* reacFinfos[] = {
		new Finfo( setKf_ ),
		new Finfo( setKb_ ),
	};
	*/
	static Finfo* shellFinfos[] = {
		new ValueFinfo< Shell, string >( 
			"name",
			"Name of object", 
			&Shell::setName, 
			&Shell::getName ),

		new ValueFinfo< Shell, bool >( 
			"quit",
			"Flag to tell the system to quit", 
			&Shell::setQuit, 
			&Shell::getQuit ),
////////////////////////////////////////////////////////////////
		new DestFinfo( "handleGet", 
			"Function to handle returning values for 'get' calls.",
			new RetFunc< Shell >( &Shell::handleGet ) ),
		new DestFinfo( "start", 
			"Starts off a simulation for the specified run time, automatically partitioning among threads if the settings are right",
			new OpFunc1< Shell, double >( & Shell::start ) ),
		new DestFinfo( "setclock", 
			"Assigns clock ticks. Args: tick#, dt, stage",
			new OpFunc3< Shell, unsigned int, double, unsigned int >( & Shell::setclock ) ),
		new DestFinfo( "loadBalance", 
			"Set up load balancing",
			new OpFunc0< Shell >( & Shell::loadBalance ) ),

////////////////////////////////////////////////////////////////
		new SrcFinfo1< FuncId >( "requestGet",
			"Function to request another Element for a value", 0 ),
			
	};

	static Cinfo shellCinfo (
		"Shell",
		0, // No base class.
		shellFinfos,
		sizeof( shellFinfos ) / sizeof( Finfo* ),
		new Dinfo< Shell >()
	);

	return &shellCinfo;
}

static const Cinfo* shellCinfo = Shell::initCinfo();


Shell::Shell()
	: name_( "" ),
		quit_( 0 ), 
		isSingleThreaded_( 0 ), numCores_( 1 ), numNodes_( 1 )
{
	;
}

void Shell::process( const ProcInfo* p, const Eref& e )
{
	quit_ = 1;
}


void Shell::setName( string name )
{
	name_ = name;
}

string Shell::getName() const
{
	return name_;
}

void Shell::setQuit( bool val )
{
	quit_ = val;
}

bool Shell::getQuit() const
{
	return quit_;
}

void Shell::handleGet( Eref e, const Qinfo* q, const char* arg )
{
	getBuf_.resize( q->size() );
	memcpy( &getBuf_[0], arg, q->size() );
	// Instead of deleting and recreating the msg, it could be a 
	// permanent msg on this object, reaching out whenever needed
	// to targets.
}

const char* Shell::getBuf() const
{
	if ( getBuf_.size() > 0 )
		return &( getBuf_[0] );
	return 0;
}
////////////////////////////////////////////////////////////////////////
// Some static utility functions
////////////////////////////////////////////////////////////////////////

// Statid func for returning the pet ProcInfo of the shell.
const ProcInfo* Shell::procInfo()
{
	return &p_;
}

/**
 * Static global, returns contents of shell buffer.
 */
const char* Shell::buf() 
{
	static Id shellid;
	static Element* shell = shellid();
	assert( shell );
	return (reinterpret_cast< Shell* >(shell->data( 0 )) )->getBuf();
}

// Function to assign hardware availability
void Shell::setHardware( 
	bool isSingleThreaded, unsigned int numCores, unsigned int numNodes )
{
	isSingleThreaded_ = isSingleThreaded;
	Qinfo::addSimGroup( 1 ); // This is the parser thread.
	if ( !isSingleThreaded ) {
		// Create the parser and the gui threads.
		numCores_ = numCores;
		numNodes_ = numNodes;
		// The zero queue is for system calls. Then there is one queue
		// per local thread. Each off-node gets another queue.
		// Note the more complex 'group' orgn for
		// eventual highly multithreaded architectures, discussed in
		// NOTES 10 Dec 2009.
		// Qinfo::setNumQs( numCores_ + numNodes_, 1024 );
		//
	} else {
		numCores_ = 1;
		numNodes_ = 1;
		// Qinfo::setNumQs( 1, 1024 );
	}
}

/**
 * Regular shell function that requires that the information about the
 * hardware have been loaded in. For now the function just assigns SimGroups
 */
void Shell::loadBalance()
{
	// Need more info here on how to set up groups distributed over
	// nodes. In fact this will have to be computed _after_ the
	// simulation is loaded. Will also need quite a bit of juggling between
	// nodes when things get really scaled up.
	//
	// Note that the messages have to be rebuilt after this call.
	for ( unsigned int i = 0; i < numNodes_; ++i )
		Qinfo::addSimGroup( numCores_ ); // These are the worker threads.
}

unsigned int Shell::numCores()
{
	return numCores_;
}

////////////////////////////////////////////////////////////////////////
// Functions for setting off clocked processes.

void Shell::start( double runtime )
{
	Id clockId( 1, 0 );
	Element* clocke = clockId();
	vector< ThreadInfo > ti( numCores_ );
	pthread_mutex_t sortMutex;
	pthread_mutex_init( &sortMutex, NULL );
	pthread_mutex_t timeMutex;
	pthread_mutex_init( &timeMutex, NULL );

	Qinfo q;
	for ( unsigned int i = 0; i < numCores_; ++i ) {
		ti[i].clocke = clocke;
		ti[i].qinfo = &q;
		ti[i].runtime = runtime;
		ti[i].threadId = i;
		ti[i].sortMutex = &sortMutex;
		ti[i].timeMutex = &timeMutex;
	}
	if ( isSingleThreaded_ ) {
		Clock::threadStartFunc( &ti[0] );
	} else {
		pthread_t* threads = new pthread_t[ numCores_ ];
		pthread_attr_t attr;

		pthread_attr_init( &attr );
		pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
		pthread_barrier_t barrier;
		if ( pthread_barrier_init( &barrier, NULL, numCores_ ) ) {
			cout << "Error: Shell::start: Unable to init barrier\n";
			exit( -1 );
		}
		Clock* clock = reinterpret_cast< Clock* >( clocke->data( 0 ) );
		clock->setBarrier( &barrier );
		clock->setNumPendingThreads( 0 ); // Used for clock scheduling
		clock->setNumThreads( numCores_ ); // Used for clock scheduling
		// pthread_t threads[ numCores_ ];
		for ( unsigned int i = 0; i < numCores_; ++i ) {
			int ret = pthread_create( 
				&threads[i], NULL, Clock::threadStartFunc, 
				reinterpret_cast< void* >( &ti[i] )
			);
			if ( ret ) {
				cout << "Error: Shell::start: Unable to create threads\n";
				exit( -1 );
			}
		}

		// Clean up.
		for ( unsigned int i = 0; i < numCores_; ++i ) {
			void* status;
			int ret = pthread_join( threads[ i ], &status );
			if ( ret ) {
				cout << "Error: Shell::start: Unable to join threads\n";
				exit( -1 );
			}
		}
		// cout << "Shell::start: Threads joined successfully\n";
		// cout << "Completed time " << runtime << " on " << numCores_ << " threads\n";

		delete[] threads;
		pthread_attr_destroy( &attr );
		pthread_barrier_destroy( &barrier );
	}
	pthread_mutex_destroy( &sortMutex );
	pthread_mutex_destroy( &timeMutex );
}

void Shell::setclock( unsigned int tickNum, double dt, unsigned int stage )
{
	Eref ce = Id( 1, 0 ).eref();
	/*
	Element* clocke = Id( 1, 0 )();
	Element* ticke = Id( 2, 0 )();
	Eref ce( clocke, 0 );
	*/
	SetGet3< unsigned int, double, unsigned int >::set( ce, "setupTick",
		tickNum, dt, stage );
}

////////////////////////////////////////////////////////////////////////

bool set( Eref& dest, const string& destField, const string& val )
{
	static Id shellid;
	static ConnId setCid = 0;
	static unsigned int setFuncIndex = 0;
	Element* shell = shellid();
	SrcFinfo1< string > sf( "set", "dummy", 0 );

	FuncId fid = dest.element()->cinfo()->getOpFuncId( destField );
	const OpFunc* func = dest.element()->cinfo()->getOpFunc( fid );
	if ( func ) {
		if ( func->checkFinfo( &sf ) ) {
			// Conn &c = shell->conn( setCid );
			shell->clearConn( setCid );
			Eref shelle = shellid.eref();
			// c.setMsgDest( shelle, dest );
			Msg* m = new SingleMsg( shelle, dest );
			shell->addMsgToConn( m->mid(), setCid );
			shell->addTargetFunc( fid, setFuncIndex );
			sf.send( shelle, Shell::procInfo(), val );
			// c.clearConn();
			return 1;
		} else {
			cout << "set::Type mismatch" << dest << "." << destField << endl;
		}
	} else {
		cout << "set::Failed to find " << dest << "." << destField << endl;
	}
	return 0;
}

bool get( const Eref& dest, const string& destField )
{
	static Id shellid;
	static ConnId getCid = 0;
	static unsigned int getFuncIndex = 0;

	static const Finfo* reqFinfo = shellCinfo->findFinfo( "requestGet" );
	static const SrcFinfo1< FuncId >* rf = 
		dynamic_cast< const SrcFinfo1< FuncId >* >( reqFinfo );
	static FuncId retFunc = shellCinfo->getOpFuncId( "handleGet" );
	static SrcFinfo1< string > sf( "get", "dummy", 0 );

	static Element* shell = shellid();
	static Eref shelle( shell, 0 );

	FuncId fid = dest.element()->cinfo()->getOpFuncId( destField );
	const OpFunc* func = dest.element()->cinfo()->getOpFunc( fid );

	assert( rf != 0 );

	if ( func ) {
		if ( func->checkFinfo( &sf ) ) {
			shell->clearConn( getCid );
			Msg* m = new SingleMsg( shelle, dest );
			shell->addMsgToConn( m->mid(), getCid );

			shell->addTargetFunc( fid, getFuncIndex );
			rf->send( shelle, Shell::procInfo(), retFunc );
			// Now, dest has to clearQ, do its stuff, then src has to clearQ
			return 1;
		} else {
			cout << "set::Type mismatch" << dest << "." << destField << endl;
		}
	} else {
		cout << "set::Failed to find " << dest << "." << destField << endl;
	}
	return 0;
}

