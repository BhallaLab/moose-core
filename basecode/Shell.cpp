/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Shell.h"
#include "Dinfo.h"

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
		new DestFinfo( "handleGet", 
			"Function to handle returning values for 'get' calls.",
			new RetFunc< Shell >( &Shell::handleGet ) ),
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

// Static function to assign hardware availability
void Shell::setHardware( 
	bool isSingleThreaded, unsigned int numCores, unsigned int numNodes )
{
	isSingleThreaded_ = isSingleThreaded;
	if ( !isSingleThreaded ) {
		// Create the parser and the gui threads.
		numCores_ = numCores;
		numNodes_ = numNodes;
		// The zero queue is for system calls. Then there is one queue
		// per local thread. Each off-node gets another queue.
		// Note the more complex 'group' orgn for
		// eventual highly multithreaded architectures, discussed in
		// NOTES 10 Dec 2009.
		Qinfo::setNumQs( numCores_ + numNodes_, 1024 );
	} else {
		numCores_ = 1;
		numNodes_ = 1;
		Qinfo::setNumQs( 1, 1024 );
	}
}

unsigned int Shell::numCores()
{
	return numCores_;
}

////////////////////////////////////////////////////////////////////////
// Functions for setting off clocked processes.

/*
void Shell::startClock()
{
	if ( isSingleThreaded ) {
		
	} else {
	vector< ThreadInfo > ti( numCores_ );
	for ( i = 0; i < numCores_; ++i ) {
		ti[i].clock = clocke;
		ti[i].qinfo = parseQinfo;
		ti[i].runtime = runtime;
		ti[i].threadId = i;
	}
	// May want to use pthread_create even if single core, when on a 
	// normal system. The strict single-thread mode will be rare.
	if ( numCores_ == 1 ) {
		
	} else {
	}
}
*/

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

