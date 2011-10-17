/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

ZombieHandler::ZombieHandler( const DataHandler* parentHandler,
	unsigned int start, unsigned int end )
	: DataHandler( 
			parentHandler->dinfo(), parentHandler->dims(),
			parentHandler->pathDepth(), parentHandler->isGlobal()
		),
		parent_( parentHandler ),
		start_( start ), end_( end )
{;}

ZombieHandler::~ZombieHandler()
{;} // This deletes itself but does not touch parent

////////////////////////////////////////////////////////////////////
// Information functions
////////////////////////////////////////////////////////////////////

char* ZombieHandler::data( DataId index ) const
{
	return parent_->data( 0 );
}

/**
 * Returns the number of data entries on local node.
 */
unsigned int ZombieHandler::localEntries() const {
	return end_ - start_;
}

bool ZombieHandler::isDataHere( DataId index ) const {
	return ( start_ <= index.value() && end_ > index.value() );
}

bool ZombieHandler::isAllocated() const {
	return parent_->isAllocated();
}

unsigned int ZombieHandler::linearIndex( DataId di ) const
{
	// Hack. need somehow to keep track of how the original Handler
	// dealt with this.
	return di.value(); 

	// return parent_->linearIndex( di );
}
////////////////////////////////////////////////////////////////////
// Load balancing
////////////////////////////////////////////////////////////////////

bool ZombieHandler::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

bool ZombieHandler::execThread( ThreadId thread, DataId di ) const
{
	return parent_->execThread( thread, di );
}

////////////////////////////////////////////////////////////////////
// Process and foreach
////////////////////////////////////////////////////////////////////

void ZombieHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{;} // Solver parent does process, this does not.

void ZombieHandler::foreach( const OpFunc* f, Element* e, const Qinfo* q,
 	const double* arg, unsigned int argSize, unsigned int numArgs ) const
{ // This should be relevant. But I still have to define threadStart_
	// Find which thread parent is willing to handle
	// I still don't know how to get at the parent DataId. I need it.
	if ( parent_->execThread( q->threadNum(), 0 ) ) {
		// On this thread, figure out which index range we should use.
		// We have start and end, so that is fine.
		// Iterate through the opfunc for all of these.
		if ( numArgs <= 1 ) {
			for( unsigned int i = start_; i != end_; ++i ) {
				f->op( Eref( e, i ), q, arg );
			}
		} else {
			unsigned int argOffset = argSize * start_;
			unsigned int maxOffset = argSize * numArgs;
			for( unsigned int i = start_; i != end_; ++i ) {
				f->op( Eref( e, i ), q, arg + argOffset );
				argOffset += argSize;
				if ( argOffset >= maxOffset )
					argOffset = 0;
			}
		}
	}
}

unsigned int ZombieHandler::getAllData( vector< char* >& dataVec ) const
{
	dataVec.resize( 0 );
	return 0;
}

////////////////////////////////////////////////////////////////////
// Data reallocation functions
////////////////////////////////////////////////////////////////////

void ZombieHandler::globalize( const char* data, unsigned int size )
{
	; // Don't know how to do this.
}

void ZombieHandler::unGlobalize()
{
	;
}

DataHandler* ZombieHandler::copy( 
	unsigned short pathDepth, bool toGlobal, unsigned int n ) const
{
	return ( new ZombieHandler( parent_, start_ * n, end_ * n ) );
}

DataHandler* ZombieHandler::copyUsingNewDinfo(
	const DinfoBase* dinfo ) const
{
	return 0;
}

// Unsure what to do here.
bool ZombieHandler::resize( unsigned int dimension, unsigned int size )
{
	return 1;
}

// Can't really do this.
void ZombieHandler::assign( const char* orig, unsigned int numOrig )
{
	;
}
