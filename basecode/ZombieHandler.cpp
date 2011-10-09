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
	: DataHandler( parentHandler->dinfo(), parentHandler->isGlobal() ),
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
 * Returns the number of data entries.
 */
unsigned int ZombieHandler::totalEntries() const {
	return end_ - start_;
}

/**
 * Returns the number of data entries on local node.
 */
unsigned int ZombieHandler::localEntries() const {
	return end_ - start_;
}

/**
 * Returns the number of dimensions of the data.
 */
unsigned int ZombieHandler::numDimensions() const {
	return parent_->numDimensions();
}

vector< unsigned int > ZombieHandler::dims() const {
	return parent_->dims();
}

unsigned int ZombieHandler::sizeOfDim( unsigned int dim ) const {
	return parent_->sizeOfDim( dim );
}

bool ZombieHandler::isDataHere( DataId index ) const {
	return ( start_ <= index.value() && end_ > index.value() );
}

bool ZombieHandler::isAllocated() const {
	return parent_->isAllocated();
}

bool ZombieHandler::isGlobal() const
{
	return parent_->isGlobal();
}

////////////////////////////////////////////////////////////////////
// Load balancing
////////////////////////////////////////////////////////////////////

bool ZombieHandler::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

////////////////////////////////////////////////////////////////////
// Process and foreach
////////////////////////////////////////////////////////////////////

void ZombieHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{;} // Solver parent does process, this does not.

void ZombieHandler::foreach( const OpFunc* f, Element* e, const Qinfo* q,
 	const double* arg, unsigned int argSize, unsigned int numArgs ) const
{ // This should be relevant. But I still have to define threadStart_
/*
	assert( q->threadNum() < threadStart_.size() );
	unsigned int end = threadStart_[ q->threadNum() + 1 ];
	if ( numArgs <= 1 ) {
		for( unsigned int i = threadStart_[ q->threadNum() ]; i != end; ++i)
			f->op( Eref( e, i ), q, arg );
	} else {
		unsigned int argOffset = 0;
		unsigned int maxOffset = argSize * numArgs;
		for( unsigned int i = threadStart_[ q->threadNum() ];
			i != end; ++i) {
			f->op( Eref( e, i ), q, arg + argOffset );
			argOffset += argSize;
			if ( argOffset >= maxOffset )
				argOffset = 0;
		}
	}
	*/
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

DataHandler* ZombieHandler::copy( bool toGlobal, unsigned int n ) const
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
