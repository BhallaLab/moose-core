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
	: DataHandler( parentHandler->dinfo() ), parent_( parentHandler ),
		start_( start ), end_( end )
{;}

ZombieHandler::~ZombieHandler()
{;} // This deletes itself but does not touch parent


DataHandler* ZombieHandler::globalize() const
{
	return 0; // Don't know how to do this.
}

DataHandler* ZombieHandler::unGlobalize() const
{
	return 0;
}

bool ZombieHandler::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

DataHandler* ZombieHandler::copy( bool toGlobal ) const
{
	return ( new ZombieHandler( parent_, start_, end_ ) );
}

DataHandler* ZombieHandler::copyUsingNewDinfo(
	const DinfoBase* dinfo ) const
{
	return 0;
}

DataHandler* ZombieHandler::copyExpand( 
	unsigned int copySize, bool toGlobal ) const
{
	// Can't really do this till we have the policy for copying to and
	// from globals sorted out.
	ZombieHandler* ret = new ZombieHandler( parent_ );
	return ret;
}

DataHandler* ZombieHandler::copyToNewDim( 
	unsigned int newDimSize, bool toGlobal ) const
{
	return 0;
}

char* ZombieHandler::data( DataId index ) const
{
	return parent_->data( 0 );
}

void ZombieHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{;} // Solver parent does process, this does not.

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

unsigned int ZombieHandler::sizeOfDim( unsigned int dim ) const {
	return parent_->sizeOfDim( dim );
}

// For now just change start_ and end_, later figure out how to put in
// node policy.
bool ZombieHandler::resize( vector< unsigned int > dims ) {
	assert( dims.size() > 0 );
	start_ = 0;
	end_ = dims[0];
	return 1;
}

vector< unsigned int > ZombieHandler::dims() const {
	return parent_->dims();
}

bool ZombieHandler::isDataHere( DataId index ) const {
	return ( start_ <= index.data() && end_ > index.data() );
}

bool ZombieHandler::isAllocated() const {
	return parent_->isAllocated();
}

bool ZombieHandler::isGlobal() const
{
	return parent_->isGlobal();
}

DataHandler::iterator ZombieHandler::begin() const
{
	return iterator( this, start_, start_ );
}

DataHandler::iterator ZombieHandler::end() const
{
	return iterator( this, end_, end_ );
}

bool ZombieHandler::setDataBlock( const char* data, 
	unsigned int numEntries, DataId startIndex ) const
{
	return 0;
}

bool ZombieHandler::setDataBlock( 
	const char* data, unsigned int numEntries, 
	const vector< unsigned int >& startIndex ) const
{
	return 0;
}

void ZombieHandler::nextIndex( DataId& index, 
	unsigned int& linearIndex ) const
{
	index.incrementDataIndex();
	++linearIndex;
}
