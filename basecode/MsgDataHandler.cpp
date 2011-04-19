/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MsgDataHandler.h"

MsgDataHandler::MsgDataHandler( const DinfoBase* dinfo )
		: DataHandler( dinfo ), 
			data_( 0 )
{;}

MsgDataHandler::MsgDataHandler( const MsgDataHandler* other )
		: DataHandler( other->dinfo() ),
			data_( 0 )
{;}

MsgDataHandler::~MsgDataHandler()
{;} // No data to destroy.

DataHandler* MsgDataHandler::globalize() const
{
	return copy();
}


DataHandler* MsgDataHandler::unGlobalize() const
{
	return copy();
}

DataHandler* MsgDataHandler::copy() const
{
	return ( new MsgDataHandler( this ) );
}

/// copyUsingNewDinfo is illegal for MsgDataHandlers.
DataHandler* MsgDataHandler::copyUsingNewDinfo( const DinfoBase* d) const
{
	assert( 0 ); 
	return 0;
}

DataHandler* MsgDataHandler::copyExpand( unsigned int copySize ) const
{
	return 0; // Illegal.
}

/**
 * Expand it into a 2-dimensional version of AnyDimGlobalHandler.
 */
DataHandler* MsgDataHandler::copyToNewDim( unsigned int newDimSize ) 
	const
{
	return 0;
}


/**
 * This is the key magic operation done by the MsgDataHandler. It looks
 * up the regular msg, using the DataId.data() as the MsgId.
 */
char* MsgDataHandler::data( DataId index ) const
{
	return reinterpret_cast< char* >( Msg::safeGetMsg( index.data() ) );
}

/**
 * Handles both the thread and node decomposition
 * Here there is no node decomposition: all Msgs are present
 * on all nodes. But they might do different things as they could have
 * node-specific content. Only rather specialized Msgs would do update
 * operations on each clock tick. Other issue is that we don't want
 * messages rebuilding during execution time, so any process operations
 * will have to put stuff into the structuralQ for serial execution.
 * Yet another issue is that this has to be done specifically for the
 * msgs of this one type, for which the implementation is pending.
 * So for now leave in the hooks.
 */
void MsgDataHandler::process( const ProcInfo* p, Element* e, 
	FuncId fid ) const
{
	/*
	char* temp = data_ + p->threadIndexInGroup * dinfo()->size();
	unsigned int stride = dinfo()->size() * p->numThreadsInGroup;

	const OpFunc* f = e->cinfo()->getOpFunc( fid );
	const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
	assert( pf );

	for ( unsigned int i = p->threadIndexInGroup; i < size_; 
		i+= p->numThreadsInGroup )
	{
		// reinterpret_cast< Data* >( temp )->process( p, Eref( e, i ) );
		pf->proc( temp, Eref( e, i ), p );
		temp += stride;
	}
	*/
}

/// Later do something intelligent based on # of Msgs of current type.
unsigned int MsgDataHandler::totalEntries() const {
	return 1;
}

/// Later do something intelligent based on # of Msgs of current type.
unsigned int MsgDataHandler::localEntries() const {
	return 1;
}

unsigned int MsgDataHandler::numDimensions() const {
	return 1;
}


/// Later do something intelligent based on # of Msgs of current type.
unsigned int MsgDataHandler::sizeOfDim( unsigned int dim ) const
{
	if ( dim == 0 )
		return 1;
	return 0;
}

bool MsgDataHandler::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

// Simply not permitted.
bool MsgDataHandler::resize( vector< unsigned int > dims )
{
	return 0;
}

vector< unsigned int > MsgDataHandler::dims() const
{
	vector< unsigned int > ret( 1, Msg::numMsgs() );
	return ret;
}

bool MsgDataHandler::isDataHere( DataId index ) const {
	return 1;
}

bool MsgDataHandler::isAllocated() const {
	return 1; // Msgs are always allocated
}

bool MsgDataHandler::isGlobal() const {
	return 1;
}


DataHandler::iterator MsgDataHandler::begin() const {
	return iterator( this, 0, 0 );
}

DataHandler::iterator MsgDataHandler::end() const {
	return iterator( this, 1, 1 );
}

// Illegal
bool MsgDataHandler::setDataBlock( 
	const char* data, unsigned int numData,
	const vector< unsigned int >& startIndex ) const
{
	return 0;
}

bool MsgDataHandler::setDataBlock( 
	const char* data, unsigned int numData,
	DataId startIndex ) const
{
	return 0;
}

void MsgDataHandler::nextIndex( DataId& index,
	unsigned int& linearIndex ) const
{
	index.incrementDataIndex();
	++linearIndex;
}
