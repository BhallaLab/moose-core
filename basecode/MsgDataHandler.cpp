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

MsgDataHandler::MsgDataHandler( const DinfoBase* dinfo, bool isGlobal )
		: DataHandler( dinfo, isGlobal ), 
			data_( 0 )
{;}

MsgDataHandler::MsgDataHandler( const MsgDataHandler* other )
		: DataHandler( other->dinfo(), other->isGlobal() ),
			data_( 0 )
{;}

MsgDataHandler::~MsgDataHandler()
{;} // No data to destroy.

/////////////////////////////////////////////////////////////////////////
// Information functions
/////////////////////////////////////////////////////////////////////////

/**
 * This is the key magic operation done by the MsgDataHandler. It looks
 * up the regular msg, using the DataId.data() as the MsgId.
 */
char* MsgDataHandler::data( DataId index ) const
{
	return reinterpret_cast< char* >( Msg::safeGetMsg( index.value() ) );
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

/////////////////////////////////////////////////////////////////////////
// Load balancing functions
/////////////////////////////////////////////////////////////////////////

bool MsgDataHandler::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

/////////////////////////////////////////////////////////////////////////
// Process and foreach functions
/////////////////////////////////////////////////////////////////////////

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
	;
}

/// Another unlikely function for MsgDataHandler
void MsgDataHandler::foreach( const OpFunc* f, Element* e, const Qinfo* q,
	const double* arg, unsigned int argIncrement ) const
{
	;
}
/////////////////////////////////////////////////////////////////////////
// Data Reallocation functions
/////////////////////////////////////////////////////////////////////////
void MsgDataHandler::globalize( const char* data, unsigned int size )
{
	; // MsgDataHandlers are always global.
}


void MsgDataHandler::unGlobalize()
{
	;
}

DataHandler* MsgDataHandler::copy( bool toGlobal, unsigned int n ) const
{
	return ( new MsgDataHandler( this ) );
}

/// copyUsingNewDinfo is illegal for MsgDataHandlers.
DataHandler* MsgDataHandler::copyUsingNewDinfo( const DinfoBase* d) const
{
	assert( 0 ); 
	return 0;
}

// Simply not permitted.
bool MsgDataHandler::resize( unsigned int dimension, unsigned int size )
{
	return 0;
}

void assign( const char* orig, unsigned int numOrig )
{
	;
}
