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

MsgDataHandler::MsgDataHandler( const DinfoBase* dinfo, 
		const vector< DimInfo >& dims, unsigned short pathDepth, 
		bool isGlobal )
		: DataHandler( dinfo, dims, pathDepth, isGlobal ), 
			data_( 0 )
{;}

MsgDataHandler::MsgDataHandler( const MsgDataHandler* other )
		: DataHandler( 
				other->dinfo(), other->dims(), 
				other->pathDepth(), other->isGlobal()
			),
			data_( 0 )
{;}

MsgDataHandler::~MsgDataHandler()
{;} // No data to destroy.

/////////////////////////////////////////////////////////////////////////
// Class specific functions
/////////////////////////////////////////////////////////////////////////
void MsgDataHandler::addMid( MsgId mid )
{
	mids_.push_back( mid );
}

/// Returns true if the mid was indeed removed. False if not found.
bool MsgDataHandler::dropMid( MsgId mid )
{
	// Oh, no, not the STL monstrosity again
	vector< MsgId >::iterator end = 
		remove( mids_.begin(), mids_.end(), mid );
	bool ret = ( end == mids_.end() );
	mids_.erase( end, mids_.end() );
	return !ret;
}


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
// unsigned int MsgDataHandler::totalEntries() const

/// How many mids have been declared?
unsigned int MsgDataHandler::localEntries() const {
	return mids_.size();
}

/// Converts index into a mid, returns true if it is present here.
bool MsgDataHandler::isDataHere( DataId index ) const {
	MsgId mid = index.value();
	vector< MsgId >::const_iterator i = find( mids_.begin(), mids_.end(), mid );
	return ( i != mids_.end() );
}

bool MsgDataHandler::isAllocated() const {
	return 1; // Msgs are always allocated
}

unsigned int MsgDataHandler::linearIndex( DataId di ) const {
	return di.value();
}

vector< vector< unsigned int > > 
	MsgDataHandler::pathIndices( DataId di ) const
{
	vector< vector< unsigned int > > ret( pathDepth_ + 1 );
	assert( pathDepth_ >= 2 );
	vector < unsigned int > temp( 1, di.value() );
	ret[2] = temp;
	return ret;
}

/// Dummy for now.
DataId MsgDataHandler::pathDataId( 
	const vector< vector< unsigned int > >& indices) const
{
	if ( indices.size() != static_cast< unsigned int >( pathDepth_ ) + 1 )
	  return DataId::bad();
	return DataId( 0 );
}
/////////////////////////////////////////////////////////////////////////
// Load balancing functions
/////////////////////////////////////////////////////////////////////////

bool MsgDataHandler::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

/// Later do something more sensible.
bool MsgDataHandler::execThread( ThreadId thread, DataId di ) const
{
	return (thread <= 1);
}

/////////////////////////////////////////////////////////////////////////
// Process and forall functions
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

/**
 * forall: Implementation pending.
 */
void MsgDataHandler::forall( const OpFunc* f, Element* e, const Qinfo* q,
	const double* arg, unsigned int argSize, unsigned int numArgs ) const
{
	;
}

/// Perhaps can return the mids_.
unsigned int MsgDataHandler::getAllData( vector< char* >& dataVec ) const
{
	dataVec.resize( 0 );
	return 0;
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

// Really a dummy, since we don't expect to have this call.
DataHandler* MsgDataHandler::copy( 
	unsigned short newParentDepth, 
	unsigned short copyRootDepth, 
	bool toGlobal, unsigned int n ) const
{
	cout << "Error: MsgDataHandler::copy: should never call me\n";
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

void MsgDataHandler::assign( const char* orig, unsigned int numOrig )
{
	;
}
