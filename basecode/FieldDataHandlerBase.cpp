/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

#include "../shell/Shell.h"

FieldDataHandlerBase::FieldDataHandlerBase( 
	const DinfoBase* dinfo,
	const DataHandler* parentDataHandler )
			: DataHandler( dinfo, parentDataHandler->isGlobal() ),
				maxFieldEntries_( 1 ),
				parentDataHandler_( parentDataHandler )
{;}

FieldDataHandlerBase::~FieldDataHandlerBase()
{;} // Don't delete data because the parent Element should do so.

/////////////////////////////////////////////////////////////////////////
// Information functions
/////////////////////////////////////////////////////////////////////////

/**
* Returns the data on the specified index.
*/
char* FieldDataHandlerBase::data( DataId di ) const
{
	return lookupField( parentDataHandler_->data( di.parentIndex( numFieldBits_ ) ), di.myIndex( mask_ ) );
}


/**
 * Returns the number of field entries.
 * If parent is global the return value is also global.
 * If parent is local then it returns # on current node.
 */
unsigned int FieldDataHandlerBase::totalEntries() const
{
	return parentDataHandler_->totalEntries() * maxFieldEntries_;
}

/**
 * Returns the number of field entries.
 * If parent is global the return value is also global.
 * If parent is local then it returns # on current node.
 */
unsigned int FieldDataHandlerBase::localEntries() const
{
	unsigned int ret = 0;
	/*
	for ( DataHandler::iterator i = parentDataHandler_->begin( 0 );
		i != parentDataHandler_->end( Shell::numProcessThreads() ); ++i ) {
		ret += getNumField( i.data() );
	}
	*/
	return ret;
}

/**
 * Returns the number of dimensions of the data.
 */
unsigned int FieldDataHandlerBase::numDimensions() const {
	// Should refine to include local dimensions.
	// For now assume 1 dim.
	return parentDataHandler_->numDimensions() + 1;
}

/**
 * Returns the indexing range of the data at the specified dimension.
 */
unsigned int FieldDataHandlerBase::sizeOfDim( unsigned int dim ) const
{
	if ( dim > 0 )
		return parentDataHandler_->sizeOfDim( dim - 1 );
	return maxFieldEntries_;
}

/**
 * Returns the dimensions of this. The Field dimension is on 
 * index 0.
 */
vector< unsigned int > FieldDataHandlerBase::dims() const
{
	vector< unsigned int > ret( parentDataHandler_->dims() );
	ret.insert( ret.begin(), maxFieldEntries_ );
	return ret;
}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool FieldDataHandlerBase::isDataHere( DataId index ) const {
	return parentDataHandler_->isDataHere( index );
}

bool FieldDataHandlerBase::isAllocated() const {
	return parentDataHandler_->isAllocated();
}

unsigned int FieldDataHandlerBase::numFieldBits() const {
	return numFieldBits_;
}

const DataHandler* FieldDataHandlerBase::parentDataHandler() const
{
	return parentDataHandler_;
}

/////////////////////////////////////////////////////////////////////////
// Load balancing
/////////////////////////////////////////////////////////////////////////

bool FieldDataHandlerBase::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

/////////////////////////////////////////////////////////////////////////
// Process and foreach
/////////////////////////////////////////////////////////////////////////

void FieldDataHandlerBase::process( const ProcInfo* p, Element* e, FuncId fid ) const 
{
	/**
	 * This is the variant with threads in a block.
	unsigned int startIndex = 0;
	unsigned int endIndex = localEntries();
	if ( Shell::numProcessThreads() > 1 ) {
		// Note that threadIndexInGroup is indexed from 1 up.
		assert( p->threadIndexInGroup >= 1 );
		startIndex = ( localEntries() * ( p->threadIndexInGroup - 1 ) + 
			Shell::numProcessThreads() - 1 ) / Shell::numProcessThreads();

		endIndex = ( localEntries() * p ->threadIndexInGroup +
			Shell::numProcessThreads() - 1 ) / Shell::numProcessThreads();
	}
	 */

	/*
	* We set up the ProcFunc of this field class to do the local iteration
	* in a manner somewhat like below. The fid is then defined locally in
	* this Field class and the parentDataHandler's process deals with it.
	const OpFunc* f = e->cinfo()->getOpFunc( fid );
	const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
	assert( pf );
	FieldProcOpFunc fp = pf->makeFieldProcOpFunc( this );
	assert( fp );
	*/
	parentDataHandler_->process( p, e, fid );
}

/**
 * Here we delegate the operation to the parent, and wrap up the local
 * iterators for the field in the FieldOpFunc. Note that the thread
 * decomposition belongs to the parent, as it must: the operations here
 * will modify data structures on the parent.
 */
void FieldDataHandlerBase::foreach( const OpFunc* f, Element* e,
	const Qinfo* q, const double* arg, unsigned int argIncrement ) const
{
	FieldOpFunc fof( f, e );
	ObjId parent = Neutral::parent( Eref( e, 0 ) );
	parentDataHandler_->foreach( &fof, parent.element(), q, arg, argIncrement );
}

/////////////////////////////////////////////////////////////////////////
// Data reallocation.
/////////////////////////////////////////////////////////////////////////
void FieldDataHandlerBase::globalize( const char* data, unsigned int size )
{;}

void FieldDataHandlerBase::unGlobalize()
{;}

/**
 * Assigns size for first (data) dimension. This usually will not
 * be called here, but by the parent data Element.
 */
bool FieldDataHandlerBase::resize( unsigned int dimension, unsigned int size)
{
	if ( dimension == 0 ) {
		unsigned int i = biggestFieldArraySize();
		assert( i <= size );
		maxFieldEntries_ = size;
	}

	cout << Shell::myNode() << ": FieldDataHandler::resize: Error: Cannot resize from Field\n";
	return 0;
}

/**
 * Assigns the size of the field array on the specified object.
 * Here we could replace objectIndex with a DataId and 
 * trim off the field part.
 */
void FieldDataHandlerBase::setFieldArraySize( 
	DataId di, unsigned int size )
{
	unsigned int objectIndex = di.value() >> numFieldBits_;
	assert( objectIndex < parentDataHandler_->totalEntries() );

	if ( parentDataHandler_->isDataHere( objectIndex ) ) {
		char* pa = parentDataHandler_->data( objectIndex );
		setNumField( pa, size );
		if ( size > maxFieldEntries_ )
			maxFieldEntries_ = size;
	}
}

/////////////////////////////////////////////////////////////////////////

/**
 * Looks up the size of the local field array on the specified object
 */
unsigned int FieldDataHandlerBase::getFieldArraySize( DataId di ) const
{
	unsigned int objectIndex = di.value() >> numFieldBits_;
	assert( objectIndex < parentDataHandler_->totalEntries() );
	if ( parentDataHandler_->isDataHere( objectIndex ) ) {
		char* pa = parentDataHandler_->data( objectIndex );
		return getNumField( pa );
	}
	return 0;
}

/**
 * Looks up the biggest field array size on all nodes.
 */
unsigned int FieldDataHandlerBase::biggestFieldArraySize() const
{
	unsigned int ret = 0;
	/*
	for ( iterator i = parentDataHandler_->begin(0); i !=
		parentDataHandler_->end( Shell::numProcessThreads() ); ++i )
	{
		char* pa = i.data();
		assert( pa );
		unsigned int numHere = getNumField( pa );
		if ( numHere > ret )
			ret = numHere;
	}
	*/

	// Here it would be nice to get FieldArraySize from all nodes. 
	// But we can't do this here as we don't know for sure that the
	// current function will be called on all nodes.
	// ret = Shell::reduceInt( ret ); 
	return ret;
}

/////////////////////////////////////////////////////////////////

void FieldDataHandlerBase::assignParentDataHandler( 
	const DataHandler* parent )
{
	parentDataHandler_ = parent;
}
