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
	const DinfoBase* dinfo, const DataHandler* parentDataHandler )
			: DataHandler( dinfo ),
				parentDataHandler_( parentDataHandler ),
				fieldDimension_( 1 )
{;}

FieldDataHandlerBase::~FieldDataHandlerBase()
{;} // Don't delete data because the parent Element should do so.

DataHandler* FieldDataHandlerBase::globalize() const
{
	return 0;
}

DataHandler* FieldDataHandlerBase::unGlobalize() const
{
	return 0;
}

bool FieldDataHandlerBase::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

DataHandler* FieldDataHandlerBase::copyExpand( 
	unsigned int copySize, bool toGlobal ) const
{
	return 0;
}

/*
DataHandler* FieldDataHandlerBase::copyToNewDim( unsigned int newDimSize )
	const
{
	return 0;
}
*/

void FieldDataHandlerBase::process( const ProcInfo* p, Element* e, FuncId fid ) const 
{
	/**
	 * This is the variant with threads in a block.
	 */
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

	const OpFunc* f = e->cinfo()->getOpFunc( fid );
	const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
	assert( pf );
	for ( unsigned int i = startIndex; i != endIndex; ++i ) {
		DataId me( 0, i );
		char* temp = data( me );
		pf->proc( temp, Eref( e, me ), p );
	}
}

/**
 * Returns the data on the specified index.
 */
char* FieldDataHandlerBase::data( DataId index ) const
{
	return lookupField( parentDataHandler_->data( index ), index.field() );
}

/**
 * Returns the number of field entries.
 * If parent is global the return value is also global.
 * If parent is local then it returns # on current node.
 */
unsigned int FieldDataHandlerBase::totalEntries() const
{
	return parentDataHandler_->totalEntries() * fieldDimension_;
}

/**
 * Returns the number of field entries.
 * If parent is global the return value is also global.
 * If parent is local then it returns # on current node.
 */
unsigned int FieldDataHandlerBase::localEntries() const
{
	unsigned int ret = 0;
	for ( DataHandler::iterator i = parentDataHandler_->begin();
		i != parentDataHandler_->end(); ++i ) {
		ret += getNumField( *i );
	}
	return ret;
}

/**
 * Returns a single number corresponding to the DataId.
 * Note that this does NOT compact the number in the case of
 * ragged arrays. It instead treats the indexing as if on a
 * square matrix.
 */
unsigned int FieldDataHandlerBase::linearIndex( const DataId& d ) const
{
	return d.data() * fieldDimension_ + d.field();
}

/**
 * Returns the DataId corresponding to a single index.
 */
DataId FieldDataHandlerBase::dataId( unsigned int linearIndex) const
{
	return DataId( linearIndex / fieldDimension_, 
		linearIndex % fieldDimension_ );
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
	return fieldDimension_;
}


/**
 * Assigns size for first (data) dimension. This usually will not
 * be called here, but by the parent data Element.
 */
bool FieldDataHandlerBase::resize( vector< unsigned int > dims )
{
	cout << Shell::myNode() << ": FieldDataHandler::setNumData1: Error: Cannot resize from Field\n";
	return 0;
}

/**
 * Returns the dimensions of this. The Field dimension is on 
 * index 0.
 */
vector< unsigned int > FieldDataHandlerBase::dims() const
{
	vector< unsigned int > ret( parentDataHandler_->dims() );
	ret.insert( ret.begin(), fieldDimension_ );
	return ret;
}

/**
 * Assigns the size of the field array on the specified object.
 */
void FieldDataHandlerBase::setFieldArraySize( 
	unsigned int objectIndex, unsigned int size )
{
	assert( objectIndex < parentDataHandler_->totalEntries() );

	if ( parentDataHandler_->isDataHere( objectIndex ) ) {
		char* pa = parentDataHandler_->data( objectIndex );
		setNumField( pa, size );
		if ( size > fieldDimension_ )
			fieldDimension_ = size;
	}
}

/**
 * Looks up the size of the local field array on the specified object
 */
unsigned int FieldDataHandlerBase::getFieldArraySize( unsigned int objectIndex ) const
{
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
	for ( iterator i = parentDataHandler_->begin(); i !=
		parentDataHandler_->end(); ++i )
	{
		char* pa = *i;
		assert( pa );
		unsigned int numHere = getNumField( pa );
		if ( numHere > ret )
			ret = numHere;
	}

	// Here it would be nice to get FieldArraySize from all nodes. 
	// But we can't do this here as we don't know for sure that the
	// current function will be called on all nodes.
	// ret = Shell::reduceInt( ret ); 
	return ret;
}

/**
 * This func gets the FieldArraySize from all nodes and updates
 * fieldDimension to the largest.
 * MUST be called on all nodes in sync.
 * Deprecated
unsigned int FieldDataHandlerBase::syncFieldArraySize()
{
	unsigned int ret = biggestFieldArraySize();
	// ret = Shell::reduceInt( ret ); 
	if ( fieldDimension_ < ret )
		fieldDimension_ = ret;
	return ret;
}
 */

/**
 * Assigns the fieldDimension. Checks that it is bigger than the
 * biggest size on this node.
 */
void FieldDataHandlerBase::setFieldDimension( unsigned int size )
{
	unsigned int i = biggestFieldArraySize();
	assert( i <= size );
	fieldDimension_ = size;
	//cout << Shell::myNode() << ": SetFieldDimension to " << size << endl;
}

/**
 * Returns fieldDimension
 */
unsigned int FieldDataHandlerBase::getFieldDimension( ) const
{
	return fieldDimension_;
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

bool FieldDataHandlerBase::isGlobal() const
{
	return parentDataHandler_->isGlobal();
}

/////////////////////////////////////////////////////////////////
// Iterators
/////////////////////////////////////////////////////////////////

DataHandler::iterator FieldDataHandlerBase::begin() const {
	for ( iterator i = parentDataHandler_->begin(); i !=
		parentDataHandler_->end(); ++i ) {
		char* pa = *i;
		assert( pa );
		unsigned int numHere = getNumField( pa );
		if ( numHere != 0 )
			return iterator( this, i.index(), 
				i.index().data() * fieldDimension_ );
	}
	// Failed to find any valid index
	return end();
}

/**
 * This is 1+(last valid field entry) on the last valid data entry
 * on the parent data handler, expressed as a single int.
 */
DataHandler::iterator FieldDataHandlerBase::end() const {
	return iterator( this, parentDataHandler_->end().index(), 
		parentDataHandler_->end().index().data() * fieldDimension_);
}

void FieldDataHandlerBase::nextIndex( DataId& index, unsigned int& linearIndex ) const {
	char* pa = parentDataHandler_->data( index );
	assert( pa );
	unsigned int numHere = getNumField( pa );
	if ( index.field() + 1 < numHere ) {
		index.incrementFieldIndex();
		++linearIndex;
		return;
	}

	index.rolloverFieldIndex();
	unsigned int j = index.data();
	for ( iterator i( parentDataHandler_, j, j * fieldDimension_ ); 
		i != parentDataHandler_->end(); ++i ) {
		index = i.index();
		char* pa = *i;
		assert( pa );
		numHere = getNumField( pa );
		if ( numHere > 0 ) {
			linearIndex = index.data() * fieldDimension_;
			return;
		}
	}
	// If we fall out of this loop we must be at the end.
	index = parentDataHandler_->end().index();
}

/////////////////////////////////////////////////////////////////

const DataHandler* FieldDataHandlerBase::parentDataHandler() const
{
	return parentDataHandler_;
}

void FieldDataHandlerBase::assignParentDataHandler( 
	const DataHandler* parent )
{
	parentDataHandler_ = parent;
}

/////////////////////////////////////////////////////////////////
// setDataBlock stuff. Defer implementation for now.
/////////////////////////////////////////////////////////////////

bool FieldDataHandlerBase::setDataBlock( const char* data, unsigned int numData,
	DataId startIndex ) const 
{
	/*
	if ( parentDataHandler_->isDataHere( startIndex.data() ) ) {
		char* temp = parentDataHandler_->data( startIndex.data() );
		assert( temp );
		Parent* pa = reinterpret_cast< Parent* >( temp );

		unsigned int numField = ( pa->*getNumField_ )();
		unsigned int max = numData;
		if ( did.field() + numData > numField  )
			max = numField - did.field();
		for ( unsigned int i = 0; i < max; ++i ) {
			Field* f = ( pa->*lookupField_ )( did.field() + i );
			*f = *reinterpret_cast< const Field* >( 
				data + i * dinfo()->size() );
		}
	}
	return 1;
	*/
	return 0;
}

/**
 * Assigns a block of data at the specified location.
 * Returns true if all OK. No allocation.
 */
bool FieldDataHandlerBase::setDataBlock( const char* data, unsigned int numData,
	const vector< unsigned int >& startIndex ) const
{
/*
	if ( startIndex.size() == 0 )
		return setDataBlock( data, numData, 0 );
	unsigned int fieldIndex = startIndex[0];
	if ( startIndex.size() == 1 )
		return setDataBlock( data, numData, DataId( 0, fieldIndex ) );
	vector< unsigned int > temp = startIndex;
	temp.pop_back(); // Get rid of fieldIndex.
	DataDimensions dd( parentDataHandler_->dims() );
	unsigned int paIndex = dd.linearIndex( temp );
	return setDataBlock( data, numData, DataId( paIndex, fieldIndex ) );
	*/
	return 0;
}
