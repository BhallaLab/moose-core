/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "FieldOpFunc.h"
#include "../shell/Shell.h"

FieldDataHandlerBase::FieldDataHandlerBase( 
	const DinfoBase* dinfo,
	const DataHandler* parentDataHandler, 
	unsigned int size )
			: DataHandler( dinfo, parentDataHandler->isGlobal() ),
				maxFieldEntries_( size ),
				parentDataHandler_( parentDataHandler ),
				mask_( 0 ),
				numFieldBits_( 0 )
{
	setMaxFieldEntries( size );
}

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

char* FieldDataHandlerBase::parentData( DataId di ) const
{
	return parentDataHandler_->data( di.parentIndex( numFieldBits_ ) );
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
	vector< char* > parents;
	parentDataHandler_->getAllData( parents );
	for ( vector< char* >::iterator i = parents.begin(); 
		i != parents.end(); ++i )
		ret += this->getNumField( *i );
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
	return parentDataHandler_->isDataHere( 
		index.parentIndex( numFieldBits_ )
	); 
}

bool FieldDataHandlerBase::isAllocated() const {
	return parentDataHandler_->isAllocated();
}

unsigned int FieldDataHandlerBase::getMaxFieldEntries() const {
	return maxFieldEntries_;
}

unsigned int FieldDataHandlerBase::fieldMask() const {
	return mask_;
}

unsigned int FieldDataHandlerBase::numFieldBits() const {
	return numFieldBits_;
}

const DataHandler* FieldDataHandlerBase::parentDataHandler() const
{
	return parentDataHandler_;
}

unsigned int FieldDataHandlerBase::getAllData( vector< char* >& data ) const
{
	data.resize( 0 );
	vector< char* > parents;
	parentDataHandler_->getAllData( parents );
	for( vector< char* >::iterator i = parents.begin(); 
		i != parents.end(); ++i ) {
		unsigned int n = this->getNumField( *i );
		for ( unsigned int j = 0; j < n; ++j ) {
			data.push_back( this->lookupField( *i, j ) );
		}
	}	
	return data.size();
}

unsigned int FieldDataHandlerBase::linearIndex( DataId di ) const
{
	return parentDataHandler_->linearIndex( 
			di.parentIndex( numFieldBits_ ) 
		) * maxFieldEntries_ + 
		di.myIndex( mask_ );
}
	

/////////////////////////////////////////////////////////////////////////
// Load balancing
/////////////////////////////////////////////////////////////////////////

bool FieldDataHandlerBase::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

unsigned int FieldDataHandlerBase::syncFieldDim()
{
	unsigned int max = biggestFieldArraySize();
	if ( max != maxFieldEntries_ )
		setMaxFieldEntries( max );
	
	return max;
}

bool FieldDataHandlerBase::execThread( ThreadId thread, DataId di ) const
{
	return parentDataHandler_->execThread( thread, di.parentIndex( numFieldBits_ ) );
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
 * may modify data structures on the parent.
 * The OpFunc belongs to the FieldElement.
 */
void FieldDataHandlerBase::foreach( const OpFunc* f, Element* e,
	const Qinfo* q, const double* arg, 
	unsigned int argSize, unsigned int numArgs ) const
{
	/*
	vector< char* > pa;
	unsigned int numPa = getAllData( pa );
	// I really want getThreadData( pa, threadNum ) to get just the
	// block of data that works on the current thread.
	for( unsigned int i = 0; i < numPa; ++i ) {
		unsigned long long val = i << numFieldBits_;
		unsigned int numFields = this->getNumfield( pa[i] );
		unsigned int argOffset = dinfo()->size * val;
		for( unsigned int j = 0; j < numPa; ++i ) {
			f->op( Eref( e, DataId( val + j ) ), q, arg + argOffset );
			argOffset += dinfo()->size();
		}
	}
	*/

	if ( numArgs > 1 ) {
		unsigned int argOffset = 0;	
		FieldOpFunc fof( f, e, argSize, numArgs, &argOffset );
		ObjId parent = Neutral::parent( Eref( e, 0 ) );
		parentDataHandler_->foreach( &fof, parent.element(), q, 
			arg, argSize * maxFieldEntries_, numArgs / maxFieldEntries_ );
	} else {
		FieldOpFuncSingle fof( f, e );
		ObjId parent = Neutral::parent( Eref( e, 0 ) );
		parentDataHandler_->foreach( &fof, parent.element(), q, 
			arg, argSize, 0 );
	}
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
		setMaxFieldEntries( size );
		return 1;
	} else {
		cout << Shell::myNode() << ": FieldDataHandler::resize: Error: Cannot resize from Field\n";
	}
	return 0;
}

/**
 * Assigns the size of the field array on the specified object.
 * Here we specify the index of the parent object as the first argument.
 */
void FieldDataHandlerBase::setFieldArraySize( 
	unsigned int objectIndex, unsigned int size )
{
	// unsigned int objectIndex = di.value() >> numFieldBits_;
	assert( objectIndex < parentDataHandler_->totalEntries() );

	if ( parentDataHandler_->isDataHere( objectIndex ) ) {
		char* pa = parentDataHandler_->data( objectIndex );
		setNumField( pa, size );
		if ( size > maxFieldEntries_ )
			setMaxFieldEntries( size );
	}
}

void FieldDataHandlerBase::setMaxFieldEntries( unsigned int num )
{
	if ( num == 0 ) {
		cout << "FieldDataHandlerBase::setMaxFieldEntries:: Error: Cannot set to zero\n";
		num = 1;
	}
	unsigned int maxBits = sizeof( long long ) * 8;
	unsigned int i = 0;
	for ( i = 0; i < maxBits; ++i ) {
		if ( ( ( num - 1 ) >> i ) == 0 )
			break;
	}
	maxFieldEntries_ = num;
	numFieldBits_ = i;
	mask_ = ( 1 << i ) - 1;
}

/////////////////////////////////////////////////////////////////////////

/**
 * Looks up the size of the local field array on the specified object
 * Here we use an integer as the index of parent object
 */
unsigned int FieldDataHandlerBase::getFieldArraySize( unsigned int i ) const
{
	assert( i < parentDataHandler_->totalEntries() );
	if ( parentDataHandler_->isDataHere( i ) ) {
		char* pa = parentDataHandler_->data( i );
		return getNumField( pa );
	}
	return 0;
}

/**
 * Looks up the size of the local field array on the specified object
 * Here we use a DataId and trim off the field part to find index of parent
 */
unsigned int FieldDataHandlerBase::getFieldArraySize( DataId di ) const
{
	unsigned int objectIndex = di.value() >> numFieldBits_;
	return getFieldArraySize( objectIndex );
}

/**
 * Looks up the biggest field array size on all nodes.
 */
unsigned int FieldDataHandlerBase::biggestFieldArraySize() const
{
	unsigned int ret = 0;
	vector< char* > parents;
	parentDataHandler_->getAllData( parents );
	for ( vector< char* >::iterator i = parents.begin(); 
		i != parents.end(); ++i ) {
		unsigned int temp = this->getNumField( *i );
		if ( ret < temp )
			ret = temp;
	}

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

//////////////////////////////////////////////////////////////////
