/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "FuncOrder.h"
#include "../shell/Shell.h"

LocalDataElement::LocalDataElement( Id id, const Cinfo* c, 
	const string& name, unsigned int numData )
	:	
		DataElement( id, c, name, setDataSize( numData ) )
{;}


/*
 * Used for copies. Note that it does NOT call the postCreation Func,
 * so FieldElements are copied rather than created by the Cinfo when
 * the parent element is created. This allows the copied FieldElements to
 * retain info from the originals.
 * n is the number of new entries made.
 */
LocalDataElement::LocalDataElement( Id id, const Element* orig, 
				unsigned int n )
	:	
		DataElement( id, orig, setDataSize( n ), 
		( 1 + (n - 1 ) / Shell::numNodes() ) * Shell::myNode() )
{;}

// Virtual destructor, but the base DataElement does the needful
LocalDataElement::~LocalDataElement()
{;}

// This is somewhat problematic to do as a low-level function. Will need
// to look up all other nodes to get their conent
Element* LocalDataElement::copyElement( Id newParent, Id newId, 
		unsigned int n, bool toGlobal ) const
{
	if ( toGlobal ) {
		cout << "Don't know yet how to copy LocalDataElement to global\n";
		assert( 0 );
		return 0;
	} else {
		return new LocalDataElement( newId, this, n );
	}
}


/////////////////////////////////////////////////////////////////////////
// LocalDataElement info functions
/////////////////////////////////////////////////////////////////////////

// virtual func.
unsigned int LocalDataElement::numData() const
{
	return numData_;
}

unsigned int LocalDataElement::localDataStart() const
{
	return numPerNode_ * Shell::myNode();
}

// localNumData() is inherited from DataElement.

unsigned int LocalDataElement::getNode( unsigned int dataId ) const {
	// Assume numData = 95. DataId = 0-9: 0, DataId=80-89:8, DataId >= 90:9
	if ( dataId == ALLDATA ) {
		if ( numLocalData() > 0 ) {
			return Shell::myNode();
		} else {
			return 0; // Sure to have some data on node 0.
		}
	}
	return dataId / numPerNode_;
}

unsigned int LocalDataElement::rawIndex( unsigned int dataId ) const {
	return dataId % numPerNode_;
}

// Utility function for computing the data size.
// Returns the number of entries on current node.
// As a side effect assigns the total numData_ for all nodes, and the
// numPerNode_ helper field.
unsigned int LocalDataElement::setDataSize( unsigned int numData )
{
	numData_ = numData;
	numPerNode_ = 1 + (numData_ -1 ) / Shell::numNodes();
	unsigned int numLocalData = numPerNode_;
	if ( Shell::myNode() == Shell::numNodes() -1 )
		numLocalData = numData_ - numPerNode_ * ( Shell::numNodes() - 1 );
	return numLocalData;
}

// virtual func, overridden.
void LocalDataElement::resize( unsigned int newNumData )
{
	DataElement::resize( setDataSize( newNumData ) ); 
}

unsigned int LocalDataElement::getNumOnNode( unsigned int node ) const
{
	if ( Shell::myNode() == Shell::numNodes() -1 )
		return numData_ - numPerNode_ * ( Shell::numNodes() - 1 );

	return numPerNode_;
}

/*
// virtual func, overridden.
char* LocalDataElement::dataFromDataId( unsigned int dataId, unsigned int fieldIndex ) const
{
	assert( dataId >= localDataStart() );
	dataId -= localDataStart();
	assert( dataId < numLocalData_ );
	return data_ + ( dataId * cinfo()->dinfo()->size() );
}
*/
