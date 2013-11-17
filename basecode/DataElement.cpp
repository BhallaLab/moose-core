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

DataElement::DataElement( Id id, const Cinfo* c, const string& name, 
	unsigned int numData, bool isGlobal )
	:	
		Element( id, c, name )
{
	data_ = c->dinfo()->allocData( numData );
	localNumData_ = numData;
	c->postCreationFunc( id, this );
}

/*
 * Used for copies. Note that it does NOT call the postCreation Func,
 * so FieldElements are copied rather than created by the Cinfo when
 * the parent element is created. This allows the copied FieldElements to
 * retain info from the originals.
 * Note that n is the number of individual  dataEntries that are made.
 */
DataElement::DataElement( Id id, const Element* orig, unsigned int n )
	:	
		Element( id, orig->cinfo(), orig->getName() )
{
	if ( n >= 1 ) {
		data_ = cinfo()->dinfo()->copyData( 
				orig->data( 0 ), orig->numData(), n * orig->numData() );
	}
	localNumData_ = n * orig->numData();
	// cinfo_->postCreationFunc( id, this );
}

// Virtual destructor.
DataElement::~DataElement()
{
	// cout << "deleting element " << getName() << endl;
	cinfo()->dinfo()->destroyData( data_ );
	data_ = 0;
	// The base class destroys the messages.
}

Element* DataElement::copyElement( Id newParent, Id newId, unsigned int n,
		bool toGlobal ) const
{
	return new DataElement( newId, this, n, toGlobal );
}

/////////////////////////////////////////////////////////////////////////
// DataElement info functions
/////////////////////////////////////////////////////////////////////////

// virtual func.
unsigned int DataElement::numLocalData() const
{
	return numLocalData_;
}

// virtual func.
unsigned int DataElement::numField( unsigned int entry ) const
{
	return 1;
}

/////////////////////////////////////////////////////////////////////////
// Data access functions.
/////////////////////////////////////////////////////////////////////////
// virtual func, overridden.
char* DataElement::data( unsigned int rawIndex, unsigned int fieldIndex ) const
{
	assert( rawIndex < numLocalData_ );
	return data_ + ( rawIndex * cinfo()->dinfo()->size() );
}

/**
 * virtual func, overridden.
 * Here we resize the local data. This function would be called by
 * derived classes to do their own data management as per node 
 * decomposition.
 */
void DataElement::resize( unsigned int newNumLocalData )
{
	numLocalData_ = newNumLocalData;
	char* temp = data_;
	data_ = cinfo()->dinfo()->copyData( 
					temp, numLocalData_, newNumLocalData );
	cinfo()->dinfo()->destroyData( temp );
	numLocalData_ = newNumLocalData;
}
