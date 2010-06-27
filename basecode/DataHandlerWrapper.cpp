/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DataHandlerWrapper.h"

DataHandlerWrapper::DataHandlerWrapper( const DataHandler* parentHandler )
	: DataHandler( parentHandler->dinfo() ), parent_( parentHandler )
{
}

DataHandlerWrapper::~DataHandlerWrapper()
{;} // This is the key function. It deletes itself but does not touch parent

DataHandler* DataHandlerWrapper::copy( unsigned int n, bool toGlobal ) 
	const
{
	return parent_->copy( n, toGlobal );
}

char* DataHandlerWrapper::data( DataId index ) const {
	return parent_->data( index );
}

/**
 * Returns the data at one level up of indexing.
 * Here there isn't any.
 */
char* DataHandlerWrapper::data1( DataId index ) const {
	return parent_->data1( index );
}

/**
 * Returns the number of data entries.
 */
unsigned int DataHandlerWrapper::numData() const {
	return parent_->numData();
}

/**
 * Returns the number of data entries at index 1.
 */
unsigned int DataHandlerWrapper::numData1() const {
	return parent_->numData1();
}

/**
 * Returns the number of data entries at index 2, if present.
 * For regular Elements and 1-D arrays this is always 1.
 */
 unsigned int DataHandlerWrapper::numData2( unsigned int index1 ) const
 {
	return parent_->numData2( index1 );
 }

/**
 * Returns the number of dimensions of the data.
 */
unsigned int DataHandlerWrapper::numDimensions() const {
	return parent_->numDimensions();
}

void DataHandlerWrapper::process( const ProcInfo* p, Element* e, FuncId fid ) const
{
	parent_->process( p, e, fid );
}

/**
 * Assign # of entries in dimension 1. 
 * Ignore here, as it is readonly.
 */
void DataHandlerWrapper::setNumData1( unsigned int size ) {
	;
}
/**
 * Assigns the sizes of all array field entries at once.
 * This is ignored as it is readonly.
 */
void DataHandlerWrapper::setNumData2( unsigned int start,
	const vector< unsigned int >& sizes ) {
	;
}

unsigned int DataHandlerWrapper::getNumData2( vector< unsigned int >& sizes ) const {
	return parent_->getNumData2( sizes );
}

bool DataHandlerWrapper::isDataHere( DataId index ) const {
	return parent_->isDataHere( index );
}

bool DataHandlerWrapper::isAllocated() const {
	return parent_->isAllocated();
}

void DataHandlerWrapper::allocate() {
	assert( 0 );
	return;
}

bool DataHandlerWrapper::isGlobal() const
{
	return parent_->isGlobal();
}

DataHandler::iterator DataHandlerWrapper::begin() const
{
	return parent_->begin();
}

DataHandler::iterator DataHandlerWrapper::end() const
{
	return parent_->end();
}

/**
 * This is relevant only for the 2 D cases like
 * FieldDataHandlers.
 */
unsigned int DataHandlerWrapper::startDim2index() const {
	return parent_->startDim2index();
}

void DataHandlerWrapper::setData( char* data, unsigned int numData )
{;}
