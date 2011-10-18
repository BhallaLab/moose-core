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

DataHandler::DataHandler( 
	const DinfoBase* dinfo, const vector< DimInfo >& dims,
		unsigned short pathDepth, bool isGlobal )
	: 
		dims_( dims ),
		pathDepth_( pathDepth ),
		isGlobal_( isGlobal ),
		dinfo_( dinfo )
{ 
	totalEntries_ = 1;
	for ( unsigned int i = 0; i < dims.size(); ++i )
		totalEntries_ *= dims[i].size;
}

DataHandler::~DataHandler()
{;}

//////////////////////////////////////////////////////////////////////
// Information functions
//////////////////////////////////////////////////////////////////////

bool DataHandler::isGlobal() const
{
	return isGlobal_;
}

char* DataHandler::parentData( DataId index ) const
{
	return 0;
}

const DataHandler* DataHandler::parentDataHandler() const
{
	return this;
}

unsigned int DataHandler::totalEntries() const
{
	return totalEntries_;
}

unsigned int DataHandler::numDimensions() const
{
	return dims_.size();
}

unsigned short DataHandler::pathDepth() const
{
	return pathDepth_;
}

unsigned int DataHandler::sizeOfDim( unsigned int dim ) const
{
	if ( dim < dims_.size() )
		return dims_[dim].size;
	return 0;
}

const vector< DimInfo >& DataHandler::dims() const
{
	return dims_;
}

bool DataHandler::changeDepth( unsigned short newDepth )
{
	short deltaDepth = newDepth - pathDepth_;
	if ( deltaDepth == 0 ) return 1;

	if ( deltaDepth < 0 ) {
		for ( unsigned int i = 0; i < dims_.size(); ++i ) {
			short d = dims_[i].depth;
			if ( d + deltaDepth < 1 ) 
				return 0;
		}
	}

	for ( unsigned int i = 0; i < dims_.size(); ++i ) {
		dims_[i].depth += deltaDepth;
	}
	pathDepth_ += deltaDepth;
	return 1;
}
/////////////////////////////////////////////////////////////////////

bool DataHandler::nodeBalance( unsigned int size )
{
	return this->innerNodeBalance( size, Shell::myNode(), Shell::numNodes() );
}

unsigned int DataHandler::syncFieldDim()
{
	return 0;
}

 /// Overridden in FieldDataHandler.
 unsigned int DataHandler::getFieldArraySize( unsigned int i ) const
 {
 	return 0;
 }

 /// Overridden in FieldDataHandler
 unsigned int DataHandler::fieldMask() const
 {
 	return 0;
 }
