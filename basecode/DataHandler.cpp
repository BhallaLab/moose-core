/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"

DataHandler::DataHandler( const DinfoBase* dinfo )
	: dinfo_( dinfo )
{;}

DataHandler::~DataHandler()
{;}

const DataHandler* DataHandler::parentDataHandler() const
{
	return this;
}

void DataHandler::setFieldArraySize( 
	unsigned int objectIndex, unsigned int size )
{
	; // Default operation does nothing. Used only in FieldDataHandlers.
}

unsigned int DataHandler::getFieldArraySize( unsigned int objectIndex )
	const
{
	return 0; // Default operation.
}

unsigned int DataHandler::linearIndex( const DataId& d ) const
{
	return d.data();
}

DataId DataHandler::dataId( unsigned int i ) const
{
	return DataId( i );
}

/**
 * Default operations for non-FieldDataHandlers
 */
void DataHandler::setFieldDimension( unsigned int size )
{
	;
}

unsigned int DataHandler::getFieldDimension() const
{
	return 0;
}


bool DataHandler::nodeBalance( unsigned int size )
{
	return this->innerNodeBalance( size, Shell::myNode(), Shell::numNodes() );
}
