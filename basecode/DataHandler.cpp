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

DataHandler::DataHandler( const DinfoBase* dinfo, bool isGlobal )
	: isGlobal_( isGlobal ),
	dinfo_( dinfo )
{;}

DataHandler::~DataHandler()
{;}


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
