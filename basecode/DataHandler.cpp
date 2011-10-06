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

const DataHandler* DataHandler::parentDataHandler() const
{
	return this;
}

bool DataHandler::nodeBalance( unsigned int size )
{
	return this->innerNodeBalance( size, Shell::myNode(), Shell::numNodes() );
}

 /// Overridden in FieldDataHandler.
 unsigned int DataHandler::getFieldArraySize( DataId di ) const
 {
 	return 0;
 }
