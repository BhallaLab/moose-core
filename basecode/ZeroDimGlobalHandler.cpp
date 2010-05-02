/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

ZeroDimGlobalHandler::~ZeroDimGlobalHandler()
{
	dinfo()->destroyData( data_ );
}

void ZeroDimGlobalHandler::process( const ProcInfo* p, Element* e ) const
{
	// We only want one thread to deal with this.
	// In principle we could subdivide the zeroDim cases using
	// the Element Id:
	// if ( p->threadIndexInGroup == e->id()->value() % p->numThreadsinGroup)
	if ( p->threadIndexInGroup == p->numThreadsInGroup - 1 )
		reinterpret_cast< Data* >( data_ )->process( p, Eref( e, 0 ) );
}

bool ZeroDimGlobalHandler::isDataHere( DataId index ) const {
	return 1;
}

bool ZeroDimGlobalHandler::isAllocated() const {
	return data_ != 0;
}

void ZeroDimGlobalHandler::allocate() {
	if ( data_ ) 
		dinfo()->destroyData( data_ );
	data_ = reinterpret_cast< char* >( dinfo()->allocData( 1 ) );
}

DataHandler::iterator ZeroDimGlobalHandler::begin() const
{
	//cout << Shell::myNode() << ": ZeroDimGlobalHandler::begin\n";
	return 0;
}

DataHandler::iterator ZeroDimGlobalHandler::end() const
{
	//cout << Shell::myNode() << ": ZeroDimGlobalHandler::end\n";
	return 1;
}
