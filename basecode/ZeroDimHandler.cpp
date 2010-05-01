/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

ZeroDimHandler::~ZeroDimHandler()
{
	dinfo()->destroyData( data_ );
}

void ZeroDimHandler::process( const ProcInfo* p, Element* e ) const
{
	if ( Shell::myNode() == 0 && 
		p->threadIndexInGroup == p->numThreadsInGroup - 1 )
		reinterpret_cast< Data* >( data_ )->process( p, Eref( e, 0 ) );
}

/**
 * Returns true if the node decomposition has the data on the
 * current node
 */
bool ZeroDimHandler::isDataHere( DataId index ) const {
	return ( Shell::myNode() == 0 );
}

bool ZeroDimHandler::isAllocated() const {
	return data_ != 0;
}

void ZeroDimHandler::allocate() {
	if ( data_ ) 
		dinfo()->destroyData( data_ );
	data_ = reinterpret_cast< char* >( dinfo()->allocData( 1 ) );
}

DataHandler::iterator ZeroDimHandler::end() const
{
	// cout << Shell::myNode() << ": ZeroDimHandler Iterator\n";
	return ( Shell::myNode() == 0 );
}
