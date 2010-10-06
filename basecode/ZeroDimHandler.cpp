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

DataHandler* ZeroDimHandler::copy( unsigned int n, bool toGlobal ) 
	const
{
	if ( Shell::myNode() > 0 ) {
		cout << Shell::myNode() << ": Error: ZeroDimHandler::copy: Should not call on multinode systems\n";
		return 0;
	}
	if ( toGlobal ) {
		if ( n <= 1 ) { // Don't need to boost dimension.
			ZeroDimGlobalHandler* ret = new ZeroDimGlobalHandler( dinfo() );
			ret->setData( dinfo()->copyData( data_, 1, 1 ), 1);
			return ret;
		} else {
			OneDimGlobalHandler* ret = new OneDimGlobalHandler( dinfo() );
			ret->setData( dinfo()->copyData( data_, 1, n ), n );
			return ret;
		}
	} else {
		if ( n <= 1 ) { // do copy only on node 0.
			ZeroDimHandler* ret = new ZeroDimHandler( dinfo() );
			if ( Shell::myNode() == 0 ) {
				ret->setData( dinfo()->copyData( data_, 1, 1 ), 1 );
			}
			return ret;
		} else {
			OneDimHandler* ret = new OneDimHandler( dinfo() );
			ret->setNumData1( n );
			unsigned int size = ret->end() - ret->begin();
			if ( size > 0 )
			ret->setData( dinfo()->copyData( data_, 1, size ), size );
			return ret;
		}
	}
}

void ZeroDimHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{
	if ( Shell::myNode() == 0 && 
		p->threadIndexInGroup == p->numThreadsInGroup - 1 ) {
		// reinterpret_cast< Data* >( data_ )->process( p, Eref( e, 0 ) );

		const OpFunc* f = e->cinfo()->getOpFunc( fid );
		const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
		assert( pf );
		pf->proc( data_, Eref( e, 0 ), p );
	}
}

char* ZeroDimHandler::data( DataId index ) const {
	return data_;
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
