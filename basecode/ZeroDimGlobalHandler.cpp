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

DataHandler* ZeroDimGlobalHandler::copy( unsigned int n, bool toGlobal ) 
	const
{
	if ( toGlobal ) {
		if ( n <= 1 ) { // Don't need to boost dimension.
			ZeroDimGlobalHandler* ret = new ZeroDimGlobalHandler( dinfo() );
			ret->data_ = dinfo()->copyData( data_, 1, 1 );
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


void ZeroDimGlobalHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{
	// We only want one thread to deal with this.
	// In principle we could subdivide the zeroDim cases using
	// the Element Id:
	// if ( p->threadIndexInGroup == e->id()->value() % p->numThreadsinGroup)
	if ( p->threadIndexInGroup == p->numThreadsInGroup - 1 ) {
		const OpFunc* f = e->cinfo()->getOpFunc( fid );
		const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
		assert( pf );
		pf->proc( data_, Eref( e, 0 ), p );
	//	reinterpret_cast< Data* >( data_ )->process( p, Eref( e, 0 ) );
	}
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
