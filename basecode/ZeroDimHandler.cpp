/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

ZeroDimHandler::ZeroDimHandler( const DinfoBase* dinfo )
	: ZeroDimGlobalHandler( dinfo )
{;}

ZeroDimHandler::ZeroDimHandler( const ZeroDimHandler* other )
	: ZeroDimGlobalHandler( other->dinfo() )
{
	data_ = dinfo()->copyData( other->data_, 1, 1 );
}

ZeroDimHandler::~ZeroDimHandler()
{
	dinfo()->destroyData( data_ );
}

DataHandler* ZeroDimHandler::copy() const
{
	return ( new ZeroDimHandler( this ) );
}

DataHandler* ZeroDimHandler::copyExpand( unsigned int copySize ) const
{
	OneDimHandler* ret = new OneDimHandler( dinfo() );
	vector< unsigned int > dims( 1, copySize );
	ret->resize( dims );
	for ( iterator i = ret->begin(); i != ret->end(); ++i ) {
		char* temp = *i;
		memcpy( temp, data_, dinfo()->size() );
	}
	return ret;
}

DataHandler* ZeroDimHandler::copyToNewDim( unsigned int newDimSize ) const
{
	return copyExpand( newDimSize );
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

DataHandler::iterator ZeroDimHandler::end() const
{
	// cout << Shell::myNode() << ": ZeroDimHandler Iterator\n";
	// end is same as begin except on node 1.
	return iterator( this, ( Shell::myNode() == 0 ) );
}
