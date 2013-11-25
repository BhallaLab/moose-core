/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "HopFunc.h"
#include "../mpi/PostMaster.h"

double* addToBuf( const Eref& e, HopIndex hopIndex, unsigned int size )
{
	static ObjId oi( 3 );
	static PostMaster* p = reinterpret_cast< PostMaster* >( oi.data() );
	if ( hopIndex.hopType() == MooseSendHop )
		return p->addToSendBuf( e, hopIndex.bindIndex(), size );
	else if ( hopIndex.hopType() == MooseSetHop ) {
		return p->addToSetBuf( e, hopIndex.bindIndex(), size );
	}
	assert( 0 ); // Should not get here.
	return 0;
}

void dispatchBuffers( const Eref& e, HopIndex hopIndex )
{
	static ObjId oi( 3 );
	static PostMaster* p = reinterpret_cast< PostMaster* >( oi.data() );
	if ( hopIndex.hopType() == MooseSetHop ) {
		p->dispatchSetBuf( e );
	}
	// More complicated stuff for get operations.
}
