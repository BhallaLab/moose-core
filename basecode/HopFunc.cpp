/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "../mpi/PostMaster.h"

const OpFunc* OpFunc0Base::makeHopFunc( unsigned int bindIndex ) const
{
	return new HopFunc0( bindIndex );
}

double* addToBuf( const Eref& e, unsigned int bindIndex, unsigned int size )
{
	static PostMaster* p = reinterpret_cast< PostMaster* >( 
					ObjId( 3 ).data(0) );
	return p->addToSendBuf( e, bindIndex_, size );
}
