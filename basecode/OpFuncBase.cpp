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

const unsigned char MooseSendHop = 0;
const unsigned char MooseSetHop = 1;
const unsigned char MooseSetVecHop = 2;
const unsigned char MooseGetHop = 4;
const unsigned char MooseGetVecHop = 5;
const unsigned char MooseReturnHop = 8;
const unsigned char MooseTestHop = 255;

vector< OpFunc* >& OpFunc::ops()
{
	static vector< OpFunc* > op;
	return op;
}

OpFunc::OpFunc()
{
	opIndex_ = ops().size();
	ops().push_back( this );
}

const OpFunc* OpFunc0Base::makeHopFunc( HopIndex hopIndex ) const
{
	return new HopFunc0( hopIndex );
}

void OpFunc0Base::opBuffer( const Eref& e, double* buf ) const
{
	op( e );
}

const OpFunc* OpFunc::lookop( unsigned int opIndex )
{
	assert ( opIndex < ops().size() );
	return ops()[ opIndex ];
}
