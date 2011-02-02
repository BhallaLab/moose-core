/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ReduceBase.h"

ReduceBase::ReduceBase()
{;}

ReduceBase::~ReduceBase()
{;}
/////////////////////////////////////////////////////////////////////////

// The function is set up by a suitable SetGet templated wrapper.
ReduceStats::ReduceStats( const GetOpFuncBase< double >* gof )
	: 
		sum_( 0.0 ),
		sumsq_( 0.0 ),
		count_( 0 ),
		gof_( gof )
{;}

ReduceStats::~ReduceStats()
{;}

void ReduceStats::primaryReduce( const Eref& e )
{
	double x = gof_->reduceOp( e );
	sum_ += x;
	sumsq_ += x * x;
	count_++;
}

// Must not use other::func_
void ReduceStats::secondaryReduce( const ReduceBase* other )
{
	const ReduceStats* r = dynamic_cast< const ReduceStats* >( other );
	assert( r );
	sum_ += r->sum_;
	sumsq_ += r->sumsq_;
	count_ += r->count_;
}
