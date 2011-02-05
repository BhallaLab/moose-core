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
#include "ReduceFinfo.h"

ReduceBase::ReduceBase() // To keep vectors happy
	: er_( Id().eref() ), rfb_( 0 )
{;}

ReduceBase::ReduceBase( const Eref& er, const ReduceFinfoBase* rfb )
	: er_( er ), rfb_( rfb )
{;}

ReduceBase::~ReduceBase()
{;}

bool ReduceBase::sameEref( const ReduceBase* other ) const
{
	return ( rfb_ == other->rfb_ && 
		er_.element() == other->er_.element() && er_.index() == other->er_.index()  );
}

bool ReduceBase::reduceNodes()
{
#ifdef USE_MPI
	char* recvBuf = new char[ Shell::numNodes() * this->dataSize() ];
	MPI_Allgather( this->data(), this->dataSize(), MPI_CHAR, 
		recvBuf, this->dataSize(), MPI_CHAR, 
		MPI_COMM_WORLD );
	for ( unsigned int i = 0; i < numNodes_; ++i ) {
		this->tertiaryReduce( recvBuf[ i * this->dataSize() ] );
	}
	delete[] recvBuf;
#endif
	
	return er_.isDataHere(); // Do we need to assign the result here?
}

void ReduceBase::assignResult() const
{
	rfb_->digestReduce( er_, this );
}

/////////////////////////////////////////////////////////////////////////

// The function is set up by a suitable SetGet templated wrapper.
ReduceStats::ReduceStats( const Eref& er, const ReduceFinfoBase* rfb, 
	const GetOpFuncBase< double >* gof )
	: 
		ReduceBase( er, rfb ),
		gof_( gof )
{
	data_.sum_ = 0.0;
	data_.sumsq_ = 0.0;
	data_.count_ = 0 ;
}

ReduceStats::~ReduceStats()
{;}

const char* ReduceStats::data() const
{
	return reinterpret_cast< const char* >( &data_ );
}

unsigned int ReduceStats::dataSize() const
{
	return sizeof( ReduceDataType );
}

void ReduceStats::primaryReduce( const Eref& e )
{
	double x = gof_->reduceOp( e );
	data_.sum_ += x;
	data_.sumsq_ += x * x;
	data_.count_++;
}

// Must not use other::func_
void ReduceStats::secondaryReduce( const ReduceBase* other )
{
	const ReduceStats* r = dynamic_cast< const ReduceStats* >( other );
	assert( r );
	data_.sum_ += r->data_.sum_;
	data_.sumsq_ += r->data_.sumsq_;
	data_.count_ += r->data_.count_;
}

// Must not use other::func_
void ReduceStats::tertiaryReduce( const char* other )
{
	const ReduceDataType* d = reinterpret_cast< const ReduceDataType* >( other );
	assert( d );
	data_.sum_ += d->sum_;
	data_.sumsq_ += d->sumsq_;
	data_.count_ += d->count_;
}

double ReduceStats::sum() const
{
	return data_.sum_;
}

double ReduceStats::sumsq() const
{
	return data_.sumsq_;
}

unsigned int ReduceStats::count() const
{
	return data_.count_;
}
