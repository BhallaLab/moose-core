#include "header.h"

void Element::process( const ProcInfo* p )
{
	d_->process( p, Eref( this, 0 ) );
}

void Element::reinit()
{
	d_->reinit( Eref( this, 0 ) );
}

/*
void Element::clearQ()
{
	;
}
*/

double Element::sumBuf( Slot slot, unsigned int i )
{
	// unsigned int offset = i * numData_ + slot;
	unsigned int offset = slot;
	assert( offset + 1 < procBufRange_.size() );
	vector< double* >::iterator begin = procBuf_.begin() + 
		procBufRange_[offset];
	vector< double* >::iterator end = procBuf_.begin() + 
		procBufRange_[offset + 1];
	double ret = 0.0;
	for ( vector< double* >::iterator i = begin; i != end; ++i )
		ret += **i;
	return ret;
}

double Element::prdBuf( Slot slot, unsigned int i, double v )
{
	// unsigned int offset = i * numData_ + slot;
	unsigned int offset = slot;
	assert( offset + 1 < procBufRange_.size() );
	vector< double* >::iterator begin = procBuf_.begin() + 
		procBufRange_[offset];
	vector< double* >::iterator end = procBuf_.begin() + 
		procBufRange_[offset + 1];
	for ( vector< double* >::iterator i = begin; i != end; ++i )
		v *= **i;
	return v;
}

double Element::oneBuf( Slot slot, unsigned int i )
{
	// unsigned int offset = i * numData_ + slot;
	unsigned int offset = slot;
	assert( offset + 1 < procBufRange_.size() );
	return *procBuf_[ procBufRange_[ offset ] ];
}

double* Element::getBufPtr( Slot slot, unsigned int i )
{
	// unsigned int offset = i * numData_ + slot;
	unsigned int offset = slot;
	assert( offset + 1 < procBufRange_.size() );
	return procBuf_[ procBufRange_[ offset ] ];
}
