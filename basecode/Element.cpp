#include "header.h"

/*
Element::Element( const Data *proto, unsigned int numEntries )
	: finfo_( d_->initClassInfo() )
{
	d_.resize( numEntries );
}
*/

Element::Element( vector< Data* >& d )
	: d_( d ), finfo_( d_[0]->initClassInfo() ), numEntries_( d.size() )
{
	;
}

Element::~Element()
{
	delete[] sendBuf_;
}

void Element::process( const ProcInfo* p )
{
	for ( unsigned int i = 0; i < d_.size(); ++i )
		d_[i]->process( p, Eref( this, i ) );
}

void Element::reinit()
{
	for ( unsigned int i = 0; i < d_.size(); ++i )
		d_[i]->reinit( Eref( this, i ) );
}

void Element::clearQ( const char* buf )
{
	FuncId f = *( static_cast < const FuncId * >( 
		static_cast< const void* >( buf ) ) );
	while ( f != ENDFUNC ) {
		buf += execFunc( f, buf );
		f = *( static_cast < const FuncId * >( 
			static_cast< const void* >( buf ) ) );
	}
}

unsigned int Element::execFunc( FuncId f, const char* buf )
{
	return finfo_[ f ]->op( Eref( this, 0 ), buf + sizeof( FuncId ) );
}

/**
 * This function pushes a synaptic event onto a queue.
 * It should be extended to provide thread safety.
 * This function is thread-safe upto the point where it calls
 * the push function on the target element.
 * Should this be an Eref function with the ElementIndex internal?
 */
void Element::addSpike( unsigned int elementIndex, 
	unsigned int synId, double time )
{
	//Decide if it should use the finfo
	// mutex lock
	// Check if index is busy: bool vector
	// Flag index as busy
	// release mutex
	// do stuff
	// ?unflag index
	// Carry merrily on.
	data( elementIndex )->addSpike( synId, time );
}

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

void Element::send1( Slot slot, unsigned int i, double v )
{
	sendBuf_[ slot + i * numEntries_ ] = v;
}

void Element::send2( Slot slot, unsigned int i, double v1, double v2 )
{
	// Actually we shouldn't use numEntries here, but some value
	// calculated based on the total size of all args that use the send buf
	double* sb = sendBuf_ + slot + i * numEntries_ * 2;
	*sb++ = v1;
	*sb = v2;
}

Data* Element::data( unsigned int index )
{
	assert( index < d_.size() );
	return d_[ index ];
}

const vector< Msg* >& Element::msg( Slot slot ) const
{
	assert( msg_.size() > slot );
	return msg_[ slot ];
}
