#include "header.h"

Element::Element( Data *d )
	: d_( d ), finfo_( d_->initClassInfo() )
{
	;
}

void Element::process( const ProcInfo* p )
{
	d_->process( p, Eref( this, 0 ) );
}

void Element::reinit()
{
	d_->reinit( Eref( this, 0 ) );
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
void Element::pushQ( unsigned int elementIndex, 
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
	data( elementIndex )->pushQ( synId, time );
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

Data* Element::data( unsigned int index )
{
	return d_;
}

const vector< Msg* >& Element::msg( Slot slot ) const
{
	assert( msg_.size() > slot );
	return msg_[ slot ];
}
