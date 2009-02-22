#include "header.h"

Eref::Eref( Element* e, unsigned int index )
			: e_( e ), i_( index )
{
	;
}

double Eref::sumBuf( Slot slot )
{
	return e_->sumBuf( slot, i_ );
}

double Eref::prdBuf( Slot slot, double v )
{
	return e_->prdBuf( slot, i_, v );
}

double Eref::oneBuf( Slot slot )
{
	return e_->oneBuf( slot, i_ );
}

double* Eref::getBufPtr( Slot slot )
{
	return e_->getBufPtr( slot, i_ );
}

Data* Eref::data()
{
	return e_->data( i_ );
}

void Eref::sendSpike( Slot src, double t )
{
	// msg should do the iteration internally, passing just the
	// double 
	const vector< Msg* >& v = e_->msg( src );
	for ( vector< Msg* >::const_iterator i = v.begin(); i != v.end(); ++i )
		( *i )->addSpike( i_, t );
}

void Eref::send1( Slot src, double v )
{
	e_->send1( src, i_, v );
}

void Eref::send2( Slot src, double v1, double v2 )
{
	e_->send2( src, i_, v1, v2 );
}
