#include "header.h"

Eref::Eref( Element* e, unsigned int index )
			: e_( e ), i_( index )
{
	;
}

double Eref::sumBuf( SyncId slot )
{
	return e_->sumBuf( slot, i_ );
}

double Eref::prdBuf( SyncId slot, double v )
{
	return e_->prdBuf( slot, i_, v );
}

double Eref::oneBuf( SyncId slot )
{
	return e_->oneBuf( slot, i_ );
}

double* Eref::getBufPtr( SyncId slot )
{
	return e_->getBufPtr( slot, i_ );
}

Data* Eref::data()
{
	return e_->data( i_ );
}

/*
void Eref::sendSpike( Slot src, double t )
{
	// msg should do the iteration internally, passing just the
	// double 
	const vector< Msg* >& v = e_->msg( src );
	for ( vector< Msg* >::const_iterator i = v.begin(); i != v.end(); ++i )
		( *i )->addSpike( i_, t );
}
*/

void Eref::ssend1( SyncId src, double v )
{
	e_->ssend1( src, i_, v );
}

void Eref::ssend2( SyncId src, double v1, double v2 )
{
	e_->ssend2( src, i_, v1, v2 );
}

void Eref::asend( ConnId conn, FuncId func, const char* arg, 
			unsigned int size ) const
{
	// e_->conn( conn ).asend( e_, Qinfo( func, i_, size ), arg );
	Qinfo q( func, i_, size );
	e_->conn( conn ).asend( e_, q, arg );
}

void Eref::tsend( ConnId conn, FuncId func, Id target, const char* arg, 
			unsigned int size ) const
{
	// e_->conn( conn ).asend( e_, Qinfo( func, i_, size, 1 ), arg );
	Qinfo q( func, i_, size, 1 );
	e_->conn( conn ).tsend( e_, target, q, arg );
}
