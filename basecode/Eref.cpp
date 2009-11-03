#include "header.h"

Eref::Eref( Element* e, DataId index )
			: e_( e ), i_( index )
{
	;
}

ostream& operator <<( ostream& s, const Eref& e )
{
	s << e.e_ << "[" << e.i_ << "]";
	return s;
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

char* Eref::data()
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

void Eref::asend( ConnId conn, unsigned int funcIndex, const char* arg, 
			unsigned int size ) const
{
	// e_->conn( conn ).asend( e_, Qinfo( func, i_, size ), arg );
	Qinfo q( e_->getTargetFunc( funcIndex ), i_, size );
	e_->conn( conn ).asend( e_, q, arg );
}

/**
 * Need to sort out: do we use FuncId here (confusing) or funcIndex
 * (when would it be set up ?)
 */
void Eref::tsend( ConnId conn, FuncId func, Id target, const char* arg, 
			unsigned int size ) const
{
	// e_->conn( conn ).asend( e_, Qinfo( func, i_, size, 1 ), arg );
	Qinfo q( func, i_, size, 1 );
	e_->conn( conn ).tsend( e_, target, q, arg );
}
