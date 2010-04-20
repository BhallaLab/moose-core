/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

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
	return e_->sumBuf( slot, i_.data() );
}

double Eref::prdBuf( SyncId slot, double v )
{
	return e_->prdBuf( slot, i_.data(), v );
}

double Eref::oneBuf( SyncId slot )
{
	return e_->oneBuf( slot, i_.data() );
}

double* Eref::getBufPtr( SyncId slot )
{
	return e_->getBufPtr( slot, i_.data() );
}

char* Eref::data()
{
	return e_->dataHandler()->data( i_ );
}

char* Eref::data1()
{
	return e_->dataHandler()->data1( i_ );
}

void Eref::ssend1( SyncId src, double v )
{
	e_->ssend1( src, i_.data(), v );
}

void Eref::ssend2( SyncId src, double v1, double v2 )
{
	e_->ssend2( src, i_.data(), v1, v2 );
}

bool Eref::isDataHere() const
{
	return e_->dataHandler()->isDataHere( i_ );
}
