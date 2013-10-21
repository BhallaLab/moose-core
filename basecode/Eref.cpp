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
	if ( e.i_ == 0 )
		s << e.e_->getName();
	else
		s << e.e_->getName() << "[" << e.i_ << "]";
	return s;
}

char* Eref::data() const
{
	return e_->data( i_ );
}

bool Eref::isDataHere() const
{
	return true;
}

ObjId Eref::objId() const
{
	return ObjId( e_->id(), i_ );
}

Id Eref::id() const
{
	return e_->id();
}

const vector< MsgDigest >& Eref::msgDigest( unsigned int bindIndex ) const
{
	return e_->msgDigest( i_ * e_->cinfo()->numBindIndex() + bindIndex );
}
