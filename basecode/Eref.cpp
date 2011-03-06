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
	if ( e.i_.data() == 0 && e.i_.field() == 0 )
		s << e.e_->getName();
	else
		s << e.e_->getName() << "[" << e.i_ << "]";
	return s;
}

char* Eref::data() const
{
	return e_->dataHandler()->data( i_ );
}

char* Eref::parentData() const
{
	return e_->dataHandler()->parentDataHandler()->data( i_ );
}

bool Eref::isDataHere() const
{
	return e_->dataHandler()->isDataHere( i_ );
}

ObjId Eref::objId() const
{
	return ObjId( e_->id(), i_ );
}

Id Eref::id() const
{
	return e_->id();
}

unsigned int Eref::linearIndex() const
{
	if ( e_ == 0 || i_ == DataId::bad() )
		return 0;
	if ( i_.data() == 0 && i_.field() == 0 )
		return 0;
	if ( i_.data() == DataId::anyPart() || i_.field() == DataId::anyPart() )
		return 0;

	return e_->dataHandler()->linearIndex( i_ );
}
