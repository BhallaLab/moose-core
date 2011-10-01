/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "DataId.h"

const DataId DataId::bad( ~unsigned long long( 0 ) );
const DataId DataId::any( ~unsigned long long( 1 ) );
const DataId DataId::globalField( ~unsigned long long( 2 ) );

ostream& operator <<( ostream& s, const DataId& d )
{
	s << d.index_;
	return s;
}

istream& operator >>( istream& s, DataId& d )
{
	s >> d.index_;
	return s;
}
