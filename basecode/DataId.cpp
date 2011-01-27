/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <iostream>
using namespace std;
#include "DataId.h"

ostream& operator <<( ostream& s, const DataId& d )
{
	s << d.data_ << ":" << d.field_;
	return s;
}

istream& operator >>( istream& s, DataId& d )
{
	string dummy;
	s >> d.data_ >> dummy >> d.field_;
	return s;
}

const DataId& DataId::bad() {
	static DataId bad_( ~0U, ~0U );
	return bad_;
}

const DataId& DataId::any() {
	static DataId any_( ~1U, ~1U );
	return any_;
}

const unsigned int DataId::anyPart()
{
	return ~1U;
}
