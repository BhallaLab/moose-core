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


const DataId DataId::bad()
{
  static DataId bad_( ~0U );
  return bad_;
}

const DataId DataId::any( ~1U );
const DataId DataId::globalField( ~2U );

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
