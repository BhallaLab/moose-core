/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <algorithm>
#include "MsgSrc.h"
#include "MsgDest.h"
#include "SimpleElement.h"
#include "send.h"
#include "DerivedFtype.h"
#include "Ftype2.h"
#include "setget.h"

bool set( Element* e, const Finfo* f )
{
	if ( Ftype0::isA( f->ftype() ) ) {
		RecvFunc rf = f->recvFunc();
		if ( rf ) {
			Conn c( e, MAXUINT );
			rf( c );
			return 1;
		}
	}
	return 0;
}

bool set( Element* e, const string& f )
{
	const Finfo* finfo = e->findFinfo( f );
	if ( finfo == 0 ) {
		cout << "Error: set( " << e->name() << ", " << f <<
				" ): Finfo not found\n";
		return 0;
	}
	return set( e, finfo );
}
