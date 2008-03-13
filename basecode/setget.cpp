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

#include "ProcInfo.h"
#include "SetConn.h"
#include "DerivedFtype.h"

bool set( Eref e, const Finfo* f )
{
	if ( Ftype0::isA( f->ftype() ) ) {
		RecvFunc rf = f->recvFunc();
		if ( rf ) {
			SetConn c( e );
			rf( &c );
			return 1;
		}
	}
	return 0;
}

bool set(  Eref e, const string& f )
{
	const Finfo* finfo = e.e->findFinfo( f );
	if ( finfo == 0 ) {
		cout << "Error: set( " << e.e->name() << "." << e.i << ", " << f <<
				" ): Finfo not found\n";
		return 0;
	}
	return set( e, finfo );
}
