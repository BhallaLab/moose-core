/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

Slot::Slot( ConnId conn, const Cinfo* c, const string& funcName )
	: conn_( conn )
{
	// func_ = c->findFunc( funcName );
}

Slot::Slot( ConnId conn, FuncId func )
	: conn_( conn ), func_( func )
{
	// func_ = c->findFunc( funcName );
}

void Slot0::send( Eref e )
{
	e.asend( conn_, func_, 0, 0 );
}
