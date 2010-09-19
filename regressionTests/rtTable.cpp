/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Shell.h"

void rtTable()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Id tabid = shell->doCreate( "Table", Id(), "tab", dims );
	assert( tabid != Id() );
	Eref tab = tabid.eref();

	bool ok = SetGet::strSet( tab, "loadXplot", "tab1.xplot,plot1" );
	assert( ok );
	ok = SetGet::strSet( tab, "compareXplot", "tab1.xplot,plot1,rmsr" );
	assert( ok );
	double val = Field< double >::get( tab, "outputValue" );
	assert( ok );
	assert( doubleEq( val, 0.0 ) );

	shell->doDelete( tabid );
	cout << "." << flush;
}
