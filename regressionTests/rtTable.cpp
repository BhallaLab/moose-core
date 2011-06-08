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

	Id tabentryId ( tabid.value() + 1 );
	ObjId tabentry( tabentryId, DataId( 0, 3 ) );

	// Check loading
	bool ok = SetGet::strSet( tabid, "loadXplot", "tab1.xplot,plot1" );
	assert( ok );

	unsigned int size = Field< unsigned int >::get( tabid, "size" );
	assert( size == 10 );

	double val = Field< double >::get( tabentry, "value" );
	assert( doubleEq( val, 3.0 ) );

	// Check rmsratio comparison
	ok = SetGet::strSet( tabid, "compareXplot", "tab1.xplot,plot1,rmsr" );
	assert( ok );
	val = Field< double >::get( tabid, "outputValue" );
	assert( doubleEq( val, 0.0 ) );


	// Check loading of second plot
	ok = SetGet::strSet( tabid, "loadXplot", "tab1.xplot,plot2" );
	assert( ok );

	size = Field< unsigned int >::get( tabid, "size" );
	assert( size == 9 );
	
	val = Field< double >::get( tabentry, "value" );
	assert( doubleEq( val, 3.1 ) );

	// Check rmsdiff comparison
	ok = SetGet::strSet( tabid, "compareXplot", "tab1.xplot,plot1,rmsd" );
	assert( ok );
	val = Field< double >::get( tabid, "outputValue" );
	assert( doubleEq( val, 0.1 ) );

	// Check loading of 2-column data
	ok = SetGet::strSet( tabid, "loadXplot", "tab2_2col.xplot,plot1" );
	assert( ok );

	size = Field< unsigned int >::get( tabid, "size" );
	assert( size == 8 );

	val = Field< double >::get( tabentry, "value" );
	assert( doubleEq( val, 13.0 ) );

	// Check loading of 3-column data
	ok = SetGet::strSet( tabid, "loadXplot", "tab2_2col.xplot,plot2" );
	assert( ok );

	size = Field< unsigned int >::get( tabid, "size" );
	assert( size == 5 );

	val = Field< double >::get( tabentry, "value" );
	assert( doubleEq( val, 3.1 ) );

	// Check rmsdiff comparison with 2 and 3 col data
	ok = SetGet::strSet( tabid, "compareXplot", "tab2_2col.xplot,plot1,rmsd" );
	assert( ok );
	val = Field< double >::get( tabid, "outputValue" );
	assert( doubleEq( val, 9.9 ) );

	/////////////////////////////////////////////////////////////////////
	shell->doDelete( tabid );
	cout << "." << flush;
}
