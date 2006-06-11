/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include <fstream>
#include "header.h"

static const char* schedResponse[] = {
"Process: Tock t0	t = 0, dt = 1",
"Process: Tock t1	t = 0, dt = 2",
"Process: Tock t2	t = 0, dt = 5",
"Process: Tock t0	t = 1, dt = 1",
"Process: Tock t0	t = 2, dt = 1",
"Process: Tock t1	t = 2, dt = 2",
"Process: Tock t0	t = 3, dt = 1",
"Process: Tock t0	t = 4, dt = 1",
"Process: Tock t1	t = 4, dt = 2",
"Process: Tock t0	t = 5, dt = 1",
"Process: Tock t2	t = 5, dt = 5",
"Process: Tock t0	t = 6, dt = 1",
"Process: Tock t1	t = 6, dt = 2",
"Process: Tock t0	t = 7, dt = 1",
"Process: Tock t0	t = 8, dt = 1",
"Process: Tock t1	t = 8, dt = 2",
"Process: Tock t0	t = 9, dt = 1",
"Process: Tock t0	t = 10, dt = 1",
"Process: Tock t1	t = 10, dt = 2",
"Process: Tock t2	t = 10, dt = 5"
};

// Assumes that the infrastructure of object creation and function
// calls and Shell access all exists.
void testScheduling()
{
	cout << "Testing Scheduling\n";
	string response;

	Element* cj = Field( "/sched/cj/name" ).getElement();
	Cinfo::find("ClockTick")->create( "ct0", cj );
	Element* ct1 = Cinfo::find("ClockTick")->create( "ct1", cj );
	Element* ct2 = Cinfo::find("ClockTick")->create( "ct2", cj );
	Field( "/sched/cj/ct0/dt" ).set( "1" );
	Field( "/sched/cj/ct1/dt" ).set( "2" );
	Field( "/sched/cj/ct2/dt" ).set( "5" );
	Field cjIn( "/sched/cj/processIn" );
	Field("/sched/processOut").add( cjIn );
	Field( "/sched/cj/ct0/path" ).set ( "/t0/tick" );
	Field( "/sched/cj/ct1/path" ).set ( "/t1/tick" );
	Field( "/sched/cj/ct2/path" ).set ( "/t2/tick" );
	
	// Note that we create these after the path. The resched really
	// does have to do some work
	Element* t0 = Cinfo::find("Tock")->create( "t0", Element::root() );
	Element* t1 = Cinfo::find("Tock")->create( "t1", Element::root() );
	Element* t2 = Cinfo::find("Tock")->create( "t2", Element::root() );

//	Field( "/sched/cj/reschedIn" ).set( "" );
	Field( "/sli_shell/response" ).set( "" );
	Field( "/sli_shell/resetIn" ).set( "" );
	Field( "/sli_shell/stepIn" ).set( "10, -t" );
	Field( "/sli_shell/response" ).get( response );
	unsigned long startpos= 0;
	unsigned long len = 0;
	unsigned int n = sizeof( schedResponse ) / sizeof( char* );
	for (unsigned int i = 0; i < n; i++ ) {
		len = strlen( schedResponse[ i ] );
		string ss = response.substr( startpos, len );
		if ( startpos + len < response.length() &&
			ss == schedResponse[ i ] ){
			cout << ".";
		} else {
			cout << "!\nFailed Sched test[" << i << "]: '" << ss <<
				"' != '" << schedResponse[ i ] << "'\n";
			return;
		}
		startpos += len + 2;
	}
	cout << "Scheduling tests complete\n";
	delete ct1;
	delete ct2;
	delete t0;
	delete t1;
	delete t2;
}

#endif
