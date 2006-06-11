/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <fstream>
#include "header.h"
#include "../builtins/String.h"
#include "../builtins/Int.h"
#include "TestField.h"
#include "TestMsg.h"
// #include "../genesis_parser/GenesisParser.h"
// #include "../genesis_parser/GenesisParserWrapper.h"

int main(int argc, const char** argv)
{
	Cinfo::initialize();

	Element* shell = Cinfo::find("Shell")->
		create( "sli_shell", Element::root() );
	Element* sli = Cinfo::find("GenesisParser")->
		create( "sli", shell );

	sli->field( "shell" ).set( "/sli_shell" );
	shell->field( "parser" ).set( "/sli_shell/sli" );
	Field f = sli->field( "process" ) ;

	Element* sched = Cinfo::find("Sched")->
		create( "sched", Element::root() );
	Element* cj = Cinfo::find("ClockJob")->create( "cj", sched );
	Element* ct0 = Cinfo::find("ClockTick")->create( "ct0", cj );
	Element* ct1 = Cinfo::find("ClockTick")->create( "ct1", cj );
	Element* ct2 = Cinfo::find("ClockTick")->create( "ct2", cj );
	Field cjIn( "/sched/cj/processIn" );
	Field("/sched/processOut").add( cjIn );
	Field cjClock( "/sched/cj/clock" );
	Field ctClock( "/sched/cj/ct0/clock" );
	ctClock.setElement( ct0 );
	cjClock.add( ctClock );
	ctClock.setElement( ct1 );
	cjClock.add( ctClock );
	ctClock.setElement( ct2 );
	cjClock.add( ctClock );
	Field( "/sched/cj/ct0/dt" ).set( "1" );
	Field( "/sched/cj/ct1/dt" ).set( "2" );
	Field( "/sched/cj/ct2/dt" ).set( "5" );
	
	Element* t0 = Cinfo::find("Tock")->create( "t0", Element::root() );
	Element* t1 = Cinfo::find("Tock")->create( "t1", Element::root() );
	Element* t2 = Cinfo::find("Tock")->create( "t2", Element::root() );
	Field tockOut( "/sched/cj/ct0/tick" );
	Field tockIn( "/t0/tick" );
	tockOut.setElement( ct0 );
	tockIn.setElement( t0 );
	tockOut.add( tockIn );
	tockOut.setElement( ct1 );
	tockIn.setElement( t1 );
	tockOut.add( tockIn );
	tockOut.setElement( ct2 );
	tockIn.setElement( t2 );
	tockOut.add( tockIn );

	/*
	Field jf = job->field( "processIn" );
	sched->field( "processOut" ).add( jf );
	*/
	
#ifdef DO_UNIT_TESTS
	testBasecode();
#endif

	// setField( sli->field( "process" ) );
	f.set( "" );

	// setField( f );

}
