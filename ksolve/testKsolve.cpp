/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "ReadKkit.h"
#include "ElementValueFinfo.h"
#include "GslIntegrator.h"
#include "../shell/Shell.h"

void testKsolveZombify()
{
	ReadKkit rk;
	// rk.read( "test.g", "dend", 0 );
	Id base = rk.read( "foo.g", "dend", Id() );
	assert( base != Id() );
	// Id kinetics = s->doFind( "/kinetics" );

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Id stoich = s->doCreate( "Stoich", base, "stoich", dims );
	assert( stoich != Id() );
	string temp = "/dend/##";
	bool ret = Field<string>::set( stoich.eref(), "path", temp );
	assert( ret );

	/*
	rk.run();
	rk.dumpPlots( "dend.plot" );
	*/

	s->doDelete( base );
	cout << "." << flush;
}

void testGslIntegrator()
{
	ReadKkit rk;
	// rk.read( "test.g", "dend", 0 );
	Id base = rk.read( "foo.g", "dend", Id() );
	assert( base != Id() );
	// Id kinetics = s->doFind( "/kinetics" );

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Id stoich = s->doCreate( "Stoich", base, "stoich", dims );
	assert( stoich != Id() );
	string temp = "/dend/##";
	bool ret = Field<string>::set( stoich.eref(), "path", temp );
	assert( ret );

	Id gsl = s->doCreate( "GslIntegrator", base, "gsl", dims );
// 	MsgId mid = 
	s->doAddMsg( "Single", 
		FullId( stoich, 0 ), "plugin", 
		FullId( gsl, 0 ), "stoich" );

	const Finfo* f = Stoich::initCinfo()->findFinfo( "plugin" );
	assert( f );
	const SrcFinfo1< Stoich* >* plug = 
		dynamic_cast< const SrcFinfo1< Stoich* >* >( f );
	assert( plug );

	Stoich* stoichData = reinterpret_cast< Stoich* >( stoich.eref().data() );
	GslIntegrator* gi = reinterpret_cast< GslIntegrator* >( gsl.eref().data() );
	ProcInfo p;
	p.dt = 1.0;
	p.currTime = 0;

	plug->send( stoich.eref(), &p, stoichData );
	Qinfo::mpiClearQ( &p );
	assert( gi->getIsInitialized() );

	s->doSetClock( 0, rk.getPlotDt() );
	s->doSetClock( 1, rk.getPlotDt() );
	s->doSetClock( 2, rk.getPlotDt() );
	string gslpath = rk.getBasePath() + "/gsl";
	string  plotpath = rk.getBasePath() + "/graphs/##[TYPE=Table]," +
		rk.getBasePath() + "/moregraphs/##[TYPE=Table]";
	s->doUseClock( gslpath, "process", 0 );
	s->doUseClock( plotpath, "process", 2 );
	s->doReinit();
	s->doStart( rk.getMaxTime() );

			/*
	Eref gsle( gsl.eref() );
	for ( double i = 0.0; i < rk.getMaxTime(); i += rk.getPlotDt() ) {
		gi->process( gsle, &p );
		p.currTime = i;
		cout << i << 
			"	" << stoichData->S()[0] <<
			"	" << stoichData->S()[1] <<
			"	" << stoichData->S()[2] <<
			"	" << stoichData->S()[3] <<
			"	" << stoichData->S()[4] <<
			"	" << stoichData->S()[5] <<
			"	" << stoichData->S()[6] <<
			"	" << stoichData->S()[7] <<
			"	" << stoichData->S()[8] <<
			"	" << stoichData->S()[9] <<
			"	" << stoichData->S()[10] <<
			"	" << stoichData->S()[11] <<
			"	" << stoichData->S()[12] <<
			"	" << stoichData->S()[13] <<
			"	" << stoichData->S()[14] <<
			endl;
	}
			*/


	/*
	rk.run();
	*/
	rk.dumpPlots( "gsl.plot" );

	s->doDelete( base );
	cout << "." << flush;
}

void testKsolve()
{
	testKsolveZombify();
	testGslIntegrator();
}

void testMpiKsolve()
{
}
