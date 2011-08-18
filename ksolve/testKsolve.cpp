/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <sys/time.h>
#include "StoichHeaders.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "ElementValueFinfo.h"
#include "GslIntegrator.h"
#include "../shell/Shell.h"
#include "ReadKkit.h"

static const double TOLERANCE = 1e-6;

// This is a regression test
void testKsolveZombify( string modelFile )
{
	ReadKkit rk;
	Id base = rk.read( modelFile, "dend", Id() );
	assert( base != Id() );
	// Id kinetics = s->doFind( "/kinetics" );

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Id stoich = s->doCreate( "Stoich", base, "stoich", dims );
	assert( stoich != Id() );
	string temp = "/dend/##";
	bool ret = Field<string>::set( stoich, "path", temp );
	assert( ret );

	/*
	rk.run();
	rk.dumpPlots( "dend.plot" );
	*/

	s->doDelete( base );
	cout << "." << flush;
}

/**
 * Benchmarks assorted models: both time and values. Returns 
 * time it takes to run the model.
 * modelName is the base name of the model
 * plotName is of the form "conc1/foo.Co"
 * simTime is the time 
 */
 double testGslIntegrator( string modelName, string plotName,
 	double plotDt, double simTime )
{
	ReadKkit rk;
	Id base = rk.read( modelName + ".g" , "model", Id() );
	assert( base != Id() );
	// Id kinetics = s->doFind( "/kinetics" );

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	/*
	Id stoich = s->doCreate( "Stoich", base, "stoich", dims );
	assert( stoich != Id() );
	string temp = "/model/##";
	bool ret = Field<string>::set( stoich, "path", temp );
	assert( ret );
	*/
	Id stoich = base;

	Id gsl = s->doCreate( "GslIntegrator", base, "gsl", dims );
	/*
	MsgId mid = s->doAddMsg( "Single", 
		ObjId( stoich, 0 ), "plugin", 
		ObjId( gsl, 0 ), "stoich" );
	assert( mid != Msg::badMsg );

	const Finfo* f = Stoich::initCinfo()->findFinfo( "plugin" );
	assert( f );
	const SrcFinfo1< Stoich* >* plug = 
		dynamic_cast< const SrcFinfo1< Stoich* >* >( f );
	assert( plug );
	Stoich* stoichData = reinterpret_cast< Stoich* >( stoich.eref().data() );
	GslIntegrator* gi = reinterpret_cast< GslIntegrator* >( gsl.eref().data() );
	*/

	ProcInfo p;
	p.dt = 1.0;
	p.currTime = 0;

	// plug->send( stoich.eref(), &p, stoichData );
	bool ret = SetGet1< Id >::set( gsl, "stoich", stoich );
	assert( ret );
	ret = Field< bool >::get( gsl, "isInitialized" );
	assert( ret );

	s->doSetClock( 0, plotDt );
	s->doSetClock( 1, plotDt );
	s->doSetClock( 2, plotDt );
	string gslpath = rk.getBasePath() + "/gsl";
	string  plotpath = rk.getBasePath() + "/graphs/##[TYPE=Table]," +
		rk.getBasePath() + "/moregraphs/##[TYPE=Table]";
	s->doUseClock( gslpath, "process", 0 );
	s->doUseClock( plotpath, "process", 2 );
	struct timeval tv0;
	struct timeval tv1;
	gettimeofday( &tv0, 0 );
	s->doReinit();
	s->doStart( simTime );
	gettimeofday( &tv1, 0 );

	if ( plotName.length() > 0 ) {
		Id plotId( string( "/model/graphs/" ) + plotName );
		assert( plotId != Id() );
		bool ok = SetGet::strSet( plotId, "compareXplot",
			modelName + ".plot,/graphs/" + plotName + ",rmsr" );
		assert( ok );
		double rmsr = Field< double >::get( plotId, "outputValue" );
		assert( rmsr < TOLERANCE );
	}

	s->doDelete( base );
	cout << "." << flush;
	double sret = tv1.tv_sec - tv0.tv_sec;
	double uret = tv1.tv_usec;
	uret -= tv0.tv_usec;
	return sret + 1e-6 * uret;
}

void testGsolver(string modelName, string plotName, double plotDt, double simTime )
{
	ReadKkit rk;
	Id base = rk.read( modelName + ".g" , "model", Id(), "GssaStoich" );
	assert( base != Id() );
	// Id kinetics = s->doFind( "/kinetics" );

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Id stoich = base;

	s->doSetClock( 0, plotDt );
	s->doSetClock( 1, plotDt );
	s->doSetClock( 2, plotDt );
	s->doSetClock( 3, plotDt );

	string  plotpath = rk.getBasePath() + "/graphs/##[TYPE=Table]," +
		rk.getBasePath() + "/moregraphs/##[TYPE=Table]";
	s->doUseClock( base.path(), "process", 0 );
	s->doUseClock( plotpath, "process", 2 );
	s->doReinit();
	s->doStart( simTime );

	string plotfile = modelName + ".out";
	if ( plotName.length() > 0 ) {
		Id plotId( string( "/model/graphs/conc1/" ) + plotName );
		assert( plotId != Id() );
		SetGet2< string, string>::set( plotId, "xplot", plotfile, plotName);
		/*
		bool ok = SetGet::strSet( plotId, "compareXplot",
			modelName + ".plot,/graphs/" + plotName + ",rmsr" );
		assert( ok );
		double rmsr = Field< double >::get( plotId, "outputValue" );
		assert( rmsr < TOLERANCE );
		*/
	}
	s->doDelete( base );
	cout << "." << flush;
}
