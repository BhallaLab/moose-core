/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "../shell/Shell.h"
#include "SimManager.h"

/*
static SrcFinfo1< Id >* plugin()
{
	static SrcFinfo1< Id > ret(
		"plugin", 
		"Sends out Stoich Id so that plugins can directly access fields and functions"
	);
	return &ret;
}
*/

const Cinfo* SimManager::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< SimManager, double > syncTime(
			"syncTime",
			"SyncTime is the interval between synchornozing solvers"
			"5 msec is a typical value",
			&SimManager::setSyncTime,
			&SimManager::getSyncTime
		);

		static ValueFinfo< SimManager, bool > autoPlot(
			"autoPlot",
			"When the autoPlot flag is true, the simManager guesses which"
			"plots are of interest, and builds them.",
			&SimManager::setAutoPlot,
			&SimManager::getAutoPlot
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo build( "build",
			"Sets up model, with the specified method. The method may be"
			"empty if the intention is that methods be set up through "
			"hints in the ChemMesh compartments.",
			new EpFunc1< SimManager, string >( &SimManager::build ) );
		//////////////////////////////////////////////////////////////

	static Finfo* simManagerFinfos[] = {
		&syncTime,		// Value
		&autoPlot,		// Value
		&build,			// DestFinfo
	};

	static Cinfo simManagerCinfo (
		"SimManager",
		Neutral::initCinfo(),
		simManagerFinfos,
		sizeof( simManagerFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SimManager >()
	);

	return &simManagerCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* simManagerCinfo = SimManager::initCinfo();

SimManager::SimManager()
	: 
		syncTime_( 0.005 ),
		autoPlot_( 1 )
{;}

SimManager::~SimManager()
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void SimManager::setAutoPlot( bool v )
{
	autoPlot_ = v;
}

bool SimManager::getAutoPlot() const
{
	return autoPlot_;
}

void SimManager::setSyncTime( double v )
{
	syncTime_ = v;
}

double SimManager::getSyncTime() const
{
	return syncTime_;
}
//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void SimManager::build( const Eref& e, const Qinfo* q, string method )
{
	TreeType tt = findTreeType( e );
	baseId_ = e.id();

	switch ( tt ) {
		case CHEM_ONLY :
			buildFromBareKineticTree( method );
			break;
		case KKIT :
			buildFromKkitTree( method );
			break;
		case CHEM_SPACE :
			break;
		case CHEM_SPACE_MULTISOLVER :
			break;
		case SIGNEUR :
			break;
		default:
			break;
	}
}

//////////////////////////////////////////////////////////////
// Utility functions
//////////////////////////////////////////////////////////////
SimManager::TreeType SimManager::findTreeType( const Eref& e )
{
	return KKIT; // dummy for now.
}

void SimManager::buildFromBareKineticTree( const string& method )
{
	;
}

void SimManager::buildFromKkitTree( const string& method )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	autoPlot_ = 0;
	vector< int > dims( 1, 1 );
	string basePath = baseId_.path();
	Id stoich = shell->doCreate( "Stoich", baseId_, "stoich", dims );
	Field< string >::set( stoich, "path", basePath + "/##" );
	if ( method == "Gillespie" || method == "gillespie" || 
		method == "GSSA" || method == "gssa" ) {
	} else {
		Id gsl = shell->doCreate( "GslIntegrator", stoich, "gsl", dims );
		bool ret = SetGet1< Id >::set( gsl, "stoich", stoich );
		assert( ret );
		ret = Field< bool >::get( gsl, "isInitialized" );
		assert( ret );
		ret = Field< string >::set( gsl, "method", method );
		assert( ret );
	}

	shell->doSetClock( 0, plotdt_ );
	shell->doSetClock( 1, plotdt_ );
	shell->doSetClock( 2, plotdt_ );
	shell->doSetClock( 3, 0 );

	string plotpath = basePath + "/graphs/##[TYPE=Table]," + 
		basePath + "/moregraphs/##[TYPE=Table]";
	shell->doUseClock( basePath + "/gsl", "process", 0);
	shell->doUseClock( plotpath, "process", 2 );
	shell->doReinit();
}

void SimManager::makeStandardElements( const Eref& e, const Qinfo* q, 
	string meshClass )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< int > dims( 1, 1 );
	Id baseId_ = e.id();
	Id kinetics = 
		shell->doCreate( meshClass, baseId_, "kinetics", dims, true );
		SetGet2< double, unsigned int >::set( kinetics, "defaultMesh", 1e-15, 1 );
	assert( kinetics != Id() );
	assert( kinetics.eref().element()->getName() == "kinetics" );

	Id graphs = Neutral::child( baseId_.eref(), "graphs" );
	if ( graphs == Id() ) {
		graphs = 
		shell->doCreate( "Neutral", baseId_, "graphs", dims, true );
	}
	assert( graphs != Id() );

	Id geometry = Neutral::child( baseId_.eref(), "geometry" );
	if ( geometry == Id() ) {

		geometry = 
		shell->doCreate( "Geometry", baseId_, "geometry", dims, true );
		// MsgId ret = shell->doAddMsg( "Single", geometry, "compt", kinetics, "reac" );
		// assert( ret != Msg::bad );
	}
	assert( geometry != Id() );

	Id groups = Neutral::child( baseId_.eref(), "groups" );
	if ( groups == Id() ) {
		groups = 
		shell->doCreate( "Neutral", baseId_, "groups", dims, true );
	}
	assert( groups != Id() );
}
