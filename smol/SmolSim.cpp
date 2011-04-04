/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "SmolHeader.h"
#include "ElementValueFinfo.h"
#include "Mol.h"
#include "Reac.h"
#include "Enz.h"
#include "MMenz.h"
#include "SmolMol.h"
#include "SmolPool.h"
#include "SmolReac.h"
#include "SmolEnz.h"
#include "SmolMMenz.h"
#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"
#include "../shell/Neutral.h"
#include "../geom/Surface.h"
#include "../geom/Panel.h"

static SrcFinfo1< Id > plugin( 
		"plugin", 
		"Sends out SmolSim Id so that plugins can directly access fields and functions"
	);

const Cinfo* SmolSim::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////

		static ElementValueFinfo< SmolSim, string > path(
			"path",
			"Path of reaction system to take over",
			&SmolSim::setPath,
			&SmolSim::getPath
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< SmolSim >( &SmolSim::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinint call",
			new ProcOpFunc< SmolSim >( &SmolSim::reinit ) );

		//////////////////////////////////////////////////////////////
		// FieldElementFinfo defintion for Ports.
		//////////////////////////////////////////////////////////////
		/*
		static FieldElementFinfo< SmolSim, Port > portFinfo( "port",
			"Sets up field Elements for ports",
			Port::initCinfo(),
			&SmolSim::getPort,
			&SmolSim::setNumPorts,
			&SmolSim::getNumPorts
		);
		*/

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* smolSimFinfos[] = {
		&path,			// Value
		&plugin,		// SrcFinfo
		&proc,			// SharedFinfo
	};

	static Cinfo smolSimCinfo (
		"SmolSim",
		Neutral::initCinfo(),
		smolSimFinfos,
		sizeof( smolSimFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SmolSim >()
	);

	return &smolSimCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* smolSimCinfo = SmolSim::initCinfo();

SmolSim::SmolSim()
	: sim_( 0 )
{
}

SmolSim::~SmolSim()
{
	if ( sim_ )
		smolFreeSim( sim_ );
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void SmolSim::process( const Eref& e, ProcPtr p )
{
	;
}

void SmolSim::reinit( const Eref& e, ProcPtr p )
{
}


//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////


void SmolSim::setPath( const Eref& e, const Qinfo* q, string v )
{
	if ( path_ != "" && path_ != v ) {
		cout << "SmolSim::setPath: need to clear old path.\n";
		return;
	}
	path_ = v;
	vector< Id > elist;
	Shell::wildcard( path_, elist );

	int dim = 3;
	double lowbounds[3];
	double highbounds[3];
	lowbounds[0] = lowbounds[1] = lowbounds[2] = -1;
	highbounds[0] = highbounds[1] = highbounds[2] = 1;
	if ( sim_ )
		smolFreeSim( sim_ );
	sim_ = smolNewSim( dim, lowbounds, highbounds );
	// allocateObjMap( elist );
	// allocateModel( elist );
	smolDebugMode( 1 ); // set to zero to shut it up.
	zombifyModel( e, elist );
	/*
	cout << "Zombified " << numVarMols_ << " Molecules, " <<
		numReac_ << " reactions\n";
		*/
}

string SmolSim::getPath( const Eref& e, const Qinfo* q ) const
{
	return path_;
}

//////////////////////////////////////////////////////////////
// Model zombification functions
//////////////////////////////////////////////////////////////

void SmolSim::zombifyModel( const Eref& e, const vector< Id >& elist )
{
	static const Cinfo* molCinfo = Mol::initCinfo();
//	static const Cinfo* bufMolCinfo = BufMol::initCinfo();
//	static const Cinfo* funcMolCinfo = FuncMol::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();
//	static const Cinfo* chemComptCinfo = ChemCompt::initCinfo();
	// static const Cinfo* sumFuncCinfo = SumFunc::initCinfo();
	// The FuncMol handles zombification of stuff coming in to it.
	static const Cinfo* surfaceCinfo = Surface::initCinfo();
	static const Cinfo* panelCinfo = Panel::initCinfo();

	vector< Id > pools;
	vector< Id > reacs;
	vector< Id > enz;
	vector< Id > mmenz;
	vector< Id > surfaces;
	vector< Id > panels;

	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == molCinfo )
			pools.push_back( *i );
		else if ( ei->cinfo() == reacCinfo )
			reacs.push_back( *i );
		else if ( ei->cinfo() == mmEnzCinfo )
			mmenz.push_back( *i );
		else if ( ei->cinfo() == enzCinfo )
			enz.push_back( *i );
		else if ( ei->cinfo() == surfaceCinfo )
			surfaces.push_back( *i );
		else if ( ei->cinfo() == panelCinfo )
			panels.push_back( *i );
	}

	/// Stage 1: Define the species in the model and set diff const.
	for ( vector< Id >::iterator i = pools.begin(); i != pools.end(); ++i )
		SmolPool::smolSpeciesInit( e.element(), (*i)() );

	/// Stage 2: Set the max # of molecules likely to occur
		SmolPool::smolMaxNumMolecules( sim_, pools );

	/// Stage 3: Add the surfaces and panels.
	for ( vector< Id >::iterator i = surfaces.begin(); i != surfaces.end(); ++i )
		parseSurface( e.element(), (*i)() );
	
	/// Stage 4: Set the surface rate/action. This needs to be done for all
	/// combinations of surface and species. I haven't yet worked out how
	/// to represent this in MOOSE, so for now I'm having it reflect
	/// everything.
	for ( vector< Id >::iterator i = pools.begin(); i != pools.end(); ++i )
		for ( vector< Id >::iterator j = surfaces.begin(); j != surfaces.end(); ++j )
			parseSurfaceAction( e.element(), (*i)(), (*j)() );

	/// Stage 5: Add molecules to the system.
	for ( vector< Id >::iterator i = pools.begin(); i != pools.end(); ++i )
		SmolPool::smolNinit( e.element(), (*i)() );

	/// Stage 6: Add reactions to the system.
	for ( vector< Id >::iterator i = reacs.begin(); i != reacs.end(); ++i )
		SmolReac::zombify( e.element(), (*i)() );


	/// To set it off:	smolSetSimTimes
	/// To run 1 timestep: smolRunTimeStep
	/// To run till end: smolRunSim
	/// To run till specified time: smolRunSimUntil

	/// Wrap up: Zombify the pools
	for ( vector< Id >::iterator i = pools.begin(); i != pools.end(); ++i )
		SmolPool::zombify( e.element(), (*i)() );

	smolDisplaySim( sim_ );

	/// Test stuff here to see if my model was set up properly.
	ErrorCode ret = smolSetSimTimes( sim_ , 0.0, 1.0 , 50e-6 );
	assert( ret == ECok );
	ret = smolRunSim( sim_ );
	assert( ret == ECok );
}

PanelShape panelShapeFromInt( unsigned int i )
{
	switch ( i ) {
		case Moose::PSrect: return PSrect;
		case Moose::PStri: return PStri;
		case Moose::PSsph: return PSsph;
		case Moose::PShemi: return PShemi;
		case Moose::PSdisk: return PSdisk;
		case Moose::PScyl: return PScyl;
		case Moose::PSall: return PSall;
		default:
			return PSnone;
	}
/*

	if ( i == PSrect )
		return PSrect;
	else if ( i == PSrect )
		return PSrect;
		*/
	return PSsph;
	// PSrect,PStri,PSsph,PScyl,PShemi,PSdisk,PSall,PSnone
}

void SmolSim::parseSurface( Element* solver, Element* surface )
{
	// static const Cinfo* panelCinfo = Panel::initCinfo();

	char* surfaceName = new char[ surface->getName().length() + 1];
	strcpy( surfaceName, surface->getName().c_str() );
	ErrorCode ret = smolAddSurface( sim_, surfaceName );
	assert( ret == ECok );
	vector< Id > panels;
	Neutral::children( Eref( surface, 0 ), panels );
	for ( vector< Id >::iterator i = panels.begin(); 
		i != panels.end(); ++i ) {
		if ( (*i)()->cinfo()->isA( "Panel" ) ) {
			Element* panelm = (*i)();
			char* panelName = new char[ panelm->getName().length() + 1];
			strcpy( panelName, panelm->getName().c_str() );
			Panel* p = reinterpret_cast< Panel* >( i->eref().data() );
			PanelShape panelShape = panelShapeFromInt( p->getShapeId() );
			vector< double > coords = p->getCoords();
			cout << "coords size = " << coords.size() << endl;
			ret = smolAddPanel( sim_, surfaceName, panelShape, panelName,
				0, &(coords[0]) );
			assert( ret == ECok );
			delete[] panelName;
		}
	}

	delete[] surfaceName;
}

void SmolSim::parseSurfaceAction( Element* solver, Element* pool, 
	Element* surface )
{
	char* poolName = new char[ pool->getName().length() + 1];
	strcpy( poolName, pool->getName().c_str() );
	char* surfaceName = new char[ surface->getName().length() + 1];
	strcpy( surfaceName, surface->getName().c_str() );


	/// For now the PanelFace is always reflective on both faces and the
	/// MolecState is always in solution.
	ErrorCode ret = smolSetSurfaceAction( 
		sim_, surfaceName, PFboth, poolName, MSsoln, SAreflect );

	assert( ret == ECok );

	delete[] poolName;
	delete[] surfaceName;
	
}

unsigned int SmolSim::convertIdToMolIndex( Id id ) const
{
	/*
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < S_.size() );
	return i;
	*/
	return 0;
}

unsigned int SmolSim::convertIdToReacIndex( Id id ) const
{
	/*
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < rates_.size() );
	return i;
	*/
	return 0;
}

//////////////////////////////////////////////////////////////
// Model running functions
//////////////////////////////////////////////////////////////

