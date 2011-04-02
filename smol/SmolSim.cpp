/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <smoldyn/libsmoldyn.h>
#include "header.h"
#include "SmolSim.h"
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
{;}

SmolSim::~SmolSim()
{
	;
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
		// unzombify( path_ );
		cout << "SmolSim::setPath: need to clear old path.\n";
		return;
	}
	path_ = v;
	/*
	vector< Id > elist;
	Shell::wildcard( path_, elist );

	allocateObjMap( elist );
	allocateModel( elist );
	zombifyModel( e, elist );
	y_.assign( Sinit_.begin(), Sinit_.begin() + numVarMols_ );

	cout << "Zombified " << numVarMols_ << " Molecules, " <<
		numReac_ << " reactions\n";
	N_.print();
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

	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == molCinfo ) {
			SmolPool::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == reacCinfo ) {
			SmolReac::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == mmEnzCinfo ) {
			SmolMMenz::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == enzCinfo ) {
			SmolEnz::zombify( e.element(), (*i)() );
		}
	}
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

