/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "PoolBase.h"

#define EPSILON 1e-15

const SpeciesId DefaultSpeciesId = 0;

static SrcFinfo1< double >* requestVolume() {
	static SrcFinfo1< double > requestVolume( 
		"requestVolume", 
		"Requests Volume of pool from matching mesh entry"
	);
	return &requestVolume;
}

const Cinfo* PoolBase::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< PoolBase, double > n(
			"n",
			"Number of molecules in pool",
			&PoolBase::setN,
			&PoolBase::getN
		);

		static ElementValueFinfo< PoolBase, double > nInit(
			"nInit",
			"Initial value of number of molecules in pool",
			&PoolBase::setNinit,
			&PoolBase::getNinit
		);

		static ElementValueFinfo< PoolBase, double > diffConst(
			"diffConst",
			"Diffusion constant of molecule",
			&PoolBase::setDiffConst,
			&PoolBase::getDiffConst
		);

		static ElementValueFinfo< PoolBase, double > conc(
			"conc",
			"Concentration of molecules in this pool",
			&PoolBase::setConc,
			&PoolBase::getConc
		);

		static ElementValueFinfo< PoolBase, double > concInit(
			"concInit",
			"Initial value of molecular concentration in pool",
			&PoolBase::setConcInit,
			&PoolBase::getConcInit
		);

		static ElementValueFinfo< PoolBase, double > volume(
			"volume",
			"Volume of compartment. Units are SI. "
			"Utility field, the actual volume info is "
			"stored on a volume mesh entry in the parent compartment."
			"This is hooked up by a message. If the message isn't"
			"available volume is just taken as 1",
			&PoolBase::setVolume,
			&PoolBase::getVolume
		);

		static ElementValueFinfo< PoolBase, unsigned int > speciesId(
			"speciesId",
			"Species identifier for this mol pool. Eventually link to ontology.",
			&PoolBase::setSpecies,
			&PoolBase::getSpecies
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< PoolBase >( &PoolBase::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< PoolBase >( &PoolBase::reinit ) );

		static DestFinfo reacDest( "reacDest",
			"Handles reaction input",
			new OpFunc2< PoolBase, double, double >( &PoolBase::reac )
		);

		static DestFinfo handleMolWt( "handleMolWt",
			"Separate finfo to assign molWt, and consequently diffusion const."
			"Should only be used in SharedMsg with species.",
			new EpFunc1< PoolBase, double >( &PoolBase::handleMolWt )
		);

		static DestFinfo remesh( "remesh",
			"Handle commands to remesh the pool. This may involve changing "
			"the number of pool entries, as well as changing their volumes",
			new EpFunc5< PoolBase, double, unsigned int, unsigned int, vector< unsigned int >, vector< double > >( &PoolBase::remesh )
		);

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////

		static SrcFinfo1< double > nOut( 
				"nOut", 
				"Sends out # of molecules in pool on each timestep"
		);

		static SrcFinfo0 requestMolWt( 
				"requestMolWt", 
				"Requests Species object for mol wt"
		);

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* reacShared[] = {
			&reacDest, &nOut
		};
		static SharedFinfo reac( "reac",
			"Connects to reaction",
			reacShared, sizeof( reacShared ) / sizeof( const Finfo* )
		);
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

		static Finfo* speciesShared[] = {
			&requestMolWt, &handleMolWt
		};

		static SharedFinfo species( "species",
			"Shared message for connecting to species objects",
			speciesShared, sizeof( speciesShared ) / sizeof ( const Finfo* )
		);

		static Finfo* meshShared[] = {
			&remesh, requestVolume()
		};

		static SharedFinfo mesh( "mesh",
			"Shared message for dealing with mesh operations",
			meshShared, sizeof( meshShared ) / sizeof ( const Finfo* )
		);

	static Finfo* poolFinfos[] = {
		&n,			// Value
		&nInit,		// Value
		&diffConst,	// Value
		&conc,		// Value
		&concInit,	// Value
		&volume,	// Readonly Value
		&speciesId,	// Value
		&reac,				// SharedFinfo
		&proc,				// SharedFinfo
		&species,			// SharedFinfo
		&mesh,				// SharedFinfo
	};

	static ZeroSizeDinfo< int > dinfo;
	static Cinfo poolCinfo (
		"PoolBase",
		Neutral::initCinfo(),
		poolFinfos,
		sizeof( poolFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &poolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* poolCinfo = PoolBase::initCinfo();

//////////////////////////////////////////////////////////////
PoolBase::PoolBase()
{;}
PoolBase::~PoolBase()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void PoolBase::process( const Eref& e, ProcPtr p )
{
	vProcess( e, p );
}

void PoolBase::reinit( const Eref& e, ProcPtr p )
{
	vReinit( e, p );
}

void PoolBase::reac( double A, double B )
{
	vReac( A, B );
}

void PoolBase::handleMolWt( const Eref& e, double v )
{
	vHandleMolWt( e, v );
}

void PoolBase::remesh( const Eref& e, 
	double oldvol,
	unsigned int numTotalEntries, unsigned int startEntry, 
	vector< unsigned int > localIndices, vector< double > vols )
{
	// cout << "PoolBase::remesh for " << e.element()->getName() << endl;
	vRemesh( e, oldvol, numTotalEntries, startEntry, localIndices, vols);
}

// Dummy function, not sure if needed any more.
void PoolBase::vRemesh( const Eref& e, 
			double oldVol,
			unsigned int numTotalEntries, unsigned int startEntry, 
			const vector< unsigned int >& localIndices, 
			const vector< double >& vols )
{;}

//////////////////////////////////////////////////////////////
// virtual MsgDest Definitions
//////////////////////////////////////////////////////////////

void PoolBase::vProcess( const Eref& e, ProcPtr p )
{;}

void PoolBase::vReinit( const Eref& e, ProcPtr p )
{;}

void PoolBase::vReac( double A, double B )
{;}

void PoolBase::vHandleMolWt( const Eref& e, double v )
{;}

// void PoolBase::vRemesh(...) is a pure virtual

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void PoolBase::setN( const Eref& e, double v )
{
	vSetN( e, v );
}

double PoolBase::getN( const Eref& e ) const
{
	return vGetN( e );
}

void PoolBase::setNinit( const Eref& e, double v )
{
	vSetNinit( e, v );
}

double PoolBase::getNinit( const Eref& e ) const
{
	return vGetNinit( e );
}

// Conc is given in millimolar. Volume is in m^3
void PoolBase::setConc( const Eref& e, double c ) 
{
	vSetConc( e, c );
}

// Returns conc in millimolar.
double PoolBase::getConc( const Eref& e ) const
{
	return vGetConc( e );
}

void PoolBase::setConcInit( const Eref& e, double c )
{
	vSetConcInit( e, c );
}

double PoolBase::getConcInit( const Eref& e ) const
{
	return vGetConcInit( e );
}

void PoolBase::setDiffConst( const Eref& e, double v )
{
	vSetDiffConst( e, v );
}

double PoolBase::getDiffConst(const Eref& e ) const
{
	return vGetDiffConst( e );
}

void PoolBase::setVolume( const Eref& e, double v )
{
	vSetVolume( e, v );
}

double PoolBase::getVolume( const Eref& e ) const
{
	return vGetVolume( e );
}

void PoolBase::setSpecies( const Eref& e, unsigned int v )
{
	vSetSpecies( e, v );
}

unsigned int PoolBase::getSpecies( const Eref& e ) const
{
	return vGetSpecies( e );
}

//////////////////////////////////////////////////////////////
// Zombie conversion routine: Converts Pool subclasses. There
// will typically be a target specific follow-up function, for example,
// to assign a pointer to the stoichiometry class.
// There should also be a subsequent call to resched for the entire tree.
//////////////////////////////////////////////////////////////
// static func
void PoolBase::zombify( Element* orig, const Cinfo* zClass, Id solver )
{
		/*
	if ( orig->cinfo() == zClass )
		return;
	DataHandler* origHandler = orig->dataHandler();
	DataHandler* dh = origHandler->copyUsingNewDinfo( zClass->dinfo() );
	Element temp( orig->id(), zClass, dh );
	Eref zombier( &temp, 0 );

	PoolBase* z = reinterpret_cast< PoolBase* >( zombier.data() );
	Eref oer( orig, 0 );

	z->setSolver( solver ); // call virtual func to assign solver info.

	PoolBase* m = reinterpret_cast< PoolBase* >( oer.data() );
	// May need to extend to entire array.
	z->vSetSpecies( zombier, 0, m->vGetSpecies( oer, 0 ) );
//	z->vSetConcInit( zombier, 0, m->vGetConcInit( oer, 0 ) );
	z->vSetN( zombier, 0, m->vGetN( oer, 0 ) );
	z->vSetNinit( zombier, 0, m->vGetNinit( oer, 0 ) );
	z->vSetDiffConst( zombier, 0, m->vGetDiffConst( oer, 0 ) );
	orig->zombieSwap( zClass, dh );
	delete origHandler;
	*/
}

// Virtual func: default does nothing.
void PoolBase::setSolver( Id solver )
{
	;
}
