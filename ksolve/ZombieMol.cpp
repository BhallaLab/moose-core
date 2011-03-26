/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include "ZombieMol.h"
#include "Mol.h"
#include "ElementValueFinfo.h"
#include "DataHandlerWrapper.h"

#define EPSILON 1e-15

static const double NA = 6.023e23;

const Cinfo* ZombieMol::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieMol, double > n(
			"n",
			"Number of molecules",
			&ZombieMol::setN,
			&ZombieMol::getN
		);

		static ElementValueFinfo< ZombieMol, double > nInit(
			"nInit",
			"Initial value of number of molecules",
			&ZombieMol::setNinit,
			&ZombieMol::getNinit
		);

		static ElementValueFinfo< ZombieMol, double > diffConst(
			"diffConst",
			"Diffusion constant of molecule",
			&ZombieMol::setDiffConst,
			&ZombieMol::getDiffConst
		);

		static ElementValueFinfo< ZombieMol, double > conc(
			"conc",
			"Concentration of molecules",
			&ZombieMol::setConc,
			&ZombieMol::getConc
		);

		static ElementValueFinfo< ZombieMol, double > concInit(
			"concInit",
			"Initial value of molecular concentration",
			&ZombieMol::setConcInit,
			&ZombieMol::getConcInit
		);

		static ReadOnlyElementValueFinfo< ZombieMol, double > size(
			"size",
			"Size of compartment. Units are SI. "
			"Utility field, the master size info is "
			"stored on the compartment itself. For voxel-based spatial"
			"models, the 'size' of the molecule at a given index is the"
			"size of that voxel.",
			&ZombieMol::getSize
		);

		static ElementValueFinfo< ZombieMol, unsigned int > species(
			"species",
			"Species identifer for this mol pool",
			&ZombieMol::setSpecies,
			&ZombieMol::getSpecies
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< ZombieMol >( &ZombieMol::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< ZombieMol >( &ZombieMol::reinit ) );

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo reacDest( "reacDest",
			"Handles reaction input",
			new OpFunc2< ZombieMol, double, double >( &ZombieMol::reac )
		);

		static DestFinfo setSize( "setSize",
			"Separate finfo to assign size, should only be used by compartment."
			"Defaults to SI units of volume: m^3",
			new EpFunc1< ZombieMol, double >( &ZombieMol::setSize )
		);

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////
		static SrcFinfo1< double > nOut( 
				"nOut", 
				"Sends out # of molecules on each timestep"
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

	static Finfo* zombieMolFinfos[] = {
		&n,				// Value
		&nInit,			// Value
		&diffConst,		// Value
		&conc,			// Value
		&concInit,		// Value
		&size,			// Value
		&species,		// Value
		&group,			// DestFinfo
		&setSize,			// DestFinfo
		&reac,				// SharedFinfo
		&proc,				// SharedFinfo
	};

	static Cinfo zombieMolCinfo (
		"ZombieMol",
		Neutral::initCinfo(),
		zombieMolFinfos,
		sizeof( zombieMolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< ZombieMol >()
	);

	return &zombieMolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieMolCinfo = ZombieMol::initCinfo();

ZombieMol::ZombieMol()
{;}

ZombieMol::~ZombieMol()
{;}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

// Doesn't do anything on its own.
void ZombieMol::process( const Eref& e, ProcPtr p )
{;}

void ZombieMol::reinit( const Eref& e, ProcPtr p )
{;}

void ZombieMol::reac( double A, double B )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieMol::setN( const Eref& e, const Qinfo* q, double v )
{
	S_[ convertIdToMolIndex( e.id() ) ] = v;
}

double ZombieMol::getN( const Eref& e, const Qinfo* q ) const
{
	return S_[ convertIdToMolIndex( e.id() ) ];
}

void ZombieMol::setNinit( const Eref& e, const Qinfo* q, double v )
{
	Sinit_[ convertIdToMolIndex( e.id() ) ] = v;
}

double ZombieMol::getNinit( const Eref& e, const Qinfo* q ) const
{
	return Sinit_[ convertIdToMolIndex( e.id() ) ];
}

void ZombieMol::setConc( const Eref& e, const Qinfo* q, double conc )
{
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	S_[ mol ] = 1e-3 * NA * conc * compartmentSize_[ index ];
}

double ZombieMol::getConc( const Eref& e, const Qinfo* q ) const
{
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	assert( compartmentSize_[ index ] > 0.0 );
	return 1e3 * S_[ mol ] / ( NA * compartmentSize_[ index ] );
}

void ZombieMol::setConcInit( const Eref& e, const Qinfo* q, double conc )
{
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	Sinit_[ mol ] = 1e-3 * NA * conc * compartmentSize_[ index ];
}

double ZombieMol::getConcInit( const Eref& e, const Qinfo* q ) const
{
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	assert( compartmentSize_[ index ] > 0.0 );
	return 1e3 * Sinit_[ mol ] / ( NA * compartmentSize_[ index ] );
}

void ZombieMol::setDiffConst( const Eref& e, const Qinfo* q, double v )
{
	// diffConst_ = v;
}

double ZombieMol::getDiffConst( const Eref& e, const Qinfo* q ) const
{
	// return diffConst_;
	return 0;
}

void ZombieMol::setSize( const Eref& e, const Qinfo* q, double v )
{
	// Illegal operation.
}

double ZombieMol::getSize( const Eref& e, const Qinfo* q ) const
{
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	return compartmentSize_[ index ];
}

void ZombieMol::setSpecies( const Eref& e, const Qinfo* q, unsigned int v )
{
	species_[ convertIdToMolIndex( e.id() ) ] = v;
}

unsigned int ZombieMol::getSpecies( const Eref& e, const Qinfo* q ) const
{
	return species_[ convertIdToMolIndex( e.id() ) ];
}

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
void ZombieMol::zombify( Element* solver, Element* orig )
{
	Element temp( orig->id(), zombieMolCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieMol* z = reinterpret_cast< ZombieMol* >( zer.data() );
	Mol* m = reinterpret_cast< Mol* >( oer.data() );

	z->setN( zer, 0, m->getN() );
	z->setNinit( zer, 0, m->getNinit() );
	z->setSpecies( zer, 0, m->getSpecies() );
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
	orig->zombieSwap( zombieMolCinfo, dh );
}

// Static func
void ZombieMol::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieMol* z = reinterpret_cast< ZombieMol* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( Mol::initCinfo(), dh );

	Mol* m = reinterpret_cast< Mol* >( oer.data() );

	m->setN( z->getN( zer, 0 ) );
	m->setNinit( z->getNinit( zer, 0 ) );
	m->setSpecies( z->getSpecies( zer, 0 ) );
}
