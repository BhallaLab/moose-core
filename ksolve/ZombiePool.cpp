/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include "ZombiePool.h"
#include "Pool.h"
#include "lookupSizeFromMesh.h"
#include "ElementValueFinfo.h"
#include "DataHandlerWrapper.h"

#define EPSILON 1e-15

static SrcFinfo1< double >* requestSize() {
	static SrcFinfo1< double > requestSize(
		"requestSize",
		"Requests Size of pool from matching mesh entry"
	);
	return &requestSize;
}


const Cinfo* ZombiePool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombiePool, double > n(
			"n",
			"Number of molecules in pool",
			&ZombiePool::setN,
			&ZombiePool::getN
		);

		static ElementValueFinfo< ZombiePool, double > nInit(
			"nInit",
			"Initial value of number of molecules in pool",
			&ZombiePool::setNinit,
			&ZombiePool::getNinit
		);

		static ElementValueFinfo< ZombiePool, double > diffConst(
			"diffConst",
			"Diffusion constant of molecule",
			&ZombiePool::setDiffConst,
			&ZombiePool::getDiffConst
		);

		static ElementValueFinfo< ZombiePool, double > conc(
			"conc",
			"Concentration of molecules in pool",
			&ZombiePool::setConc,
			&ZombiePool::getConc
		);

		static ElementValueFinfo< ZombiePool, double > concInit(
			"concInit",
			"Initial value of molecular concentration in pool",
			&ZombiePool::setConcInit,
			&ZombiePool::getConcInit
		);

		static ReadOnlyElementValueFinfo< ZombiePool, double > size(
			"size",
			"Size of compartment. Units are SI. "
			"Utility field, the master size info is "
			"stored on the compartment itself. For voxel-based spatial"
			"models, the 'size' of the pool at a given index is the"
			"size of that voxel.",
			&ZombiePool::getSize
		);

		static ElementValueFinfo< ZombiePool, unsigned int > species(
			"species",
			"Species identifer for this mol pool",
			&ZombiePool::setSpecies,
			&ZombiePool::getSpecies
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< ZombiePool >( &ZombiePool::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< ZombiePool >( &ZombiePool::reinit ) );

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo reacDest( "reacDest",
			"Handles reaction input",
			new OpFunc2< ZombiePool, double, double >( &ZombiePool::reac )
		);

		static DestFinfo setSize( "setSize",
			"Separate finfo to assign size, should only be used by compartment."
			"Defaults to SI units of volume: m^3",
			new EpFunc1< ZombiePool, double >( &ZombiePool::setSize )
		);

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////
		static SrcFinfo1< double > nOut( 
				"nOut", 
				"Sends out # of molecules in pool on each timestep"
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

	static Finfo* zombiePoolFinfos[] = {
		&n,				// Value
		&nInit,			// Value
		&diffConst,		// Value
		&conc,			// Value
		&concInit,		// Value
		&size,			// Value
		&species,		// Value
		&group,			// DestFinfo
		requestSize(),	// SrcFinfo
		&reac,				// SharedFinfo
		&proc,				// SharedFinfo
	};

	static Cinfo zombiePoolCinfo (
		"ZombiePool",
		Neutral::initCinfo(),
		zombiePoolFinfos,
		sizeof( zombiePoolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< ZombiePool >()
	);

	return &zombiePoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombiePoolCinfo = ZombiePool::initCinfo();

ZombiePool::ZombiePool()
{;}

ZombiePool::~ZombiePool()
{;}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

// Doesn't do anything on its own.
void ZombiePool::process( const Eref& e, ProcPtr p )
{;}

void ZombiePool::reinit( const Eref& e, ProcPtr p )
{;}

void ZombiePool::reac( double A, double B )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombiePool::setN( const Eref& e, const Qinfo* q, double v )
{
	S_[ convertIdToPoolIndex( e.id() ) ] = v;
}

double ZombiePool::getN( const Eref& e, const Qinfo* q ) const
{
	return S_[ convertIdToPoolIndex( e.id() ) ];
}

void ZombiePool::setNinit( const Eref& e, const Qinfo* q, double v )
{
	Sinit_[ convertIdToPoolIndex( e.id() ) ] = v;
}

double ZombiePool::getNinit( const Eref& e, const Qinfo* q ) const
{
	return Sinit_[ convertIdToPoolIndex( e.id() ) ];
}

void ZombiePool::setConc( const Eref& e, const Qinfo* q, double conc )
{
	unsigned int pool = convertIdToPoolIndex( e.id() );
	S_[ pool ] = 1e-3 * NA * conc * lookupSizeFromMesh( e, requestSize() );

	/*
	unsigned int index = compartment_[ pool ];
	assert( index < compartmentSize_.size() );
	S_[ pool ] = 1e-3 * NA * conc * compartmentSize_[ index ];
	*/
}

double ZombiePool::getConc( const Eref& e, const Qinfo* q ) const
{
	/*
	const ProcInfo* p = q->getProcInfo();
	cout << p->groupId << ":" << p->threadIndexInGroup << " ZombiePool getConc " << e.element()->getName() << endl << flush;
	*/
	unsigned int pool = convertIdToPoolIndex( e.id() );
	return 1e3 * S_[ pool ] / ( NA * lookupSizeFromMesh( e, requestSize()));


	/*
	unsigned int index = compartment_[ pool ];
	assert( index < compartmentSize_.size() );
	assert( compartmentSize_[ index ] > 0.0 );
	return 1e3 * S_[ pool ] / ( NA * compartmentSize_[ index ] );
	*/
}

void ZombiePool::setConcInit( const Eref& e, const Qinfo* q, double conc )
{
	unsigned int pool = convertIdToPoolIndex( e.id() );
	unsigned int index = compartment_[ pool ];
	assert( index < compartmentSize_.size() );
	Sinit_[ pool ] = 1e-3 * NA * conc * compartmentSize_[ index ];
}

double ZombiePool::getConcInit( const Eref& e, const Qinfo* q ) const
{
	unsigned int pool = convertIdToPoolIndex( e.id() );
	unsigned int index = compartment_[ pool ];
	assert( index < compartmentSize_.size() );
	assert( compartmentSize_[ index ] > 0.0 );
	return 1e3 * Sinit_[ pool ] / ( NA * compartmentSize_[ index ] );
}

void ZombiePool::setDiffConst( const Eref& e, const Qinfo* q, double v )
{
	// diffConst_ = v;
}

double ZombiePool::getDiffConst( const Eref& e, const Qinfo* q ) const
{
	// return diffConst_;
	return 0;
}

void ZombiePool::setSize( const Eref& e, const Qinfo* q, double v )
{
	// Illegal operation.
}

double ZombiePool::getSize( const Eref& e, const Qinfo* q ) const
{
	unsigned int pool = convertIdToPoolIndex( e.id() );
	unsigned int index = compartment_[ pool ];
	assert( index < compartmentSize_.size() );
	return compartmentSize_[ index ];
}

void ZombiePool::setSpecies( const Eref& e, const Qinfo* q, unsigned int v )
{
	species_[ convertIdToPoolIndex( e.id() ) ] = v;
}

unsigned int ZombiePool::getSpecies( const Eref& e, const Qinfo* q ) const
{
	return species_[ convertIdToPoolIndex( e.id() ) ];
}

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
void ZombiePool::zombify( Element* solver, Element* orig )
{
	Element temp( orig->id(), zombiePoolCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombiePool* z = reinterpret_cast< ZombiePool* >( zer.data() );
	Pool* m = reinterpret_cast< Pool* >( oer.data() );

	z->setN( zer, 0, m->getN() );
	z->setNinit( zer, 0, m->getNinit() );
	z->setSpecies( zer, 0, m->getSpecies() );
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
	orig->zombieSwap( zombiePoolCinfo, dh );
}

// Static func
void ZombiePool::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombiePool* z = reinterpret_cast< ZombiePool* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( Pool::initCinfo(), dh );

	Pool* m = reinterpret_cast< Pool* >( oer.data() );

	m->setN( z->getN( zer, 0 ) );
	m->setNinit( z->getNinit( zer, 0 ) );
	m->setSpecies( z->getSpecies( zer, 0 ) );
}
