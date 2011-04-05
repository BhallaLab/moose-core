/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "SmolHeader.h"
#include "SmolPool.h"
#include "Mol.h"
#include "ElementValueFinfo.h"
#include "DataHandlerWrapper.h"

const Cinfo* SmolPool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< SmolPool, double > n(
			"n",
			"Number of molecules",
			&SmolPool::setN,
			&SmolPool::getN
		);

		static ElementValueFinfo< SmolPool, double > nInit(
			"nInit",
			"Initial value of number of molecules",
			&SmolPool::setNinit,
			&SmolPool::getNinit
		);

		static ElementValueFinfo< SmolPool, double > diffConst(
			"diffConst",
			"Diffusion constant of molecule",
			&SmolPool::setDiffConst,
			&SmolPool::getDiffConst
		);

		static ElementValueFinfo< SmolPool, double > conc(
			"conc",
			"Concentration of molecules",
			&SmolPool::setConc,
			&SmolPool::getConc
		);

		static ElementValueFinfo< SmolPool, double > concInit(
			"concInit",
			"Initial value of molecular concentration",
			&SmolPool::setConcInit,
			&SmolPool::getConcInit
		);

		static ReadOnlyElementValueFinfo< SmolPool, double > size(
			"size",
			"Size of compartment. Units are SI. "
			"Utility field, the master size info is "
			"stored on the compartment itself. For voxel-based spatial"
			"models, the 'size' of the molecule at a given index is the"
			"size of that voxel.",
			&SmolPool::getSize
		);

		static ElementValueFinfo< SmolPool, unsigned int > species(
			"species",
			"Species identifer for this mol pool",
			&SmolPool::setSpecies,
			&SmolPool::getSpecies
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< SmolPool >( &SmolPool::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< SmolPool >( &SmolPool::reinit ) );

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo reacDest( "reacDest",
			"Handles reaction input",
			new OpFunc2< SmolPool, double, double >( &SmolPool::reac )
		);

		static DestFinfo setSize( "setSize",
			"Separate finfo to assign size, should only be used by compartment."
			"Defaults to SI units of volume: m^3",
			new EpFunc1< SmolPool, double >( &SmolPool::setSize )
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

	static Finfo* smolPoolFinfos[] = {
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

	static Cinfo smolPoolCinfo (
		"SmolPool",
		Neutral::initCinfo(),
		smolPoolFinfos,
		sizeof( smolPoolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SmolPool >()
	);

	return &smolPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* smolPoolCinfo = SmolPool::initCinfo();

SmolPool::SmolPool()
{;}

SmolPool::~SmolPool()
{;}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

// Doesn't do anything on its own.
void SmolPool::process( const Eref& e, ProcPtr p )
{;}

void SmolPool::reinit( const Eref& e, ProcPtr p )
{;}

void SmolPool::reac( double A, double B )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void SmolPool::setN( const Eref& e, const Qinfo* q, double v )
{
	// S_[ convertIdToMolIndex( e.id() ) ] = v;
}

double SmolPool::getN( const Eref& e, const Qinfo* q ) const
{
	return smolGetMoleculeCount( sim_, e.element()->getName().c_str(), 
		MSsoln );
}

void SmolPool::setNinit( const Eref& e, const Qinfo* q, double v )
{
	nInit_ = v;
}

double SmolPool::getNinit( const Eref& e, const Qinfo* q ) const
{
	// return Sinit_[ convertIdToMolIndex( e.id() ) ];
	return nInit_;
}

void SmolPool::setConc( const Eref& e, const Qinfo* q, double conc )
{
	/*
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	S_[ mol ] = 1e-3 * NA * conc * compartmentSize_[ index ];
	*/
}

double SmolPool::getConc( const Eref& e, const Qinfo* q ) const
{
	/*
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	assert( compartmentSize_[ index ] > 0.0 );
	return 1e3 * S_[ mol ] / ( NA * compartmentSize_[ index ] );
	*/
	return 0;
}

void SmolPool::setConcInit( const Eref& e, const Qinfo* q, double conc )
{
	/*
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	Sinit_[ mol ] = 1e-3 * NA * conc * compartmentSize_[ index ];
	*/
}

double SmolPool::getConcInit( const Eref& e, const Qinfo* q ) const
{
	/*
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	assert( compartmentSize_[ index ] > 0.0 );
	return 1e3 * Sinit_[ mol ] / ( NA * compartmentSize_[ index ] );
	*/
	return 0;
}

void SmolPool::setDiffConst( const Eref& e, const Qinfo* q, double v )
{
	diffConst_ = v;
	// Here instead of MSsoln I would put in something depending on where
	// the molecule resides. 
	// Likely options are MSfront, MSback, MSup, MSdown
	ErrorCode ret = smolSetSpeciesMobility( sim_, 
		e.element()->getName().c_str(), 
		MSsoln, v,
		0, 0 ); // I'm ignoring drift and difmatrix.
	assert( ret == ECok );
}

double SmolPool::getDiffConst( const Eref& e, const Qinfo* q ) const
{
	return diffConst_;
}

void SmolPool::setSize( const Eref& e, const Qinfo* q, double v )
{
	// Illegal operation.
}

double SmolPool::getSize( const Eref& e, const Qinfo* q ) const
{
	/*
	unsigned int mol = convertIdToMolIndex( e.id() );
	unsigned int index = compartment_[ mol ];
	assert( index < compartmentSize_.size() );
	return compartmentSize_[ index ];
	*/
	return 0;
}

void SmolPool::setSpecies( const Eref& e, const Qinfo* q, unsigned int v )
{
	// species_[ convertIdToMolIndex( e.id() ) ] = v;
}

unsigned int SmolPool::getSpecies( const Eref& e, const Qinfo* q ) const
{
	// return species_[ convertIdToMolIndex( e.id() ) ];
	return 0;
}

//////////////////////////////////////////////////////////////
// Smol conversion functions.
//////////////////////////////////////////////////////////////

// static func
void SmolPool::zombify( Element* solver, Element* orig )
{
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
	orig->zombieSwap( smolPoolCinfo, dh );
}

/// Static func.
void SmolPool::smolSpeciesInit( Element* solver, Element* orig )
{
	Element temp( orig->id(), smolPoolCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	SmolPool* z = reinterpret_cast< SmolPool* >( zer.data() );
	Mol* m = reinterpret_cast< Mol* >( oer.data() );

	ErrorCode ret = smolAddSpecies( z->sim_, orig->getName().c_str(), 0 );
	assert( ret == ECok );


	// Here instead of MSsoln I would put in something depending on where
	// the molecule resides. 
	// Likely options are MSfront, MSback, MSup, MSdown
	ret = smolSetSpeciesMobility( z->sim_, orig->getName().c_str(), 
		MSsoln, m->getDiffConst(), 
		0, 0 ); // I'm ignoring drift and difmatrix.
	z->diffConst_ = m->getDiffConst();

	assert( ret == ECok );
	cout << "added species " << orig->getName() << endl;
}

// static func
void SmolPool::smolMaxNumMolecules( simptr sim, const vector< Id >& pools )
{
	double totNum = 0.0;
	for ( vector< Id >::const_iterator i = pools.begin(); i != pools.end(); ++i )
		totNum += reinterpret_cast< const Mol *>( i->eref().data() )->getNinit();
	
	ErrorCode ret = smolSetMaxMolecules( sim, totNum * 2 );
	assert( ret == ECok );
}

// static func
void SmolPool::smolNinit( Element* solver, Element* orig )
{
	Element temp( orig->id(), smolPoolCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	SmolPool* z = reinterpret_cast< SmolPool* >( zer.data() );
	Mol* m = reinterpret_cast< Mol* >( oer.data() );

	// Here I should really use smolAddCompartmentMolecules.
	vector< double > lowposition( 3, -0.2 );
	vector< double > highposition( 3, 0.2 );
	ErrorCode ret = smolAddSolutionMolecules( z->sim_, 
		orig->getName().c_str(),
		static_cast< int >( m->getNinit() ),
		&lowposition[0],
		&highposition[0] );

	assert( ret == ECok );

	z->nInit_ = m->getNinit();
}

// Static func
void SmolPool::unzombify( Element* zombie )
{
	/*
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	SmolPool* z = reinterpret_cast< SmolPool* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( Mol::initCinfo(), dh );

	Mol* m = reinterpret_cast< Mol* >( oer.data() );

	m->setN( z->getN( zer, 0 ) );
	m->setNinit( z->getNinit( zer, 0 ) );
	m->setSpecies( z->getSpecies( zer, 0 ) );
	*/
}
