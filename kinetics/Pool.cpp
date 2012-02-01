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
#include "lookupSizeFromMesh.h"
#include "Pool.h"

#define EPSILON 1e-15

const SpeciesId DefaultSpeciesId = 0;

static SrcFinfo1< double >* requestSize() {
	static SrcFinfo1< double > requestSize( 
		"requestSize", 
		"Requests Size of pool from matching mesh entry"
	);
	return &requestSize;
}

const Cinfo* Pool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Pool, double > n(
			"n",
			"Number of molecules in pool",
			&Pool::setN,
			&Pool::getN
		);

		static ValueFinfo< Pool, double > nInit(
			"nInit",
			"Initial value of number of molecules in pool",
			&Pool::setNinit,
			&Pool::getNinit
		);

		static ValueFinfo< Pool, double > diffConst(
			"diffConst",
			"Diffusion constant of molecule",
			&Pool::setDiffConst,
			&Pool::getDiffConst
		);

		static ElementValueFinfo< Pool, double > conc(
			"conc",
			"Concentration of molecules in this pool",
			&Pool::setConc,
			&Pool::getConc
		);

		static ElementValueFinfo< Pool, double > concInit(
			"concInit",
			"Initial value of molecular concentration in pool",
			&Pool::setConcInit,
			&Pool::getConcInit
		);

		static ReadOnlyElementValueFinfo< Pool, double > size(
			"size",
			"Size of compartment. Units are SI. "
			"Utility field, the actual size info is "
			"stored on a volume mesh entry in the parent compartment."
			"This is hooked up by a message. If the message isn't"
			"available size is just taken as 1",
			&Pool::getSize
		);

		static ValueFinfo< Pool, unsigned int > speciesId(
			"speciesId",
			"Species identifier for this mol pool. Eventually link to ontology.",
			&Pool::setSpecies,
			&Pool::getSpecies
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Pool >( &Pool::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< Pool >( &Pool::reinit ) );

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo reacDest( "reacDest",
			"Handles reaction input",
			new OpFunc2< Pool, double, double >( &Pool::reac )
		);

		static DestFinfo increment( "increment",
			"Increments mol numbers by specified amount. Can be +ve or -ve",
			new OpFunc1< Pool, double >( &Pool::increment )
		);

		static DestFinfo decrement( "decrement",
			"Decrements mol numbers by specified amount. Can be +ve or -ve",
			new OpFunc1< Pool, double >( &Pool::decrement )
		);

		static DestFinfo handleMolWt( "handleMolWt",
			"Separate finfo to assign molWt, and consequently diffusion const."
			"Should only be used in SharedMsg with species.",
			new OpFunc1< Pool, double >( &Pool::handleMolWt )
		);

		static DestFinfo remesh( "remesh",
			"Handle commands to remesh the pool. This may involve changing "
			"the number of pool entries, as well as changing their volumes",
			new EpFunc4< Pool, unsigned int, unsigned int, vector< unsigned int >, vector< double > >( &Pool::remesh )
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
			&remesh, requestSize()
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
		&size,		// Readonly Value
		&speciesId,	// Value
		&group,			// DestFinfo
		&increment,			// DestFinfo
		&decrement,			// DestFinfo
//		requestSize(),		// SrcFinfo, but defined in SharedFinfo
		&reac,				// SharedFinfo
		&proc,				// SharedFinfo
		&species,			// SharedFinfo
		&mesh,				// SharedFinfo
	};

	static Cinfo poolCinfo (
		"Pool",
		Neutral::initCinfo(),
		poolFinfos,
		sizeof( poolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Pool >()
	);

	return &poolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* poolCinfo = Pool::initCinfo();
const SrcFinfo1< double >& nOut = 
	*dynamic_cast< const SrcFinfo1< double >* >( 
	poolCinfo->findFinfo( "nOut" ) );


Pool::Pool()
	: n_( 0.0 ), nInit_( 0.0 ), diffConst_( 0.0 ),
		A_( 0.0 ), B_( 0.0 ), species_( 0 )
{;}

Pool::Pool( double nInit)
	: n_( 0.0 ), nInit_( nInit ), diffConst_( 0.0 ),
		A_( 0.0 ), B_( 0.0 ), species_( 0 )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Pool::process( const Eref& e, ProcPtr p )
{
	// double A = e.sumBuf( aSlot );
	// double B = e.sumBuf( bSlot );
	if ( n_ < 0 )
		cout << "nugh" << e.index() << endl;
	if ( B_ < 0 )
		cout << "bugh" << e.index() << endl;
	if ( p->dt < 0 )
		cout << "tugh" << e.index() << endl;

	if ( n_ > EPSILON && B_ > EPSILON ) {
		double C = exp( -B_ * p->dt / n_ );
		n_ *= C + (A_ / B_ ) * ( 1.0 - C );
	} else {
		n_ += ( A_ - B_ ) * p->dt;
		if ( n_ < 0.0 )
			n_ = 0.0;
	}

	A_ = B_ = 0.0;

	nOut.send( e, p->threadIndexInGroup, n_ );
}

void Pool::reinit( const Eref& e, ProcPtr p )
{
	A_ = B_ = 0.0;
	n_ = nInit_;

	nOut.send( e, p->threadIndexInGroup, n_ );
}

void Pool::reac( double A, double B )
{
	A_ += A;
	B_ += B;
}

void Pool::increment( double val )
{
	if ( val > 0 )
		A_ += val;
	else
		B_ -= val;
}

void Pool::decrement( double val )
{
	if ( val < 0 )
		A_ -= val;
	else
		B_ += val;
}

void Pool::remesh( const Eref& e, const Qinfo* q, 
	unsigned int numTotalEntries, unsigned int startEntry, 
	vector< unsigned int > localIndices, vector< double > vols )
{
	cout << "In Pool::remesh for " << e << endl;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Pool::setN( double v )
{
	n_ = v;
}

double Pool::getN() const
{
	return n_;
}

void Pool::setNinit( double v )
{
	n_ = nInit_ = v;
}

double Pool::getNinit() const
{
	return nInit_;
}

/// Utility function
/*
static double lookupSize( const Eref& e )
{
	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( requestSize()->getBindIndex() );
	if ( !mfb ) return 1.0;
	if ( mfb->size() == 0 ) return 1.0;

	double size = 
		Field< double >::fastGet( e, (*mfb)[0].mid, (*mfb)[0].fid );

	if ( size <= 0 ) size = 1.0;

	return size;
}
*/

// Conc is given in millimolar. Size is in m^3
void Pool::setConc( const Eref& e, const Qinfo* q, double c ) 
{
	
	n_ = NA * c * lookupSizeFromMesh( e, requestSize() );
}

// Returns conc in millimolar.
double Pool::getConc( const Eref& e, const Qinfo* q ) const
{
	return (n_ / NA) / lookupSizeFromMesh( e, requestSize() );
}

void Pool::setConcInit( const Eref& e, const Qinfo* q, double c )
{
	nInit_ = NA * c * lookupSizeFromMesh( e, requestSize() );
}

double Pool::getConcInit( const Eref& e, const Qinfo* q ) const
{
	return ( nInit_ / NA ) / lookupSizeFromMesh( e, requestSize() );
}

void Pool::setDiffConst( double v )
{
	diffConst_ = v;
}

double Pool::getDiffConst() const
{
	return diffConst_;
}

double Pool::getSize( const Eref& e, const Qinfo* q ) const
{
	return lookupSizeFromMesh( e, requestSize() );
}

void Pool::setSpecies( unsigned int v )
{
	species_ = v;
}

unsigned int Pool::getSpecies() const
{
	return species_;
}

void Pool::handleMolWt( double v )
{
	; // Here I should update DiffConst too.
}
