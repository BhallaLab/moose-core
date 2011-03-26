/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Mol.h"

#define EPSILON 1e-15

static const double NA = 6.023e23;

const Cinfo* Mol::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Mol, double > n(
			"n",
			"Number of molecules",
			&Mol::setN,
			&Mol::getN
		);

		static ValueFinfo< Mol, double > nInit(
			"nInit",
			"Initial value of number of molecules",
			&Mol::setNinit,
			&Mol::getNinit
		);

		static ValueFinfo< Mol, double > diffConst(
			"diffConst",
			"Diffusion constant of molecule",
			&Mol::setDiffConst,
			&Mol::getDiffConst
		);

		static ValueFinfo< Mol, double > conc(
			"conc",
			"Concentration of molecules",
			&Mol::setConc,
			&Mol::getConc
		);

		static ValueFinfo< Mol, double > concInit(
			"concInit",
			"Initial value of molecular concentration",
			&Mol::setConcInit,
			&Mol::getConcInit
		);

		static ReadOnlyValueFinfo< Mol, double > size(
			"size",
			"Size of compartment. Units are SI. "
			"Utility field, the master size info is "
			"stored on the compartment itself. For voxel-based spatial"
			"models, the 'size' of the molecule at a given index is the"
			"size of that voxel.",
			&Mol::getSize
		);

		static ValueFinfo< Mol, unsigned int > species(
			"species",
			"Species identifier for this mol pool. Eventually link to ontology.",
			&Mol::setSpecies,
			&Mol::getSpecies
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Mol >( &Mol::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< Mol >( &Mol::reinit ) );

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo reacDest( "reacDest",
			"Handles reaction input",
			new OpFunc2< Mol, double, double >( &Mol::reac )
		);

		static DestFinfo setSize( "setSize",
			"Separate finfo to assign size, should only be used by compartment."
			"Defaults to SI units of volume: m^3",
			new OpFunc1< Mol, double >( &Mol::setSize )
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

	static Finfo* molFinfos[] = {
		&n,			// Value
		&nInit,		// Value
		&diffConst,	// Value
		&conc,		// Value
		&concInit,	// Value
		&size,		// Readonly Value
		&species,	// Value
		&group,			// DestFinfo
		&setSize,			// DestFinfo
		&reac,				// SharedFinfo
		&proc,				// SharedFinfo
	};

	static Cinfo molCinfo (
		"Mol",
		Neutral::initCinfo(),
		molFinfos,
		sizeof( molFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Mol >()
	);

	return &molCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* molCinfo = Mol::initCinfo();
const SrcFinfo1< double >& nOut = 
	*dynamic_cast< const SrcFinfo1< double >* >( 
	molCinfo->findFinfo( "nOut" ) );


Mol::Mol()
	: n_( 0.0 ), nInit_( 0.0 ), size_( 1.0 ), diffConst_( 0.0 ),
		A_( 0.0 ), B_( 0.0 ), species_( 0 )
{;}

Mol::Mol( double nInit)
	: n_( 0.0 ), nInit_( nInit ), size_( 1.0 ), diffConst_( 0.0 ),
		A_( 0.0 ), B_( 0.0 ), species_( 0 )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Mol::process( const Eref& e, ProcPtr p )
{
	// double A = e.sumBuf( aSlot );
	// double B = e.sumBuf( bSlot );
	if ( n_ > EPSILON && B_ > EPSILON ) {
		double C = exp( -B_ * p->dt / n_ );
		n_ *= C + (A_ / B_ ) * ( 1.0 - C );
	} else {
		n_ += ( A_ - B_ ) * p->dt;
		if ( n_ < 0.0 )
			n_ = 0.0;
	}

	A_ = B_ = 0.0;

	nOut.send( e, p, n_ );
}

void Mol::reinit( const Eref& e, ProcPtr p )
{
	A_ = B_ = 0.0;
	n_ = nInit_;

	nOut.send( e, p, n_ );
}

void Mol::reac( double A, double B )
{
	A_ += A;
	B_ += B;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Mol::setN( double v )
{
	n_ = v;
}

double Mol::getN() const
{
	return n_;
}

void Mol::setNinit( double v )
{
	n_ = nInit_ = v;
}

double Mol::getNinit() const
{
	return nInit_;
}

void Mol::setConc( double c ) // Conc is given micromolar. Size is in m^3
{
	n_ = 1e-3 * NA * c * size_;
}

double Mol::getConc() const // Returns conc in micromolar.
{
	return 1e3 * (n_ / NA) / size_;
}

void Mol::setConcInit( double c )
{
	nInit_ = 1e-3 * NA * c * size_;
}

double Mol::getConcInit() const
{
	return 1e3 * ( nInit_ / NA ) / size_;
}

void Mol::setDiffConst( double v )
{
	diffConst_ = v;
}

double Mol::getDiffConst() const
{
	return diffConst_;
}

void Mol::setSize( double v )
{
	size_ = v;
}

double Mol::getSize() const
{
	return size_;
}

void Mol::setSpecies( unsigned int v )
{
	species_ = v;
}

unsigned int Mol::getSpecies() const
{
	return species_;
}
