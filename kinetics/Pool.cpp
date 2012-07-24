/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "lookupSizeFromMesh.h"
#include "PoolBase.h"
#include "Pool.h"

#define EPSILON 1e-15

const Cinfo* Pool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions: All inherited from Pool.
		//////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions: All but increment and decrement inherited
		//////////////////////////////////////////////////////////////
		static DestFinfo increment( "increment",
			"Increments mol numbers by specified amount. Can be +ve or -ve",
			new OpFunc1< Pool, double >( &Pool::increment )
		);

		static DestFinfo decrement( "decrement",
			"Decrements mol numbers by specified amount. Can be +ve or -ve",
			new OpFunc1< Pool, double >( &Pool::decrement )
		);

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions: All inherited.
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions: All inherited.
		//////////////////////////////////////////////////////////////
	static Finfo* poolFinfos[] = {
		&increment,			// DestFinfo
		&decrement,			// DestFinfo
	};

	static Cinfo poolCinfo (
		"Pool",
		PoolBase::initCinfo(),
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

const SrcFinfo1< double >* requestSize = 
	dynamic_cast< const SrcFinfo1< double >* >( 
	poolCinfo->findFinfo( "requestSize" ) );

Pool::Pool()
	: n_( 0.0 ), concInit_( 0.0 ), diffConst_( 0.0 ),
		A_( 0.0 ), B_( 0.0 ), species_( 0 )
{;}

Pool::~Pool()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Pool::vProcess( const Eref& e, ProcPtr p )
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

void Pool::vReinit( const Eref& e, ProcPtr p )
{
	A_ = B_ = 0.0;
	n_ = getNinit( e, 0 );

	nOut.send( e, p->threadIndexInGroup, n_ );
}

void Pool::vReac( double A, double B )
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

void Pool::vRemesh( const Eref& e, const Qinfo* q, 
	unsigned int numTotalEntries, unsigned int startEntry, 
	const vector< unsigned int >& localIndices, 
	const vector< double >& vols )
{
	if ( e.index().value() != 0 )
		return;
	if ( q->addToStructuralQ() )
		return;
	Neutral* n = reinterpret_cast< Neutral* >( e.data() );
	assert( vols.size() > 0 );
	double concInit = concInit_; // replace when we fix the conc access
	if ( vols.size() != e.element()->dataHandler()->localEntries() )
		n->setLastDimension( e, q, vols.size() );
	// Note that at this point the Pool pointer may be invalid!
	// But we need to update the concs anyway.
	assert( e.element()->dataHandler()->localEntries() == vols.size() );
	Pool* pooldata = reinterpret_cast< Pool* >( e.data() );
	for ( unsigned int i = 0; i < vols.size(); ++i ) {
		pooldata[i].n_ = concInit * vols[i] * NA;
		pooldata[i].concInit_ = concInit;
	}
}

void Pool::vHandleMolWt( const Eref& e, const Qinfo* q, double v )
{
	; // Here I should update DiffConst too.
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Pool::vSetN( const Eref& e, const Qinfo* q, double v )
{
//	conc_ =  v /  ( NA * lookupSizeFromMesh( e, requestSize() ) );
	n_ = v;
}

double Pool::vGetN( const Eref& e, const Qinfo*q ) const
{
	return n_;
// 	return NA * conc_ * lookupSizeFromMesh( e, requestSize() );
}

void Pool::vSetNinit( const Eref& e, const Qinfo* q, double v )
{
	concInit_ =  v /  ( NA * lookupSizeFromMesh( e, requestSize ) );
}

double Pool::vGetNinit( const Eref& e, const Qinfo* q ) const
{
	return NA * concInit_ * lookupSizeFromMesh( e, requestSize );
}

// Conc is given in millimolar. Size is in m^3
void Pool::vSetConc( const Eref& e, const Qinfo* q, double c ) 
{
	n_ = NA * c * lookupSizeFromMesh( e, requestSize );
}

// Returns conc in millimolar.
double Pool::vGetConc( const Eref& e, const Qinfo* q ) const
{
	return (n_ / NA) / lookupSizeFromMesh( e, requestSize );
}

void Pool::vSetConcInit( const Eref& e, const Qinfo* q, double c )
{
	// nInit_ = NA * c * lookupSizeFromMesh( e, requestSize() );
	concInit_ = c;
}

double Pool::vGetConcInit( const Eref& e, const Qinfo* q ) const
{
	return concInit_;
	// return ( nInit_ / NA ) / lookupSizeFromMesh( e, requestSize() );
}

void Pool::vSetDiffConst( const Eref& e, const Qinfo* q, double v )
{
	diffConst_ = v;
}

double Pool::vGetDiffConst( const Eref& e, const Qinfo* q ) const
{
	return diffConst_;
}

void Pool::vSetSize( const Eref& e, const Qinfo* q,  double v )
{
	assert( 0 ); // Don't currently know how to do this.
}

double Pool::vGetSize( const Eref& e, const Qinfo* q ) const
{
	return lookupSizeFromMesh( e, requestSize );
}

void Pool::vSetSpecies( const Eref& e, const Qinfo* q,  SpeciesId v )
{
	species_ = v;
}

SpeciesId Pool::vGetSpecies( const Eref& e, const Qinfo* q ) const
{
	return species_;
}
