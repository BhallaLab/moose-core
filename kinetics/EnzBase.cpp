/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "../basecode/header.h"
#include "../basecode/ElementValueFinfo.h"
#include "lookupVolumeFromMesh.h"
#include "EnzBase.h"

#define EPSILON 1e-15

static SrcFinfo2< double, double > *subOut() {
	static SrcFinfo2< double, double > subOut(
			"subOut",
			"Sends out increment of molecules on product each timestep"
			);
	return &subOut;
}

static SrcFinfo2< double, double > *prdOut() {
	static SrcFinfo2< double, double > prdOut(
			"prdOut",
			"Sends out increment of molecules on product each timestep"
			);
	return &prdOut;
}

const Cinfo* EnzBase::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< EnzBase, double > Km(
			"Km",
			"Michaelis-Menten constant in SI conc units (milliMolar)",
			&EnzBase::setKm,
			&EnzBase::getKm
		);

		static ElementValueFinfo< EnzBase, double > numKm(
			"numKm",
			"Michaelis-Menten constant in number units, volume dependent",
			&EnzBase::setNumKm,
			&EnzBase::getNumKm
		);

		static ElementValueFinfo< EnzBase, double > kcat(
			"kcat",
			"Forward rate constant for enzyme, units 1/sec",
			&EnzBase::setKcat,
			&EnzBase::getKcat
		);

		static ReadOnlyElementValueFinfo< EnzBase, unsigned int > numSub(
			"numSubstrates",
			"Number of substrates in this MM reaction. Usually 1."
			"Does not include the enzyme itself",
			&EnzBase::getNumSub
		);

		static ReadOnlyElementValueFinfo< EnzBase, unsigned int > numPrd(
			"numProducts",
			"Number of products in this MM reaction. Usually 1.",
			&EnzBase::getNumPrd
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo enzDest( "enzDest",
				"Handles # of molecules of Enzyme",
				new OpFunc1< EnzBase, double >( &EnzBase::enz ) );
		static DestFinfo subDest( "subDest",
				"Handles # of molecules of substrate",
				new OpFunc1< EnzBase, double >( &EnzBase::sub ) );
		static DestFinfo prdDest( "prdDest",
				"Handles # of molecules of product. Dummy.",
				new OpFunc1< EnzBase, double >( &EnzBase::prd ) );
		static Finfo* subShared[] = {
			subOut(), &subDest
		};

		static Finfo* prdShared[] = {
			prdOut(), &prdDest
		};
		static SharedFinfo sub( "sub",
			"Connects to substrate molecule",
			subShared, sizeof( subShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo prd( "prd",
			"Connects to product molecule",
			prdShared, sizeof( prdShared ) / sizeof( const Finfo* )
		);

	static Finfo* enzBaseFinfos[] = {
		&Km,	// ElementValue
		&numKm,	// ElementValue
		&kcat,	// Value
		&numSub,	// ReadOnlyElementValue
		&numPrd,	// ReadOnlyElementValue
		&enzDest,			// DestFinfo
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
	};

	static string doc[] =
	{
		"Name", "EnzBase",
		"Author", "Upi Bhalla",
		"Description", "Abstract base class for enzymes."
	};
	static ZeroSizeDinfo< int > dinfo;
	static Cinfo enzBaseCinfo (
		"EnzBase",
		Neutral::initCinfo(),
		enzBaseFinfos,
		sizeof( enzBaseFinfos ) / sizeof ( Finfo* ),
		&dinfo,
		doc,
		sizeof( doc )/sizeof( string ),
		true // Don't create it, it is a an astract base class.
	);

	return &enzBaseCinfo;
}

 static const Cinfo* enzBaseCinfo = EnzBase::initCinfo();

//////////////////////////////////////////////////////////////
// EnzBase internal functions
//////////////////////////////////////////////////////////////

EnzBase::EnzBase()
	: 
		Km_( 1.0e-3 ),
		kcat_( 1.0 )
{;}

EnzBase::~EnzBase( )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void EnzBase::sub( double n )
{;}
void EnzBase::prd( double n ) // dummy
{;}

void EnzBase::enz( double n )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void EnzBase::setKm( const Eref& enz, double v )
{
	if ( v < EPSILON )
		v = EPSILON;
	vSetKm( enz, v );
}

double EnzBase::getKm( const Eref& enz ) const
{
	return Km_;
}

void EnzBase::setNumKm( const Eref& enz, double v )
{
	double volScale = convertConcToNumRateUsingMesh( enz, subOut(), 1 );
	Km_ = v / volScale;
	vSetKm( enz, Km_ );
}

double EnzBase::getNumKm( const Eref& enz ) const
{
	double volScale = convertConcToNumRateUsingMesh( enz, subOut(), 1 );
	return Km_ * volScale;
}


void EnzBase::setKcat( const Eref& e, double v )
{
	// We don't set it here because we need the old value in the Enz
	// function. 
	if ( v < 0.0 )
		v = 0.0;
	vSetKcat( e, v );
}

double EnzBase::getKcat( const Eref& e ) const
{
		return kcat_;
}

unsigned int EnzBase::getNumSub( const Eref& e ) const
{
	const vector< MsgFuncBinding >* mfb =
		e.element()->getMsgAndFunc( subOut()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}

unsigned int EnzBase::getNumPrd( const Eref& e ) const
{
	const vector< MsgFuncBinding >* mfb =
		e.element()->getMsgAndFunc( prdOut()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}
