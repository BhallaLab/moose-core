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
#include "EnzBase.h"

#define EPSILON 1e-15

static SrcFinfo2< double, double > *toSub() {
	static SrcFinfo2< double, double > toSub( 
			"toSub", 
			"Sends out increment of molecules on product each timestep"
			);
	return &toSub;
}

static SrcFinfo2< double, double > *toPrd() {
	static SrcFinfo2< double, double > toPrd( 
			"toPrd", 
			"Sends out increment of molecules on product each timestep"
			);
	return &toPrd;
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


		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< EnzBase >( &EnzBase::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< EnzBase >( &EnzBase::reinit ) );

		static DestFinfo group( "group",
			"Handle for group msgs. Doesn't do anything",
			new OpFuncDummy() );

		static DestFinfo remesh( "remesh",
			"Tells the MMEnz to recompute its numKm after remeshing",
			new EpFunc0< EnzBase >( &EnzBase::remesh ) );

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
			toSub(), &subDest
		};

		static Finfo* prdShared[] = {
			toPrd(), &prdDest
		};
		static SharedFinfo sub( "sub",
			"Connects to substrate molecule",
			subShared, sizeof( subShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo prd( "prd",
			"Connects to product molecule",
			prdShared, sizeof( prdShared ) / sizeof( const Finfo* )
		);
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* enzBaseFinfos[] = {
		&Km,	// ElementValue
		&numKm,	// ElementValue
		&kcat,	// Value
		&numSub,	// ReadOnlyElementValue
		&enzDest,			// DestFinfo
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&proc,				// SharedFinfo
		&remesh,			// Destfinfo
	};

	static Cinfo enzBaseCinfo (
		"EnzBase",
		Neutral::initCinfo(),
		enzBaseFinfos,
		sizeof( enzBaseFinfos ) / sizeof ( Finfo* ),
		new ZeroSizeDinfo< int >()
	);

	return &enzBaseCinfo;
}

 static const Cinfo* enzBaseCinfo = EnzBase::initCinfo();

//////////////////////////////////////////////////////////////
// EnzBase internal functions
//////////////////////////////////////////////////////////////

EnzBase::EnzBase( )
{;}

EnzBase::~EnzBase( )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void EnzBase::sub( double n )
{
	vSub( n );
}

void EnzBase::prd( double n ) // dummy
{;}

void EnzBase::enz( double n )
{
	vEnz( n );
}

void EnzBase::process( const Eref& e, ProcPtr p )
{
	vProcess( e, p );
}

void EnzBase::reinit( const Eref& e, ProcPtr p )
{
	vReinit( e, p );
}

void EnzBase::remesh( const Eref& e, const Qinfo* q )
{
	vRemesh( e, q );
}

//////////////////////////////////////////////////////////////
// Virtual MsgDest Definitions. Mostly dummys, the derived classes don't
// need to do anything here.
//////////////////////////////////////////////////////////////
void EnzBase::vSub( double n )
{;}

void EnzBase::vEnz( double n )
{;}

void EnzBase::vProcess( const Eref& e, ProcPtr p )
{;}

void EnzBase::vReinit( const Eref& e, ProcPtr p )
{;}

void EnzBase::vRemesh( const Eref& e, const Qinfo* q )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void EnzBase::setKm( const Eref& enz, const Qinfo* q, double v )
{
		vSetKm( enz, q, v );
}

double EnzBase::getKm( const Eref& enz, const Qinfo* q ) const
{
		return vGetKm( enz, q );
}

void EnzBase::setNumKm( const Eref& enz, const Qinfo* q, double v )
{
		vSetNumKm( enz, q, v );
}

double EnzBase::getNumKm( const Eref& enz, const Qinfo* q ) const
{
		return vGetNumKm( enz, q );
}


void EnzBase::setKcat( const Eref& e, const Qinfo* q, double v )
{
		vSetKcat( e, q, v );
}

double EnzBase::getKcat( const Eref& e, const Qinfo* q ) const
{
		return vGetKcat( e, q );
}

unsigned int EnzBase::getNumSub( const Eref& e, const Qinfo* q ) const
{
	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( toSub()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}

////////////////////////////////////////////////////////////////////////
// Zombie conversion routine to convert between Enz subclasses.
////////////////////////////////////////////////////////////////////////
// static func

/**
 * This function helps the conversion between Enzyme subclasses. Note that
 * we may need a second zombify function to convert to and from explicit enz
 * classes because there is information lost if we go right down to the
 * EnzBase. Specifically, EnzBase only knows about two parameters, the
 * Km and kcat. Explicit enzymes also need to know a k2, or equivalently
 * a ratio between kcat and k2. But in principle this function allows
 * conversion between the two cases.
 */
void EnzBase::zombify( Element* orig, const Cinfo* zClass, Id solver )
{
	if ( orig->cinfo() == zClass )
		return;
	DataHandler* origHandler = orig->dataHandler();
	DataHandler* dh = origHandler->copyUsingNewDinfo( zClass->dinfo() );
	Element temp( orig->id(), zClass, dh );
	Eref zombier( &temp, 0 );

	EnzBase* z = reinterpret_cast< EnzBase* >( zombier.data() );
	Eref oer( orig, 0 );

	const EnzBase* m = reinterpret_cast< EnzBase* >( oer.data() );
	z->setSolver( solver, orig->id() ); // call virtual func to assign solver info.
	// May need to extend to entire array.
	z->vSetKm( zombier, 0, m->vGetKm( oer, 0 ) );
	z->vSetKcat( zombier, 0, m->vGetKcat( oer, 0 ) );
	orig->zombieSwap( zClass, dh );
	delete origHandler;
}

// Virtual func: default does nothing.
void EnzBase::setSolver( Id solver, Id orig )
{
	;
}

