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
#include "ReacBase.h"

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

const Cinfo* ReacBase::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ReacBase, double > kf(
			"kf",
			"Forward rate constant, in # units",
			&ReacBase::setNumKf,
			&ReacBase::getNumKf
		);

		static ElementValueFinfo< ReacBase, double > kb(
			"kb",
			"Reverse rate constant, in # units",
			&ReacBase::setNumKb,
			&ReacBase::getNumKb
		);

		static ElementValueFinfo< ReacBase, double > Kf(
			"Kf",
			"Forward rate constant, in concentration units",
			&ReacBase::setConcKf,
			&ReacBase::getConcKf
		);

		static ElementValueFinfo< ReacBase, double > Kb(
			"Kb",
			"Reverse rate constant, in concentration units",
			&ReacBase::setConcKb,
			&ReacBase::getConcKb
		);

		static ReadOnlyElementValueFinfo< ReacBase, unsigned int > numSub(
			"numSubstrates",
			"Number of substrates of reaction",
			&ReacBase::getNumSub
		);

		static ReadOnlyElementValueFinfo< ReacBase, unsigned int > numPrd(
			"numProducts",
			"Number of products of reaction",
			&ReacBase::getNumPrd
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< ReacBase >( &ReacBase::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< ReacBase >( &ReacBase::reinit ) );

		static DestFinfo group( "group",
			"Handle for group msgs. Doesn't do anything",
			new OpFuncDummy() );

		static DestFinfo remesh( "remesh",
			"Tells the reac to recompute its numRates, as remeshing has happened",
			new EpFunc0< ReacBase >( & ReacBase::remesh ) );

		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo subDest( "subDest",
				"Handles # of molecules of substrate",
				new OpFunc1< ReacBase, double >( &ReacBase::sub ) );
		static DestFinfo prdDest( "prdDest",
				"Handles # of molecules of product",
				new OpFunc1< ReacBase, double >( &ReacBase::prd ) );
		static Finfo* subShared[] = {
			toSub(), &subDest
		};
		static Finfo* prdShared[] = {
			toPrd(), &prdDest
		};
		static SharedFinfo sub( "sub",
			"Connects to substrate pool",
			subShared, sizeof( subShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo prd( "prd",
			"Connects to substrate pool",
			prdShared, sizeof( prdShared ) / sizeof( const Finfo* )
		);
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);


	static Finfo* reacFinfos[] = {
		&kf,	// Value
		&kb,	// Value
		&Kf,	// Value
		&Kb,	// Value
		&numSub,	// ReadOnlyValue
		&numPrd,	// ReadOnlyValue
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&proc,				// SharedFinfo
		&remesh,	// DestFinfo
	};

	static string doc[] = 
	{
		"Name", "ReacBase",
		"Author", "Upinder S. Bhalla, 2012, NCBS",
		"Description", "Base class for reactions. Provides the MOOSE API"
		"functions, but ruthlessly refers almost all of them to derived"
		"classes, which have to provide the man page output."
	};

	static Cinfo reacBaseCinfo (
		"ReacBase",
		Neutral::initCinfo(),
		reacFinfos,
		sizeof( reacFinfos ) / sizeof ( Finfo* ),
		new ZeroSizeDinfo< int>()
	);

	return &reacBaseCinfo;
}

 static const Cinfo* reacBaseCinfo = ReacBase::initCinfo();

//////////////////////////////////////////////////////////////
// ReacBase internal functions
//////////////////////////////////////////////////////////////


ReacBase::ReacBase( )
	: concKf_( 0.1 ), concKb_( 0.2 )
{
	;
}

ReacBase::~ReacBase( )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ReacBase::sub( double v )
{
	vSub( v );
}

void ReacBase::prd( double v )
{
	vPrd( v );
}

void ReacBase::process( const Eref& e, ProcPtr p )
{
	vProcess( e, p );
}

void ReacBase::reinit( const Eref& e, ProcPtr p )
{
	vReinit( e, p );
}

void ReacBase::remesh( const Eref& e, const Qinfo* q )
{
	vRemesh( e, q );
}
//////////////////////////////////////////////////////////////
//
// Virtual MsgDest Definitions, all dummies, but many derived classes
// will want to use these dummies.
//////////////////////////////////////////////////////////////

void ReacBase::vSub( double v )
{
		;
}

void ReacBase::vPrd( double v )
{
		;
}

void ReacBase::vProcess( const Eref& e, ProcPtr p )
{
		;
}

void ReacBase::vReinit( const Eref& e, ProcPtr p )
{
		;
}

void ReacBase::vRemesh( const Eref& e, const Qinfo* q )
{
		;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ReacBase::setNumKf( const Eref& e, const Qinfo* q, double v )
{
	vSetNumKf( e, q, v );
}

double ReacBase::getNumKf( const Eref& e, const Qinfo* q) const
{
	return vGetNumKf( e, q );
}

void ReacBase::setNumKb( const Eref& e, const Qinfo* q, double v )
{
	vSetNumKb( e, q, v );
}

double ReacBase::getNumKb( const Eref& e, const Qinfo* q ) const
{
	return vGetNumKb( e, q );
}

////////////////////////////////////////////////////////////////////////

void ReacBase::setConcKf( const Eref& e, const Qinfo* q, double v )
{
	vSetConcKf( e, q, v );
}

double ReacBase::getConcKf( const Eref& e, const Qinfo* q ) const
{
	return vGetConcKf( e, q );
}

void ReacBase::setConcKb( const Eref& e, const Qinfo* q, double v )
{
	vSetConcKb( e, q, v );
}

double ReacBase::getConcKb( const Eref& e, const Qinfo* q ) const
{
	return vGetConcKb( e, q );
}

unsigned int ReacBase::getNumSub( const Eref& e, const Qinfo* q ) const
{
	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( toSub()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}

unsigned int ReacBase::getNumPrd( const Eref& e, const Qinfo* q ) const
{
	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( toPrd()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}

////////////////////////////////////////////////////////////////////////
// Zombie conversion routine to convert between Reac subclasses.
////////////////////////////////////////////////////////////////////////
// static func

void ReacBase::zombify( Element* orig, const Cinfo* zClass, Id solver )
{
	if ( orig->cinfo() == zClass )
		return;
	DataHandler* origHandler = orig->dataHandler();
	DataHandler* dh = origHandler->copyUsingNewDinfo( zClass->dinfo() );
	Element temp( orig->id(), zClass, dh );
	Eref zombier( &temp, 0 );

	ReacBase* z = reinterpret_cast< ReacBase* >( zombier.data() );
	Eref oer( orig, 0 );

	const ReacBase* m = reinterpret_cast< ReacBase* >( oer.data() );
	z->setSolver( solver, orig->id() ); // call virtual func to assign solver info.
	// May need to extend to entire array.
	z->vSetConcKf( oer, 0, m->vGetConcKf( oer, 0 ) );
	z->vSetConcKb( oer, 0, m->vGetConcKb( oer, 0 ) );
	/*
	z->vSetConcKf( zombier, 0, m->vGetConcKf( oer, 0 ) );
	z->vSetConcKb( zombier, 0, m->vGetConcKb( oer, 0 ) );
	*/
	orig->zombieSwap( zClass, dh );
	delete origHandler;
}

// Virtual func: default does nothing.
void ReacBase::setSolver( Id solver, Id orig )
{
	;
}

