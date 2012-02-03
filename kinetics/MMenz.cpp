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
#include "MMenz.h"

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

const Cinfo* MMenz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< MMenz, double > Km(
			"Km",
			"Michaelis-Menten constant in SI conc units (milliMolar)",
			&MMenz::setKm,
			&MMenz::getKm
		);

		static ElementValueFinfo< MMenz, double > numKm(
			"numKm",
			"Michaelis-Menten constant in number units, volume dependent",
			&MMenz::setNumKm,
			&MMenz::getNumKm
		);

		static ValueFinfo< MMenz, double > kcat(
			"kcat",
			"Forward rate constant for enzyme, units 1/sec",
			&MMenz::setKcat,
			&MMenz::getKcat
		);

		static ReadOnlyElementValueFinfo< MMenz, unsigned int > numSub(
			"numSubstrates",
			"Number of substrates in this MM reaction. Usually 1."
			"Does not include the enzyme itself",
			&MMenz::getNumSub
		);


		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< MMenz >( &MMenz::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< MMenz >( &MMenz::reinit ) );

		static DestFinfo group( "group",
			"Handle for group msgs. Doesn't do anything",
			new OpFuncDummy() );

		static DestFinfo remesh( "remesh",
			"Tells the MMEnz to recompute its numKm after remeshing",
			new EpFunc0< MMenz >( &MMenz::remesh ) );

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo enzDest( "enz",
				"Handles # of molecules of MMenzyme",
				new OpFunc1< MMenz, double >( &MMenz::enz ) );
		static DestFinfo subDest( "subDest",
				"Handles # of molecules of substrate",
				new OpFunc1< MMenz, double >( &MMenz::sub ) );
		static DestFinfo prdDest( "prdDest",
				"Handles # of molecules of product. Dummy.",
				new OpFunc1< MMenz, double >( &MMenz::prd ) );
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

	static Finfo* mmEnzFinfos[] = {
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

	static Cinfo mmEnzCinfo (
		"MMenz",
		Neutral::initCinfo(),
		mmEnzFinfos,
		sizeof( mmEnzFinfos ) / sizeof ( Finfo* ),
		new Dinfo< MMenz >()
	);

	return &mmEnzCinfo;
}

 static const Cinfo* mmEnzCinfo = MMenz::initCinfo();

//////////////////////////////////////////////////////////////
// MMenz internal functions
//////////////////////////////////////////////////////////////


MMenz::MMenz( )
	: Km_( 5 ), kcat_( 0.1 ), sub_( 0.0 ), enz_( 0.0 )
{
	;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void MMenz::sub( double n )
{
	sub_ *= n;
}

void MMenz::prd( double n ) // dummy
{
	;
}

void MMenz::enz( double n )
{
	enz_ = n;
}

void MMenz::process( const Eref& e, ProcPtr p )
{
	double rate = kcat_ * enz_ * sub_ / ( numKm_ + sub_ );
	toSub()->send( e, p->threadIndexInGroup, 0, rate );
	toPrd()->send( e, p->threadIndexInGroup, rate, 0 );
	
	sub_ = 1.0;
}

void MMenz::reinit( const Eref& e, ProcPtr p )
{
	sub_ = 1.0;
	enz_ = 0.0;
}

void MMenz::remesh( const Eref& e, const Qinfo* q )
{
	cout << "MMenz::remesh for " << e << endl;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void MMenz::setKm( const Eref& enz, const Qinfo* q, double v )
{
	Km_ = v;
	double volScale = convertConcToNumRateUsingMesh( enz, toSub(), 1 );
	numKm_ = v * volScale;
}

double MMenz::getKm( const Eref& enz, const Qinfo* q ) const
{
	return Km_;
}

void MMenz::setNumKm( const Eref& enz, const Qinfo* q, double v )
{
	double volScale = convertConcToNumRateUsingMesh( enz, toSub(), 1 );
	numKm_ = v;
	Km_ = v / volScale;
}

double MMenz::getNumKm( const Eref& enz, const Qinfo* q ) const
{
	double volScale = convertConcToNumRateUsingMesh( enz, toSub(), 1 );
	return Km_ * volScale;
}


void MMenz::setKcat( double v )
{
	kcat_ = v;
}

double MMenz::getKcat() const
{
	return kcat_;
}

unsigned int MMenz::getNumSub( const Eref& e, const Qinfo* q ) const
{
	const vector< MsgFuncBinding >* mfb = 
		e.element()->getMsgAndFunc( toSub()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}
