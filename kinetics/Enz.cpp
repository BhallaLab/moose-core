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
// #include "FuncTerm.h"
#include "../basecode/SparseMatrix.h"
// #include "KinSparseMatrix.h"
// #include "VoxelPoolsBase.h"
// #include "../mesh/VoxelJunction.h"
// #include "XferInfo.h"
// #include "KsolveBase.h"
// #include "Stoich.h"

#include "../ksolve/RateTerm.h"
#include "../ksolve/KinSparseMatrix.h"
class KsolveBase;
#include "../ksolve/Stoich.h"
#include "EnzBase.h"
#include "EnzBase.h"
#include "Enz.h"

#define EPSILON 1e-15


static SrcFinfo2< double, double > *enzOut() {
	static SrcFinfo2< double, double > enzOut(
			"enzOut",
			"Sends out increment of molecules on product each timestep"
			);
	return &enzOut;
}

static SrcFinfo2< double, double > *cplxOut() {
	static SrcFinfo2< double, double > cplxOut(
			"cplxOut",
			"Sends out increment of molecules on product each timestep"
			);
	return &cplxOut;
}

DestFinfo* enzDest()
{
	static const Finfo* f1 = EnzBase::initCinfo()->findFinfo( "enzDest" );
	static const DestFinfo* f2 = dynamic_cast< const DestFinfo* >( f1 );
	static DestFinfo* enzDest = const_cast< DestFinfo* >( f2 );
	assert( f1 );
	assert( f2 );
	return enzDest;
}
const Cinfo* Enz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< Enz, double > k1(
			"k1",
			"Forward reaction from enz + sub to complex, in # units."
			"This parameter is subordinate to the Km. This means that"
			"when Km is changed, this changes. It also means that when"
			"k2 or k3 (aka kcat) are changed, we assume that Km remains"
			"fixed, and as a result k1 must change. It is only when"
			"k1 is assigned directly that we assume that the user knows"
			"what they are doing, and we adjust Km accordingly."
			"k1 is also subordinate to the 'ratio' field, since setting "
			"the ratio reassigns k2."
			"Should you wish to assign the elementary rates k1, k2, k3,"
		    "of an enzyme directly, always assign k1 last.",
			&Enz::setK1,
			&Enz::getK1
		);

		static ElementValueFinfo< Enz, double > k2(
			"k2",
			"Reverse reaction from complex to enz + sub",
			&Enz::setK2,
			&Enz::getK2
		);

		static ElementValueFinfo< Enz, double > k3(
			"k3",
			"Forward rate constant from complex to product + enz",
			&Enz::setKcat,
			&Enz::getKcat
		);

		static ElementValueFinfo< Enz, double > ratio(
			"ratio",
			"Ratio of k2/k3",
			&Enz::setRatio,
			&Enz::getRatio
		);

		static ElementValueFinfo< Enz, double > concK1(
			"concK1",
			"K1 expressed in concentration (1/millimolar.sec) units"
			"This parameter is subordinate to the Km. This means that"
			"when Km is changed, this changes. It also means that when"
			"k2 or k3 (aka kcat) are changed, we assume that Km remains"
			"fixed, and as a result concK1 must change. It is only when"
			"concK1 is assigned directly that we assume that the user knows"
			"what they are doing, and we adjust Km accordingly."
			"concK1 is also subordinate to the 'ratio' field, since"
			"setting the ratio reassigns k2."
			"Should you wish to assign the elementary rates concK1, k2, k3,"
		    "of an enzyme directly, always assign concK1 last.",
			&Enz::setConcK1,
			&Enz::getConcK1
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions: most are inherited from EnzBase.
		//////////////////////////////////////////////////////////////
		static DestFinfo setSolver( "setSolver",
			"Assigns solver to this MMEnz.",
			new EpFunc1< Enz, ObjId >( &Enz::setSolver ) );

		static DestFinfo cplxDest( "cplxDest",
				"Handles # of molecules of enz-sub complex",
				// Dummy
				new OpFunc1< Enz, double >( &Enz::sub ) );

		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* enzShared[] = {
			enzOut(), enzDest()
		};
		static Finfo* cplxShared[] = {
			cplxOut(), &cplxDest
		};

		static SharedFinfo enz( "enz",
			"Connects to enzyme pool",
			enzShared, sizeof( enzShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo cplx( "cplx",
			"Connects to enz-sub complex pool",
			cplxShared, sizeof( cplxShared ) / sizeof( const Finfo* )
		);

	static Finfo* enzFinfos[] = {
		&k1,	// Value
		&k2,	// Value
		&k3,	// Value
		&ratio,	// Value
		&concK1,	// Value
		&enz,				// SharedFinfo
		&cplx,				// SharedFinfo
		&setSolver,				// DestFinfo
	};

	static string doc[] =
	{
			"Name", "Enz",
			"Author", "Upi Bhalla",
			"Description:",
			"Enz handles mass-action enzymes in which there is an "
			" explicit pool for the enzyme-substrate complex. "
 			"It models the reaction: "
 			"E + S <===> E.S ----> E + P"
	};
	static Dinfo< Enz > dinfo;
	static Cinfo enzCinfo (
		"Enz",
		EnzBase::initCinfo(),
		enzFinfos,
		sizeof( enzFinfos ) / sizeof ( Finfo* ),
		&dinfo,
		doc,
		sizeof( doc )/sizeof( string )
	);

	return &enzCinfo;
}
//////////////////////////////////////////////////////////////

static const Cinfo* enzCinfo = Enz::initCinfo();

static const SrcFinfo2< double, double >* subOut =
	dynamic_cast< const SrcFinfo2< double, double >* >(
	enzCinfo->findFinfo( "subOut" ) );

//////////////////////////////////////////////////////////////
// Enz internal functions
//////////////////////////////////////////////////////////////

Enz::Enz( )
		:
				stoich_( 0 ),
				k2_( 4.0 ) // EnzBase sets kcat_ to 1.0
{ ; }

Enz::~Enz( )
{ ; }


//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

// v is in number units.
void Enz::setK1( const Eref& e, double v )
{
	double volScale =
		convertConcToNumRateUsingMesh( e, subOut, true );

	double concK1 = v * volScale;
	if ( concK1 < EPSILON )
		concK1 = EPSILON;
	Km_ = (k2_ + kcat_)/concK1;
	if (stoich_)
		stoich_->setEnzK1( e, concK1 );
}

// k1 is In number units.
double Enz::getK1( const Eref& e ) const
{
	double volScale =
		convertConcToNumRateUsingMesh( e, subOut, true );

	return (k2_ + kcat_) / (volScale * Km_);
}

void Enz::setK2( const Eref& e, double v )
{
	if ( v < EPSILON )
		v = EPSILON;
	k2_ = v;
	if ( stoich_)
		stoich_->setEnzK2( e, v );
}

double Enz::getK2( const Eref& e ) const
{
	return k2_;
}

void Enz::vSetKcat( const Eref& e, double v )
{
	// Assumes kcat_ has been set by EnzBase::setKcat
	// Tries to preserve 'ratio'.
	double ratio = 4.0;
	if ( kcat_ > EPSILON ) {
		ratio = k2_/kcat_;
	}
	kcat_ = v;
	k2_ = v * ratio;
	double concK1 = v * (1.0 + ratio) / Km_;

	if (stoich_ ) {
		stoich_->setEnzK1( e, concK1 );
		stoich_->setEnzK3( e, kcat_ );
		stoich_->setEnzK2( e, k2_ );
	}
}

void Enz::vSetKm( const Eref& e, double v )
{
	Km_ = v;
	double concK1 = ( k2_ + kcat_ ) / v;
	if ( stoich_ )
		stoich_->setEnzK1( e, concK1 );
}

void Enz::setRatio( const Eref& e, double v )
{
	k2_ = v * kcat_;

	if ( stoich_ ) {
		stoich_->setEnzK2( e, k2_ );
		double k1 = ( k2_ + kcat_ ) / Km_;
		setConcK1( e, k1 );
	}
}

double Enz::getRatio( const Eref& e ) const
{
	return k2_ / kcat_;
}

void Enz::setConcK1( const Eref& e, double v )
{
	if ( v <= EPSILON ) // raise error
		return;
	Km_ = (k2_ + kcat_) / v;
	if ( stoich_ )
		stoich_->setEnzK1( e, v );
}

double Enz::getConcK1( const Eref& e ) const
{
	return (k2_ + kcat_)/Km_;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

// static func
void Enz::setSolver( const Eref& e, ObjId stoich )
{
	static const Finfo* subFinfo = Cinfo::find("Enz")->findFinfo( "subOut");
	static const Finfo* prdFinfo = Cinfo::find("Enz")->findFinfo( "prdOut");
	static const Finfo* enzFinfo = Cinfo::find("Enz")->findFinfo( "enzOut");
	static const Finfo* cplxFinfo= Cinfo::find("Enz")->findFinfo("cplxOut");

	assert( subFinfo );
	assert( prdFinfo );
	assert( enzFinfo );
	assert( cplxFinfo );
	vector< Id > enzMols;
	vector< Id > cplxMols;
	bool isOK = true;
	unsigned int numReactants;
	numReactants = e.element()->getNeighbors( enzMols, enzFinfo );
	bool hasEnz = ( numReactants == 1 );
	vector< Id > subs;
	numReactants = e.element()->getNeighbors( subs, subFinfo );
	bool hasSubs = ( numReactants > 0 );
	numReactants = e.element()->getNeighbors( cplxMols, cplxFinfo );
	bool hasCplx = ( numReactants == 1 );
	vector< Id > prds;
	numReactants = e.element()->getNeighbors( prds, prdFinfo );
	bool hasPrds = ( numReactants > 0 );
	assert( stoich.element()->cinfo()->isA( "Stoich" ) );
	stoich_ = reinterpret_cast< Stoich* >( stoich.data() );

	if ( hasEnz && hasSubs && hasCplx && hasPrds ) {
		stoich_->installEnzyme( e.id(), enzMols[0], cplxMols[0], subs, prds );
	} else {
		stoich_->installDummyEnzyme( e.id(), Id() );
		string msg = "";
		if ( !hasEnz ) msg = msg + " enzyme";
		if ( !hasCplx ) msg = msg + " enzyme-substrate complex";
		if ( !hasSubs ) msg = msg + " substrates";
		if ( !hasPrds ) msg = msg + " products";
		cout << "Warning: Enz:setSolver: Dangling Enz '" <<
			e.objId().path() << "':\nMissing " << msg << endl;
	}
}
