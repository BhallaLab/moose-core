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
#include "lookupVolumeFromMesh.h"
#include "EnzBase.h"
#include "CplxEnzBase.h"

static SrcFinfo2< double, double > *toEnz() {
	static SrcFinfo2< double, double > toEnz( 
			"toEnz", 
			"Sends out increment of molecules on product each timestep"
			);
	return &toEnz;
}

static SrcFinfo2< double, double > *toCplx() {
	static SrcFinfo2< double, double > toCplx( 
			"toCplx", 
			"Sends out increment of molecules on product each timestep"
			);
	return &toCplx;
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

/*
static DestFinfo* enzDest = 
	dynamic_cast< DestFinfo* >( 
			const_cast< Finfo* >(
			EnzBase::initCinfo()->findFinfo( "enzDest" ) ) );
			*/

const Cinfo* CplxEnzBase::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< CplxEnzBase, double > k1(
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
			&CplxEnzBase::setK1,
			&CplxEnzBase::getK1
		);

		static ElementValueFinfo< CplxEnzBase, double > k2(
			"k2",
			"Reverse reaction from complex to enz + sub",
			&CplxEnzBase::setK2,
			&CplxEnzBase::getK2
		);

		static ElementValueFinfo< CplxEnzBase, double > k3(
			"k3",
			"Forward rate constant from complex to product + enz",
			&CplxEnzBase::setKcat,
			&CplxEnzBase::getKcat
		);

		static ElementValueFinfo< CplxEnzBase, double > ratio(
			"ratio",
			"Ratio of k2/k3",
			&CplxEnzBase::setRatio,
			&CplxEnzBase::getRatio
		);

		static ElementValueFinfo< CplxEnzBase, double > concK1(
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
			&CplxEnzBase::setConcK1,
			&CplxEnzBase::getConcK1
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions: most are inherited from EnzBase.
		//////////////////////////////////////////////////////////////
		static DestFinfo cplxDest( "cplxDest",
				"Handles # of molecules of enz-sub complex",
				new OpFunc1< CplxEnzBase, double >( &CplxEnzBase::cplx ) );

		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* enzShared[] = {
			toEnz(), enzDest()
		};
		static Finfo* cplxShared[] = {
			toCplx(), &cplxDest
		};

		static SharedFinfo enz( "enz",
			"Connects to enzyme pool",
			enzShared, sizeof( enzShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo cplx( "cplx",
			"Connects to enz-sub complex pool",
			cplxShared, sizeof( cplxShared ) / sizeof( const Finfo* )
		);

	static Finfo* cplxEnzFinfos[] = {
		&k1,	// Value
		&k2,	// Value
		&k3,	// Value
		&ratio,	// Value
		&concK1,	// Value
		&enz,				// SharedFinfo
		&cplx,				// SharedFinfo
	};

	static string doc[] = 
	{
			"Name", "CplxEnzBase",
			"Author", "Upi Bhalla",
			"Description:", 
			"Base class for mass-action enzymes in which there is an "
			" explicit pool for the enzyme-substrate complex. "
 			"It models the reaction: "
 			"E + S <===> E.S ----> E + P"
	};

	static ZeroSizeDinfo< int > dinfo;
	static Cinfo cplxEnzCinfo (
		"CplxEnzBase",
		EnzBase::initCinfo(),
		cplxEnzFinfos,
		sizeof( cplxEnzFinfos ) / sizeof ( Finfo* ),
		&dinfo,
		doc,
		sizeof( doc )/sizeof( string )
	);

	return &cplxEnzCinfo;
}

 static const Cinfo* cplxEnzCinfo = CplxEnzBase::initCinfo();

//////////////////////////////////////////////////////////////
// Enz internal functions
//////////////////////////////////////////////////////////////
CplxEnzBase::CplxEnzBase( )
{ ; }
CplxEnzBase::~CplxEnzBase( )
{ ; }

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void CplxEnzBase::cplx( double n )
{
	vCplx( n );
}

void CplxEnzBase::vCplx( double n )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void CplxEnzBase::setK1( const Eref& e, double v )
{
	vSetK1( e, v );
}

double CplxEnzBase::getK1( const Eref& e ) const
{
	return vGetK1( e );
}

void CplxEnzBase::setK2( const Eref& e, double v )
{
	vSetK2( e, v );
}

double CplxEnzBase::getK2( const Eref& e ) const
{
	return vGetK2( e );
}

//////////////////////////////////////////////////////////////
// Scaled field terms.
// We assume that when we set these, the k1, k2 and k3 vary as needed
// to preserve the other field terms. So when we set Km, then kcat
// and ratio remain unchanged.
//////////////////////////////////////////////////////////////

void CplxEnzBase::setRatio( const Eref& e, double v )
{
	vSetRatio( e, v );
}

double CplxEnzBase::getRatio( const Eref& e ) const
{
	return vGetRatio( e );
}

void CplxEnzBase::setConcK1( const Eref& e, double v )
{
	vSetConcK1( e, v );
}

double CplxEnzBase::getConcK1( const Eref& e ) const
{
	return vGetConcK1( e );
}

////////////////////////////////////////////////////////////////////////
// Zombie conversion routine.
////////////////////////////////////////////////////////////////////////


/**
 * This function helps the conversion between CplxEnz subclasses, these
 * are the ones that have an explicit enzyme-substrate complex molecule.
 * Note that I use the ConcK1 terms because those are independent of the
 * volume decomposition. K2 and kcat (ie., k3) only have time units.
 */
void CplxEnzBase::zombify( Element* orig, const Cinfo* zClass, Id solver )
{
		/*
	if ( orig->cinfo() == zClass )
		return;
	DataHandler* origHandler = orig->dataHandler();
	DataHandler* dh = origHandler->copyUsingNewDinfo( zClass->dinfo() );
	Element temp( orig->id(), zClass, dh );
	Eref zombier( &temp, 0 );

	CplxEnzBase* z = reinterpret_cast< CplxEnzBase* >( zombier.data() );
	Eref oer( orig, 0 );

	const CplxEnzBase* m = reinterpret_cast< CplxEnzBase* >( oer.data() );
	z->setSolver( solver, orig->id() ); // call virtual func to assign solver info.
	// May need to extend to entire array.
	z->vSetConcK1( oer, 0, m->vGetConcK1( oer, 0 ) );
	// z->vSetConcK1( zombier, 0, m->vGetConcK1( oer, 0 ) );
	z->vSetK2( zombier, 0, m->vGetK2( oer, 0 ) );
	z->vSetKcat( zombier, 0, m->vGetKcat( oer, 0 ) );
	orig->zombieSwap( zClass, dh );
	delete origHandler;
	*/
}

