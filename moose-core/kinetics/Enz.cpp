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
#include "Enz.h"

#define EPSILON 1e-15
const Cinfo* Enz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions: all inherited
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////
	static Dinfo< Enz > dinfo;
	static Cinfo enzCinfo (
		"Enz",
		CplxEnzBase::initCinfo(),
		0,
		0,
		&dinfo
	);

	return &enzCinfo;
}
//////////////////////////////////////////////////////////////

static const Cinfo* enzCinfo = Enz::initCinfo();
static const SrcFinfo2< double, double >* subOut =
	dynamic_cast< const SrcFinfo2< double, double >* >(
	enzCinfo->findFinfo( "subOut" ) );

static const SrcFinfo2< double, double >* prdOut =
	dynamic_cast< const SrcFinfo2< double, double >* >(
	enzCinfo->findFinfo( "prdOut" ) );

static const SrcFinfo2< double, double >* enzOut =
	dynamic_cast< const SrcFinfo2< double, double >* >(
	enzCinfo->findFinfo( "enzOut" ) );

static const SrcFinfo2< double, double >* cplxOut =
	dynamic_cast< const SrcFinfo2< double, double >* >(
	enzCinfo->findFinfo( "cplxOut" ) );


//////////////////////////////////////////////////////////////
// Enz internal functions
//////////////////////////////////////////////////////////////
Enz::Enz( )
	: k1_( 0.1 ), k2_( 0.4 ), k3_( 0.1 )
{
	;
}

Enz::~Enz()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Enz::vSub( double n )
{
	r1_ *= n;
}

void Enz::vEnz( double n )
{
	r1_ *= n;
}

void Enz::vCplx( double n )
{
	r2_ = k2_ * n;
	r3_ = k3_ * n;
}

void Enz::vProcess( const Eref& e, ProcPtr p )
{
	subOut->send( e, r2_, r1_ );
	prdOut->send( e, r3_, 0 );
	enzOut->send( e, r3_ + r2_, r1_ );
	cplxOut->send( e, r1_, r3_ + r2_ );

	// cout << "	proc: " << r1_ << ", " << r2_ << ", " << r3_ << endl;
	
	r1_ = k1_;
}

void Enz::vReinit( const Eref& e, ProcPtr p )
{
	r1_ = k1_;
}

void Enz::vRemesh( const Eref& e )
{
	setKm( e, Km_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Enz::vSetK1( const Eref& e, double v )
{
	r1_ = k1_ = v;
	double volScale = 
		convertConcToNumRateUsingMesh( e, subOut, 1 );
	Km_ = ( k2_ + k3_ ) / ( k1_ * volScale );
}

double Enz::vGetK1( const Eref& e ) const
{
	return k1_;
}

void Enz::vSetK2( const Eref& e, double v )
{
	k2_ = v; // Assume this overrides the default ratio.
	vSetKm( e, Km_ ); // Update k1_ here as well.
}

double Enz::vGetK2( const Eref& e ) const
{
	return k2_;
}

void Enz::vSetKcat( const Eref& e, double v )
{
	double ratio = k2_ / k3_;
	k3_ = v;
	k2_ = v * ratio;
	vSetKm( e, Km_ ); // Update k1_ here as well.
}

double Enz::vGetKcat( const Eref& e ) const
{
	return k3_;
}

//////////////////////////////////////////////////////////////
// Scaled field terms.
// We assume that when we set these, the k1, k2 and k3 vary as needed
// to preserve the other field terms. So when we set Km, then kcat
// and ratio remain unchanged.
//////////////////////////////////////////////////////////////

void Enz::vSetKm( const Eref& e, double v )
{
	Km_ = v;
	double volScale = 
		convertConcToNumRateUsingMesh( e, subOut, 1 );
	k1_ = ( k2_ + k3_ ) / ( v * volScale );
}

double Enz::vGetKm( const Eref& e ) const
{
	return Km_;
}

void Enz::vSetNumKm( const Eref& e, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, subOut, 1 );
	k1_ = ( k2_ + k3_ ) / v;
	Km_ = v / volScale;
}

double Enz::vGetNumKm( const Eref& e ) const
{
	double volScale = convertConcToNumRateUsingMesh( e, subOut, 1 );
	return Km_ * volScale;
}

void Enz::vSetRatio( const Eref& e, double v )
{
	k2_ = v * k3_;
	double volScale = 
		convertConcToNumRateUsingMesh( e, subOut, 1 );

	k1_ = ( k2_ + k3_ ) / ( Km_ * volScale );
}

double Enz::vGetRatio( const Eref& e ) const
{
	return k2_ / k3_;
}

void Enz::vSetConcK1( const Eref& e, double v )
{
	if ( v < EPSILON ) {
		cout << "Enz::vSetConcK1: Warning: value " << v << " too small\n";
		return;
	}
	double volScale = convertConcToNumRateUsingMesh( e, subOut, 1 );
	r1_ = k1_ = v * volScale;
	Km_ = ( k2_ + k3_ ) / ( v );
}

double Enz::vGetConcK1( const Eref& e ) const
{
	if ( Km_ < EPSILON ) {
		cout << "Enz::vGetConcK1: Warning: Km_ too small\n";
		return 1.0;
	}
	return ( k2_ + k3_ ) / Km_;
}

