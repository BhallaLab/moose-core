/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "moose.h"
#include <math.h>
#include "Molecule.h"

const double Molecule::EPSILON = 1.0e-15;

const Cinfo* initMoleculeCinfo()
{
	static TypeFuncPair processTypes[] =
	{
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &Molecule::processFunc ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &Molecule::reinitFunc ) ),
	};
	static TypeFuncPair reacTypes[] =
	{
		TypeFuncPair( Ftype2< double, double >::global(),
			RFCAST( &Molecule::reacFunc ) ),
		TypeFuncPair( Ftype1< double >::global(), 0 )
	};

	static Finfo* moleculeFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "nInit", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getNinit ), 
			RFCAST( &Molecule::setNinit ) 
		),
		new ValueFinfo( "volumeScale", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getVolumeScale ), 
			RFCAST( &Molecule::setVolumeScale )
		),
		new ValueFinfo( "n", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getN ), 
			RFCAST( &Molecule::setN )
		),
		new ValueFinfo( "mode", 
			ValueFtype1< int >::global(),
			GFCAST( &Molecule::getMode ), 
			RFCAST( &Molecule::setMode )
		),
		new ValueFinfo( "slave_enable", 
			ValueFtype1< int >::global(),
			GFCAST( &Molecule::getMode ), 
			RFCAST( &Molecule::setMode )
		),
		new ValueFinfo( "conc", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getConc ), 
			RFCAST( &Molecule::setConc )
		),
		new ValueFinfo( "concInit", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getConcInit ), 
			RFCAST( &Molecule::setConcInit )
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "nSrc", Ftype1< double >::global() ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	
		/**
		 * This is a backward compat feature to handle
		 * one-ended input from enzymes, but using the same reacFunc
		 * as the shared message for reac.
		 */
		new DestFinfo( "prd",
			Ftype2< double, double >::global(),
			RFCAST( &Molecule::reacFunc )
		),
	
		new DestFinfo( "sumTotal",
			Ftype1< double >::global(),
			RFCAST( &Molecule::sumTotalFunc )
		),
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "process", processTypes, 2 ),
		new SharedFinfo( "reac", reacTypes, 2 ),
	};

	static Cinfo moleculeCinfo(
		"Molecule",
		"Upinder S. Bhalla, 2007, NCBS",
		"Molecule: Pool of molecules.",
		initNeutralCinfo(),
		moleculeFinfos,
		sizeof( moleculeFinfos )/sizeof(Finfo *),
		ValueFtype1< Molecule >::global()
	);

	return &moleculeCinfo;
}

static const Cinfo* moleculeCinfo = initMoleculeCinfo();

static const unsigned int reacSlot =
	initMoleculeCinfo()->getSlotIndex( "reac" );
static const unsigned int nSlot =
	initMoleculeCinfo()->getSlotIndex( "nSrc" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

Molecule::Molecule()
	: nInit_( 0.0 ), 
	volumeScale_( 1.0 ),
	n_( 0.0 ),
	mode_( 0 )
{
		;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void Molecule::setNinit( const Conn& c, double value )
{
	static_cast< Molecule* >( c.data() )->nInit_ = value;
}

double Molecule::getNinit( const Element* e )
{
	return static_cast< Molecule* >( e->data() )->nInit_;
}

void Molecule::setVolumeScale( const Conn& c, double value )
{
	static_cast< Molecule* >( c.data() )->volumeScale_ = value;
}

double Molecule::getVolumeScale( const Element* e )
{
	return static_cast< Molecule* >( e->data() )->volumeScale_;
}

void Molecule::setN( const Conn& c, double value )
{
	static_cast< Molecule* >( c.data() )->n_ = value;
}

double Molecule::getN( const Element* e )
{
	return static_cast< Molecule* >( e->data() )->n_;
}

void Molecule::setMode( const Conn& c, int value )
{
	static_cast< Molecule* >( c.data() )->mode_ = value;
}

double Molecule::getMode( const Element* e )
{
	return static_cast< Molecule* >( e->data() )->mode_;
}

double Molecule::localGetConc() const
{
			if ( volumeScale_ > 0.0 )
				return n_ / volumeScale_ ;
			else
				return n_;
}
double Molecule::getConc( const Element* e )
{
	return static_cast< Molecule* >( e->data() )->localGetConc();
}

void Molecule::localSetConc( double value ) {
			if ( volumeScale_ > 0.0 )
				n_ = value * volumeScale_ ;
			else
				n_ = value;
}
void Molecule::setConc( const Conn& c, double value )
{
	static_cast< Molecule* >( c.data() )->localSetConc( value );
}

double Molecule::localGetConcInit() const
{
			if ( volumeScale_ > 0.0 )
				return nInit_ / volumeScale_ ;
			else
				return nInit_;
}
double Molecule::getConcInit( const Element* e )
{
	return static_cast< Molecule* >( e->data() )->localGetConcInit();
}

void Molecule::localSetConcInit( double value ) {
			if ( volumeScale_ > 0.0 )
				nInit_ = value * volumeScale_ ;
			else
				nInit_ = value;
}
void Molecule::setConcInit( const Conn& c, double value )
{
	static_cast< Molecule* >( c.data() )->localSetConcInit( value );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////


void Molecule::reacFunc( const Conn& c, double A, double B )
{
	static_cast< Molecule* >( c.data() )->A_ += A;
	static_cast< Molecule* >( c.data() )->B_ += B;
}

void Molecule::sumTotalFunc( const Conn& c, double n )
{
	static_cast< Molecule* >( c.data() )->total_ += n;
}

void Molecule::sumProcessFuncLocal( )
{
		n_ = total_;
		total_ = 0.0;
}
void Molecule::sumProcessFunc( const Conn& c, ProcInfo info )
{
	static_cast< Molecule* >( c.data() )->sumProcessFuncLocal();
}

void Molecule::reinitFunc( const Conn& c, ProcInfo info )
{
	static_cast< Molecule* >( c.data() )->reinitFuncLocal( 
					c.targetElement() );
}
void Molecule::reinitFuncLocal( Element* e )
{
	static const Finfo* sumTotFinfo = 
			Cinfo::find( "Molecule" )->findFinfo( "sumTotal" );

	A_ = B_ = total_ = 0.0;
	n_ = nInit_;
	if ( mode_ == 0 && sumTotFinfo->numIncoming( e ) > 0 )
		mode_ = 1;
	else if ( mode_ == 1 && sumTotFinfo->numIncoming( e ) == 0 )
		mode_ = 0;
	send1< double >( e, reacSlot, n_ );
	send1< double >( e, nSlot, n_ );
}

void Molecule::processFunc( const Conn& c, ProcInfo info )
{
	Element* e = c.targetElement();
	static_cast< Molecule* >( e->data() )->processFuncLocal( e, info );
}
void Molecule::processFuncLocal( Element* e, ProcInfo info )
{
			if ( mode_ == 0 ) {
				if ( n_ > EPSILON && B_ > EPSILON ) {
					double C = exp( -B_ * info->dt_ / n_ );
					n_ *= C + ( A_ / B_ ) * ( 1.0 - C );
				} else {
					n_ += ( A_ - B_ ) * info->dt_;
				}
				A_ = B_ = 0.0;
			} else if ( mode_ == 1 ) {
				n_ = total_;
				total_ = 0.0;
			} else if ( mode_ == 2 ) {
				n_ = total_ * volumeScale_;
				total_ = 0.0;
			} else { 
				n_ = nInit_;
			}
			send1< double >( e, reacSlot, n_ );
			send1< double >( e, nSlot, n_ );
}
