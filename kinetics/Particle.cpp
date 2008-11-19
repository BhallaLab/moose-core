/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "moose.h"
#include <math.h>
#include "Molecule.h"
#include "RateTerm.h"
#include "SmoldynHub.h"
#include "Particle.h"

const Cinfo* initParticleCinfo()
{
	/*
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &Particle::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &Particle::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	static Finfo* reacShared[] =
	{
		new DestFinfo( "reac", Ftype2< double, double >::global(),
			RFCAST( &Particle::reacFunc ) ),
		new SrcFinfo( "n", Ftype1< double >::global() )
	};
	*/

	static Finfo* particleFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new LookupFinfo( "x", 
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &Particle::getX ), 
			RFCAST( &Particle::setX ) 
		),
		new LookupFinfo( "y", 
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &Particle::getY ), 
			RFCAST( &Particle::setY ) 
		),
		new LookupFinfo( "z", 
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &Particle::getZ ), 
			RFCAST( &Particle::setZ ) 
		),
		new ValueFinfo( "xVector", 
			ValueFtype1< vector< double > >::global(),
			GFCAST( &Particle::getXvector ), 
			RFCAST( &Particle::setXvector )
		),
		new ValueFinfo( "yVector", 
			ValueFtype1< vector< double > >::global(),
			GFCAST( &Particle::getYvector ), 
			RFCAST( &Particle::setYvector )
		),
		new ValueFinfo( "zVector", 
			ValueFtype1< vector< double > >::global(),
			GFCAST( &Particle::getZvector ), 
			RFCAST( &Particle::setZvector )
		),

		// Override the old version, because here we lookup from solver.
		new ValueFinfo( "n", 
			ValueFtype1< double >::global(),
			GFCAST( &Particle::getN ), 
			RFCAST( &Particle::setN )
		),
		new ValueFinfo( "conc", 
			ValueFtype1< double >::global(),
			GFCAST( &Particle::getConc ), 
			RFCAST( &Particle::setConc )
		),

		new ValueFinfo( "D", 
			ValueFtype1< double >::global(),
			GFCAST( &Molecule::getD ), 
			RFCAST( &Particle::setD )
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		// new SrcFinfo( "nSrc", Ftype1< double >::global() ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
	};

	// Schedule particles for the slower clock, stage 0.
	// static SchedInfo schedInfo[] = { { Molecule::process, 0, 0 } };
	
	static string doc[] =
	{
		"Name", "Particle",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Particle: Interface to Smoldyn pool of molecules.",
	};
	
	static Cinfo particleCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initMoleculeCinfo(),
		particleFinfos,
		sizeof( particleFinfos )/sizeof(Finfo *),
		ValueFtype1< Particle >::global()
	);

	return &particleCinfo;
}

static const Cinfo* particleCinfo = initParticleCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

Particle::Particle()
{
		;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

/*
void Particle::setVolumeScale( const Conn* c, double value )
{
	static_cast< Particle* >( c->data() )->volumeScale_ = value;
}

double Particle::getVolumeScale( const Element* e )
{
	return static_cast< Particle* >( e->data() )->volumeScale_;
}
*/

void Particle::setPos( const Conn* c, double value, 
	unsigned int i, unsigned int dim )
{
	unsigned int molIndex;
	SmoldynHub* sh = SmoldynHub::getHubFromZombie(
		c->targetElement(), SmoldynHub::molSolveFinfo, molIndex );
	if ( sh ) {
		assert( molIndex < sh->numSpecies() );
		sh->setPos( molIndex, value, i, dim );
	}
}

void Particle::setX( const Conn* c, double value, const unsigned int& i )
{
	setPos( c, value, i, 0 );
}

void Particle::setY( const Conn* c, double value, const unsigned int& i )
{
	setPos( c, value, i, 1 );
}

void Particle::setZ( const Conn* c, double value, const unsigned int& i )
{
	setPos( c, value, i, 2 );
}

double Particle::getPos( const Element* e, unsigned int i, unsigned int dim)
{
	unsigned int molIndex;
	SmoldynHub* sh = SmoldynHub::getHubFromZombie( 
		e, SmoldynHub::molSolveFinfo, molIndex );
	if ( sh ) {
		assert( molIndex < sh->numSpecies() );
		return sh->getPos( molIndex, i, dim );
	}
	return 0.0;
}

double Particle::getX( const Element* e, const unsigned int& i )
{
	return getPos( e, i, 0 );
}

double Particle::getY( const Element* e, const unsigned int& i )
{
	return getPos( e, i, 1 );
}

double Particle::getZ( const Element* e, const unsigned int& i )
{
	return getPos( e, i, 2 );
}

void Particle::setN( const Conn* c, double value )
{
	unsigned int molIndex;
	SmoldynHub* sh = SmoldynHub::getHubFromZombie( 
		c->targetElement(), SmoldynHub::molSolveFinfo, molIndex );
	if ( sh ) {
		assert( molIndex < sh->numSpecies() );
		sh->setNparticles( molIndex, static_cast< unsigned int >( value ) );
	}
	Molecule::setN( c, value );
}

double Particle::getN( const Element* e )
{
	unsigned int molIndex;
	SmoldynHub* sh = SmoldynHub::getHubFromZombie( 
		e, SmoldynHub::molSolveFinfo, molIndex );
	if ( sh ) {
		assert( molIndex < sh->numSpecies() );
		double n = sh->getNparticles( molIndex );
		return n;
	}
	return 0.0;
}

/**
 * set and getConc really need to deal with the volume defined by the
 * surface and geometry of the local region, and may not be well defined
 * if there is a spatial gradient. For now, I simply use compartmental
 * estimates of volume by drawing on the base Molecule class for the
 * conversion.
 */
void Particle::setConc( const Conn* c, double value )
{
	Molecule::setConc( c, value );
	double n = Molecule::getN( c->targetElement() );
	setN( c, n );
}

double Particle::getConc( const Element* e )
{
	double n = getN( e );
	// Doing this so the internal fields of the Molecule are updated.
	static_cast< Particle* >( e->data() )->localSetConc( n );
	return Molecule::getConc( e );
}

void Particle::setPosVector( const Conn* c, const vector< double >& value, 
	unsigned int dim )
{
	unsigned int molIndex;
	SmoldynHub* sh = SmoldynHub::getHubFromZombie( 
		c->targetElement(), SmoldynHub::molSolveFinfo, molIndex );
	if ( sh ) {
		assert( molIndex < sh->numSpecies() );
		sh->setPosVector( molIndex, value, dim );
	}
}

void Particle::setXvector( const Conn* c, vector< double > value )
{
	setPosVector( c, value, 0 );
}

void Particle::setYvector( const Conn* c, vector< double > value )
{
	setPosVector( c, value, 1 );
}

void Particle::setZvector( const Conn* c, vector< double > value )
{
	setPosVector( c, value, 2 );
}

vector< double > Particle::getPosVector( const Element* e, unsigned int dim)
{
	vector< double > ret;
	unsigned int molIndex;
	SmoldynHub* sh = SmoldynHub::getHubFromZombie( 
		e, SmoldynHub::molSolveFinfo, molIndex );
	if ( sh ) {
		assert( molIndex < sh->numSpecies() );
		sh->getPosVector( molIndex, ret, dim );
	}
	return ret;
}

vector< double > Particle::getXvector( const Element* e )
{
	return getPosVector( e, 0 );
}

vector< double > Particle::getYvector( const Element* e )
{
	return getPosVector( e, 1 );
}

vector< double > Particle::getZvector( const Element* e )
{
	return getPosVector( e, 2 );
}

void Particle::setD( const Conn* c, double value )
{
	Molecule::setD( c, value );
	unsigned int molIndex;
	SmoldynHub* sh = SmoldynHub::getHubFromZombie( 
		c->targetElement(), SmoldynHub::molSolveFinfo, molIndex );
	if ( sh ) {
		assert( molIndex < sh->numSpecies() );
		sh->setD( molIndex, value );
	}
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////
