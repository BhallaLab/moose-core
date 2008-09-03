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
#include "setgetLookup.h"
#include "../element/Neutral.h"
#include "SigNeur.h"


const Cinfo* initSigNeurCinfo()
{
	static Finfo* sigNeurFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "cell", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getCell ), 
			RFCAST( &SigNeur::setCell ) 
		),
		new ValueFinfo( "spine", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getSpine ), 
			RFCAST( &SigNeur::setSpine )
		),
		new ValueFinfo( "dend", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getDend ), 
			RFCAST( &SigNeur::setDend )
		),
		new ValueFinfo( "soma", 
			ValueFtype1< Id >::global(),
			GFCAST( &SigNeur::getSoma ), 
			RFCAST( &SigNeur::setSoma )
		),

		new ValueFinfo( "cellMethod", 
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getCellMethod ), 
			RFCAST( &SigNeur::setCellMethod )
		),
		new ValueFinfo( "spineMethod", 
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getSpineMethod ), 
			RFCAST( &SigNeur::setSpineMethod )
		),
		new ValueFinfo( "dendMethod", 
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getDendMethod ), 
			RFCAST( &SigNeur::setDendMethod )
		),
		new ValueFinfo( "somaMethod", 
			ValueFtype1< string >::global(),
			GFCAST( &SigNeur::getSomaMethod ), 
			RFCAST( &SigNeur::setSomaMethod )
		),

		new ValueFinfo( "Dscale", 
			ValueFtype1< double >::global(),
			GFCAST( &SigNeur::getDscale ), 
			RFCAST( &SigNeur::setDscale )
		),
		new ValueFinfo( "parallelMode", 
			ValueFtype1< int >::global(),
			GFCAST( &SigNeur::getParallelMode ), 
			RFCAST( &SigNeur::setParallelMode )
		),
		new ValueFinfo( "updateStep", // Time between sig<->neuro updates
			ValueFtype1< double >::global(),
			GFCAST( &SigNeur::getUpdateStep ), 
			RFCAST( &SigNeur::setUpdateStep )
		),
		new LookupFinfo( "channelMap", // Mapping from channels to sig mols
			LookupFtype< string, string >::global(),
			GFCAST( &SigNeur::getChannelMap ), 
			RFCAST( &SigNeur::setChannelMap )
		),
		new LookupFinfo( "calciumMap",  // Mapping from calcium to sig.
			LookupFtype< string, string >::global(),
			GFCAST( &SigNeur::getCalciumMap ), 
			RFCAST( &SigNeur::setCalciumMap )
		),
		new ValueFinfo( "calciumScale",
			ValueFtype1< double >::global(),
			GFCAST( &SigNeur::getCalciumScale ), 
			RFCAST( &SigNeur::setCalciumScale )
		),
	// Would be nice to have a way to include synaptic input into
	// the mGluR input.
	
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "build", Ftype0::global(),
			RFCAST( &SigNeur::build )
		),
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
	};

	// Schedule it to tick 1 stage 0
	// static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static Cinfo sigNeurCinfo(
		"SigNeur",
		"Upinder S. Bhalla, 2007, NCBS",
		"SigNeur: Multiscale simulation setup object for doing combined electrophysiological and signaling models of neurons. Takes the geometry from the neuronal model and sets up diffusion between signaling models to fit in this geometry. Arranges interfaces between channel conductances and molecular species representing channels. Also interfaces calcium conc in the two kinds of model.",
		initNeutralCinfo(),
		sigNeurFinfos,
		sizeof( sigNeurFinfos )/sizeof(Finfo *),
		ValueFtype1< SigNeur >::global()
	);

	// methodMap.size(); // dummy function to keep compiler happy.

	return &sigNeurCinfo;
}

static const Cinfo* sigNeurCinfo = initSigNeurCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

SigNeur::SigNeur()
	: 	cellMethod_( "hsolve" ), 
		spineMethod_( "rk5" ), 
		dendMethod_( "rk5" ), 
		somaMethod_( "rk5" ), 
		Dscale_( 1.0 ),
		parallelMode_( 0 ),
		updateStep_( 1.0 ),
		calciumScale_( 1.0 )
{
		;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void SigNeur::setCell( const Conn* c, Id value )
{
	static_cast< SigNeur* >( c->data() )->cell_ = value;
}

Id SigNeur::getCell( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->cell_;
}

void SigNeur::setSpine( const Conn* c, Id value )
{
	static_cast< SigNeur* >( c->data() )->spine_ = value;
}

Id SigNeur::getSpine( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->spine_;
}

void SigNeur::setDend( const Conn* c, Id value )
{
	static_cast< SigNeur* >( c->data() )->dend_ = value;
}

Id SigNeur::getDend( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->dend_;
}

void SigNeur::setSoma( const Conn* c, Id value )
{
	static_cast< SigNeur* >( c->data() )->soma_ = value;
}

Id SigNeur::getSoma( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->soma_;
}

void SigNeur::setCellMethod( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->cellMethod_ = value;
}

string SigNeur::getCellMethod( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->cellMethod_;
}

void SigNeur::setSpineMethod( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->spineMethod_ = value;
}

string SigNeur::getSpineMethod( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->spineMethod_;
}

void SigNeur::setDendMethod( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->dendMethod_ = value;
}

string SigNeur::getDendMethod( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->dendMethod_;
}

void SigNeur::setSomaMethod( const Conn* c, string value )
{
	static_cast< SigNeur* >( c->data() )->somaMethod_ = value;
}

string SigNeur::getSomaMethod( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->somaMethod_;
}

void SigNeur::setDscale( const Conn* c, double value )
{
	static_cast< SigNeur* >( c->data() )->Dscale_ = value;
}

double SigNeur::getDscale( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->Dscale_;
}

void SigNeur::setParallelMode( const Conn* c, int value )
{
	static_cast< SigNeur* >( c->data() )->parallelMode_ = value;
}

int SigNeur::getParallelMode( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->parallelMode_;
}

void SigNeur::setUpdateStep( const Conn* c, double value )
{
	static_cast< SigNeur* >( c->data() )->updateStep_ = value;
}

double SigNeur::getUpdateStep( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->updateStep_;
}

void SigNeur::setCalciumMap( const Conn* c, string val, const string& i )
{
	static_cast< SigNeur* >( c->data() )->calciumMap_[ i ] = val;
}

string SigNeur::getCalciumMap( Eref e, const string& i )
{
	SigNeur* sn = static_cast< SigNeur* >( e.data() );
	map< string, string >::iterator j = sn->calciumMap_.find( i );
	if ( j != sn->calciumMap_.end() )
		return j->second;
	return "";
}

void SigNeur::setCalciumScale( const Conn* c, double value )
{
	static_cast< SigNeur* >( c->data() )->calciumScale_ = value;
}

double SigNeur::getCalciumScale( Eref e )
{
	return static_cast< SigNeur* >( e.data() )->calciumScale_;
}

void SigNeur::setChannelMap( const Conn* c, string val, const string& i )
{
	static_cast< SigNeur* >( c->data() )->channelMap_[ i ] = val;
}

string SigNeur::getChannelMap( Eref e, const string& i )
{
	SigNeur* sn = static_cast< SigNeur* >( e.data() );
	map< string, string >::iterator j = sn->channelMap_.find( i );
	if ( j != sn->channelMap_.end() )
		return j->second;
	return "";
}

void SigNeur::build( const Conn* c )
{
	static_cast< SigNeur* >( c->data() )->innerBuild( c );
}

//////////////////////////////////////////////////////////////////
// Here we set up some of the messier inner functions.
//////////////////////////////////////////////////////////////////

void SigNeur::innerBuild( const Conn* c )
{
	if ( !( spine_.good() || dend_.good() || soma_.good() ) ) {
		cout << "SigNeur::build: " << c->target().name() << 
			" : Warning: Unable to find any signaling models to use\n";
		return;
	}
	if ( !traverseCell() ) {
		cout << "SigNeur::build: " << c->target().name() << 
		cout << " : Warning: Unable to traverse cell\n";
		return;
	}
}

bool SigNeur::traverseCell()
{
	return 0;
}
