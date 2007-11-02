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
#include "Smoldyn/source/smollib.h"
#include "SmoldynHub.h"
#include "ThisFinfo.h"
#include "SolveFinfo.h"
#include "../element/Wildcard.h"


const Cinfo* initSmoldynHubCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &SmoldynHub::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &SmoldynHub::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );


	static Finfo* smoldynHubFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "nSpecies", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SmoldynHub::getNspecies ), 
			&dummyFunc
		),
		new ValueFinfo( "path", 
			ValueFtype1< string >::global(),
			GFCAST( &SmoldynHub::getPath ),
			RFCAST( &SmoldynHub::setPath )
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	//	new SrcFinfo( "nSrc", Ftype1< double >::global() ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		process,
	};

	// Schedule smoldynHubs for the slower clock, stage 0.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static Cinfo smoldynHubCinfo(
		"SmoldynHub",
		"Upinder S. Bhalla, 2007, NCBS",
		"SmoldynHub: Interface object between Smoldyn (by Steven Andrews) and MOOSE.",
		initNeutralCinfo(),
		smoldynHubFinfos,
		sizeof( smoldynHubFinfos )/sizeof(Finfo *),
		ValueFtype1< SmoldynHub >::global(),
			schedInfo, 1
	);

	return &smoldynHubCinfo;
}

static const Cinfo* smoldynHubCinfo = initSmoldynHubCinfo();

const Finfo* SmoldynHub::particleFinfo = 
	initSmoldynHubCinfo()->findFinfo( "particleFinfo" );

/*
static const unsigned int reacSlot =
	initSmoldynHubCinfo()->getSlotIndex( "reac.n" );
static const unsigned int nSlot =
	initSmoldynHubCinfo()->getSlotIndex( "nSrc" );
*/

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

SmoldynHub::SmoldynHub()
	: simptr_( 0 ), path_( "" )
{
		;
}

///////////////////////////////////////////////////
// Hub utility function definitions
///////////////////////////////////////////////////

/**
 * Looks up the solver from the zombie element e. Returns the solver
 * element, or null on failure. 
 * It needs the originating Finfo on the solver that connects to the zombie,
 * as the srcFinfo.
 * Also passes back the index of the zombie element on this set of
 * messages. This is NOT the absolute Conn index.
 */
SmoldynHub* SmoldynHub::getHubFromZombie( 
	const Element* e, const Finfo* srcFinfo, unsigned int& index )
{
	const SolveFinfo* f = dynamic_cast< const SolveFinfo* > (
			       	e->getThisFinfo() );
	if ( !f ) return 0;
	const Conn& c = f->getSolvedConn( e );
	unsigned int slot;
       	srcFinfo->getSlotIndex( srcFinfo->name(), slot );
	Element* hub = c.targetElement();
	index = hub->connSrcRelativeIndex( c, slot );
	return static_cast< SmoldynHub* >( hub->data() );
}

void SmoldynHub::setPos( unsigned int molIndex, double value, 
			unsigned int i, unsigned int dim )
{
}

double SmoldynHub::getPos( unsigned int molIndex, unsigned int i, 
			unsigned int dim )
{
	return 0.0;
}

void SmoldynHub::setPosVector( unsigned int molIndex, 
			const vector< double >& value, unsigned int dim )
{
}

void SmoldynHub::getPosVector( unsigned int molIndex,
			vector< double >& value, unsigned int dim )
{
}

void SmoldynHub::setNinit( unsigned int molIndex, unsigned int value )
{
}

unsigned int SmoldynHub::getNinit( unsigned int molIndex )
{
	return 0;
}

void SmoldynHub::setNmol( unsigned int molIndex, unsigned int value )
{
	;
}

unsigned int SmoldynHub::getNmol( unsigned int molIndex )
{
	return 0;
}

void SmoldynHub::setD( unsigned int molIndex, double value )
{
}

double SmoldynHub::getD( unsigned int molIndex )
{
	return 0.0;
}


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

unsigned int SmoldynHub::getNspecies( const Element* e )
{
	return static_cast< SmoldynHub* >( e->data() )->numSpecies();
}

unsigned int SmoldynHub::numSpecies() const
{
	return 0;
}

string SmoldynHub::getPath( const Element* e )
{
	return static_cast< const SmoldynHub* >( e->data() )->path_;
}

void SmoldynHub::setPath( const Conn& c, string value )
{
	Element* e = c.targetElement();
	static_cast< SmoldynHub* >( e->data() )->localSetPath( e, value );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void SmoldynHub::reinitFunc( const Conn& c, ProcInfo info )
{
	static_cast< SmoldynHub* >( c.data() )->reinitFuncLocal( 
					c.targetElement() );
}

void SmoldynHub::reinitFuncLocal( Element* e )
{
	// do stuff here.
}

void SmoldynHub::processFunc( const Conn& c, ProcInfo info )
{
	Element* e = c.targetElement();
	static_cast< SmoldynHub* >( e->data() )->processFuncLocal( e, info );
}

void SmoldynHub::processFuncLocal( Element* e, ProcInfo info )
{
	// do stuff here
}

///////////////////////////////////////////////////
// This is where the business happens
///////////////////////////////////////////////////

void SmoldynHub::localSetPath( Element* stoich, const string& value )
{
	path_ = value;
	vector< Element* > ret;
	wildcardFind( path_, ret );
	if ( ret.size() > 0 ) {
		;
	}
	cout << "found " << ret.size() << " elements\n";
}
