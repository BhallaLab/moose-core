/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "element/Neutral.h"
#include "biophysics/Compartment.h"
#include "biophysics/HHChannel.h"
#include "biophysics/CaConc.h"
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"
#include "ThisFinfo.h"
#include "SolveFinfo.h"
#include "HSolveHub.h"

// Defined below.
const Cinfo* initHSolveHubCinfoInner();
Finfo* initCompartmentZombieFinfo();
Finfo* initHHChannelZombieFinfo();
Finfo* initCaConcZombieFinfo();

/**
 * This function is required to initialize the replacement Finfos
 * (i.e., SolveFinfo:s standing in for ThisFinfo:s for zombified objects).
 * We will need to ensure that this function is called explicitly during
 * the initialization of MOOSE (currently done in maindir/initCinfos.cpp).
 */
const Cinfo* initHSolveHubCinfo()
{
	static const Cinfo* HSolveHubCinfo = initHSolveHubCinfoInner();
	
	static Finfo* f1 = initCompartmentZombieFinfo();
	static Finfo* f2 = initHHChannelZombieFinfo();
	static Finfo* f3 = initCaConcZombieFinfo();
	
	// Only to avoid compiler warnings.
	f1 = 0;
	f2 = 0;
	f3 = 0;
	
	return HSolveHubCinfo;
}

const Cinfo* initHSolveHubCinfoInner()
{
	static Finfo* zombieShared[] =
	{
		new SrcFinfo( "process", Ftype1< ProcInfo >::global() ),
		new SrcFinfo( "reinit", Ftype1< ProcInfo >::global() ),
	};
	
	static Finfo* compartmentChannelShared[] =
	{
		new DestFinfo(
			"channel",
			Ftype2< double, double >::global(),
			RFCAST( &HSolveHub::compartmentChannelFunc ) ),
		new SrcFinfo(
			"Vm",
			Ftype1< double >::global() ),
	};
	
	static Finfo* HSolveHubFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "integ-hub",
			Ftype1< HSolveActive* >::global(),
			RFCAST( &HSolveHub::hubFunc ),
			"In this message, the hub receives a handle (a pointer) to the "
			"solver. This is used by the hub to access the fields of the solver." ),
		new DestFinfo( "destroy", Ftype0::global(),
			&HSolveHub::destroy ),
		new DestFinfo( "child", Ftype1< int >::global(),
			RFCAST( &HSolveHub::childFunc ),
			"override the Neutral::childFunc here, so that when this is deleted "
			"all the zombies are reanimated." ),
		new DestFinfo( "compartmentInjectMsg", Ftype1< double >::global(),
			RFCAST( &HSolveHub::compartmentInjectMsgFunc ) ),
		new SharedFinfo( "compartmentChannel", compartmentChannelShared,
			sizeof( compartmentChannelShared ) / sizeof( Finfo* ),
			"This message allows communication between external channels and "
			"the Hines' solver." ),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "compartmentSolve", zombieShared, 
			sizeof( zombieShared ) / sizeof( Finfo* ),
			"This is identical to the message sent from clock Ticks to objects. "
			"Here it is used to take over the Process message, usually only as "
			"a handle from the solver to the object." ),
		new SharedFinfo( "hhchannelSolve", zombieShared, 
			sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "caconcSolve", zombieShared, 
			sizeof( zombieShared ) / sizeof( Finfo* ) ),
	};
	
	static string doc[] =
	{
		"Name",
		"HSolveHub",
		
		"Author",
		"Niraj Dudani, 2007, NCBS",
		
		"Description",
		"HSolveHub: Ensures that field and message requests to solved objects "
		"are cleanly redirected to their respective HSolve object.",
	};
	
	static Cinfo HSolveHubCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),
		initNeutralCinfo(),
		HSolveHubFinfos,
		sizeof( HSolveHubFinfos ) / sizeof( Finfo* ),
		ValueFtype1< HSolveHub >::global()
	);
	
	return &HSolveHubCinfo;
}

static const Cinfo* HSolveHubCinfo = initHSolveHubCinfo();

static const Finfo* compartmentSolveFinfo = 
	initHSolveHubCinfo()->findFinfo( "compartmentSolve" );
static const Finfo* hhchannelSolveFinfo = 
	initHSolveHubCinfo()->findFinfo( "hhchannelSolve" );
static const Finfo* caconcSolveFinfo = 
	initHSolveHubCinfo()->findFinfo( "caconcSolve" );
static const Finfo* hubCompartmentInjectFinfo =
	initHSolveHubCinfo()->findFinfo( "compartmentInjectMsg" );
static const Finfo* hubCompartmentChannelFinfo =
	initHSolveHubCinfo()->findFinfo( "compartmentChannel" );

/*
 * Finfos from biophysical objects. Needed so that 'set' operations are done
 * on the solver as well as the objects. Also needed for redirecting any dest
 * messages on these finfos to the solver (e.g.: to the inject field).
 */
static const Finfo* compartmentInjectFinfo =
	initCompartmentCinfo()->findFinfo( "injectMsg" );
static const Finfo* compartmentInjectFieldFinfo =
	initCompartmentCinfo()->findFinfo( "inject" );
static const Finfo* compartmentChannelFinfo =
	initCompartmentCinfo()->findFinfo( "channel" );
static const Finfo* compartmentVmFinfo =
	initCompartmentCinfo()->findFinfo( "Vm" );
static const Finfo* channelGbarFinfo =
	initHHChannelCinfo()->findFinfo( "Gbar" );
static const Finfo* channelEkFinfo =
	initHHChannelCinfo()->findFinfo( "Ek" );
static const Finfo* channelGkFinfo =
	initHHChannelCinfo()->findFinfo( "Gk" );
static const Finfo* channelXFinfo =
	initHHChannelCinfo()->findFinfo( "X" );
static const Finfo* channelYFinfo =
	initHHChannelCinfo()->findFinfo( "Y" );
static const Finfo* channelZFinfo =
	initHHChannelCinfo()->findFinfo( "Z" );
static const Finfo* caConcCaFinfo =
	initCaConcCinfo()->findFinfo( "Ca" );
/////////////////////////////////////////////////////////////////////////
// Replacement fields for aspiring zombies
/////////////////////////////////////////////////////////////////////////

Finfo* initCompartmentZombieFinfo()
{
	static Finfo* compartmentFields[] =
	{
		new ValueFinfo( "Vm",
			ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getVm ),
			RFCAST( &HSolveHub::setVm )
		),
		new ValueFinfo( "Im",
			ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getIm ),
			&dummyFunc
		),
		new ValueFinfo( "inject",
			ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getInject ),
			RFCAST( &HSolveHub::setInject )
		),
	};

	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initCompartmentCinfo()->getThisFinfo( ) );
	assert( tf != 0 );

	static SolveFinfo compartmentZombieFinfo( 
		compartmentFields, 
		sizeof( compartmentFields ) / sizeof( Finfo* ),
		tf,
		"These fields will replace the original compartment fields so that the lookups refer to the solver rather "
		"than the compartment."
	);

	return &compartmentZombieFinfo;
}

Finfo* initHHChannelZombieFinfo()
{
	static Finfo* hhchannelFields[] =
	{
		new ValueFinfo( "Gbar", ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getHHChannelGbar ), 
			RFCAST( &HSolveHub::setHHChannelGbar )
		),
		new ValueFinfo( "Ek", ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getEk ), 
			RFCAST( &HSolveHub::setEk )
		),
		new ValueFinfo( "Gk", ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getGk ), 
			RFCAST( &HSolveHub::setGk )
		),
		new ValueFinfo( "Ik", ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getIk ), 
			&dummyFunc
		),
		new ValueFinfo( "X", ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getX ), 
			RFCAST( &HSolveHub::setX )
		),
		new ValueFinfo( "Y", ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getY ), 
			RFCAST( &HSolveHub::setY )
		),
		new ValueFinfo( "Z", ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getZ ), 
			RFCAST( &HSolveHub::setZ )
		),
	};

	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initHHChannelCinfo()->getThisFinfo( ) );
	assert( tf != 0 );

	static SolveFinfo hhchannelZombieFinfo( 
		hhchannelFields, 
		sizeof( hhchannelFields ) / sizeof( Finfo* ),
		tf
	);

	return &hhchannelZombieFinfo;
}

Finfo* initCaConcZombieFinfo()
{
	static Finfo* caconcFields[] =
	{
		new ValueFinfo( "Ca", ValueFtype1< double >::global(),
			GFCAST( &HSolveHub::getCa ), 
			RFCAST( &HSolveHub::setCa )
		),
	};

	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initCaConcCinfo()->getThisFinfo( ) );
	assert( tf != 0 );

	static SolveFinfo caconcZombieFinfo( 
		caconcFields, 
		sizeof( caconcFields ) / sizeof( Finfo* ),
		tf
	);

	return &caconcZombieFinfo;
}

static Finfo* compartmentZombieFinfo = initCompartmentZombieFinfo();
static Finfo* hhchannelZombieFinfo = initHHChannelZombieFinfo();
static Finfo* caconcZombieFinfo = initCaConcZombieFinfo();

/////////////////////////////////////////////////////////////////////////
// End of static initializers.
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// Constructor
/////////////////////////////////////////////////////////////////////////
HSolveHub::HSolveHub( )
	: integ_( 0 )
{ ; }

/////////////////////////////////////////////////////////////////////////
// Field access functions (for Hub)
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// Dest functions (for Hub)
/////////////////////////////////////////////////////////////////////////

void HSolveHub::hubFunc( const Conn* c, HSolveActive* integ )
{
	static_cast< HSolveHub* >( c->data() )->
		innerHubFunc( c->target(), integ );
}

void HSolveHub::innerHubFunc( Eref hub, HSolveActive* integ )
{
	hub_ = hub;
	integ_ = integ;
	
	manageCompartments( );
	manageHHChannels( );
	manageCaConcs( );
}

/**
 * In this destructor we need to put messages back to process,
 * and we need to replace the SolveFinfos on zombies with the
 * original ThisFinfo. This should really just use the clearFunc.
 */
void HSolveHub::destroy( const Conn* c )
{
	static Finfo* origCompartmentFinfo =
		const_cast< Finfo* >(
			initCompartmentCinfo()->getThisFinfo( ) );
	static Finfo* origHHChannelFinfo =
		const_cast< Finfo* >(
			initHHChannelCinfo()->getThisFinfo( ) );
	
	Element* hub = c->target().e;
	unsigned int eIndex = c->target().i;

	Conn* i = hub->targets( compartmentSolveFinfo->msg(), eIndex );
	while ( i->good() ) {
		i->target().e->setThisFinfo( origCompartmentFinfo );
		i->increment();
	}
	delete i;

	i = hub->targets( hhchannelSolveFinfo->msg(), eIndex );
	while ( i->good() ) {
		i->target().e->setThisFinfo( origHHChannelFinfo );
		i->increment();
	}
	delete i;

	Neutral::destroy( c );
}

void HSolveHub::childFunc( const Conn* c, int stage )
{
	// clear messages: first clean out zombies before the messages are
	// all deleted.
	if ( stage == 1 )
		clearFunc( c->target() );
	
	// Then fall back into what the Neutral version does
	Neutral::childFunc( c, stage );
}

/////////////////////////////////////////////////////////////////////////
// Class functions
/////////////////////////////////////////////////////////////////////////

void HSolveHub::idlist2elist(
	const vector< Id >& idlist,
	vector< Element* >& elist )
{
	vector< Id >::const_iterator id;
	for ( id = idlist.begin(); id != idlist.end(); id++ )
		elist.push_back( ( *id )() );
}	

void HSolveHub::manageCompartments( )
{
	const vector< Id >& idlist = integ_->getCompartments( );
	vector< Element* > elist;
	idlist2elist( idlist, elist );
	
	const vector< vector< Id > >& externalChannelIds =
		integ_->getExternalChannels();
	vector< Element* > extChanList;
	
	const Finfo* initFinfo = initCompartmentCinfo()->findFinfo( "init" );
	vector< Element* >::const_iterator i;
	for ( i = elist.begin(); i != elist.end(); i++ ) {
		zombify( hub_, *i, compartmentSolveFinfo, compartmentZombieFinfo );
		
		// Compartment receives 2 shared messages from Tick's "process"
		Eref( *i ).dropAll( initFinfo->msg() );
		
		redirectDynamicMessages( *i );
	}
	
	/*
	 * Redirecting dest/shared messages
	 */
	for ( unsigned int ic = 0; ic < elist.size(); ic++ ) {
		// The 'retain' flag at the end is 1: we do not delete the original
		// message to the compartment.
		redirectDestMessages(
			hub_, elist[ ic ],
			hubCompartmentInjectFinfo, compartmentInjectFinfo, 
			ic, compartmentInjectMap_,
			&elist, 0,
			1 );
		
		extChanList.clear();
		idlist2elist( externalChannelIds[ ic ], extChanList );
		redirectDestMessages(
			hub_, elist[ ic ],
			hubCompartmentChannelFinfo, compartmentChannelFinfo, 
			ic, compartmentChannelMap_,
			&elist, &extChanList,
			1 );
	}
}

void HSolveHub::manageHHChannels( )
{
	const vector< Id >& idlist = integ_->getHHChannels( );
	vector< Element* > elist;
	idlist2elist( idlist, elist );
	
	vector< Element* >::const_iterator i;
	for ( i = elist.begin(); i != elist.end(); i++ ) {
		zombify( hub_, *i, hhchannelSolveFinfo, hhchannelZombieFinfo );
		
		redirectDynamicMessages( *i );
	}
}

void HSolveHub::manageCaConcs( )
{
	const vector< Id >& idlist = integ_->getCaConcs( );
	vector< Element* > elist;
	idlist2elist( idlist, elist );
	
	vector< Element* >::const_iterator i;
	for ( i = elist.begin(); i != elist.end(); i++ ) {
		zombify( hub_, *i, caconcSolveFinfo, caconcZombieFinfo );
		
		redirectDynamicMessages( *i );
	}
}

/**
 * Clears out all the messages to zombie objects
 */
void HSolveHub::clearFunc( Eref hub )
{
	clearMsgsFromFinfo( hub, compartmentSolveFinfo );
	clearMsgsFromFinfo( hub, hhchannelSolveFinfo );

	hub.dropAll( compartmentInjectFinfo->msg() );
}

void HSolveHub::clearMsgsFromFinfo( Eref hub, const Finfo * f )
{
	Conn* c = hub.e->targets( f->msg(), hub.i );
	vector< Element* > list;
	while ( c->good() ) {
		list.push_back( c->target().e );
		c->increment();
	}
	delete c;
	hub.dropAll( f->msg() );
	
	vector< Element* >::iterator i;
	for ( i = list.begin(); i != list.end(); i++ ) unzombify( *i );
}

void HSolveHub::unzombify( Element* e )
{
	const Cinfo* ci = e->cinfo();
	bool ret = ci->schedule( e, ConnTainer::Default );
	assert( ret );
	e->setThisFinfo( const_cast< Finfo* >( ci->getThisFinfo() ) );
	redirectDynamicMessages( e );
}

/**
 * This operation turns the target element e into a zombie controlled
 * by the hub/solver. It gets rid of any process message coming into 
 * the zombie and replaces it with one from the solver.
 */
void HSolveHub::zombify( 
	Eref hub, Eref e,
	const Finfo* hubFinfo, Finfo* solveFinfo )
{
	// Replace the original procFinfo with one from the hub.
	const Finfo* procFinfo = e->findFinfo( "process" );
	e.dropAll( procFinfo->msg() );
	bool ret = hub.add( hubFinfo->msg(), e, procFinfo->msg(), 
		ConnTainer::Default );
	assert( ret );

	// Redirect original messages from the zombie to the hub.
	// Pending.

	// Replace the 'ThisFinfo' on the solved element
	e->setThisFinfo( solveFinfo );
}

/**
 * This function redirects messages arriving at zombie elements onto
 * the hub. 
 * e is the zombie element whose messages are being redirected to the hub.
 * eFinfo is the Finfo holding those messages.
 * hubFinfo is the Finfo on the hub which will now handle the messages.
 * eIndex is the index to look up the element.
 */
void HSolveHub::redirectDestMessages(
	Eref hub,
	Eref e,
	const Finfo* hubFinfo,
	const Finfo* eFinfo,
	unsigned int eIndex,
	vector< unsigned int >& map, 
	vector< Element * >* elist,
	vector< Element * >* include,
	bool retain )
{
	Conn* i = e.e->targets( eFinfo->msg(), e.i );
	vector< Eref > srcElements;
	vector< int > srcMsg;
	vector< const ConnTainer* > dropList;

	while( i->good() ) {
		Element* tgt = i->target().e;
		
		// Handle messages going outside purview of solver.
		bool inElist = find( elist->begin(), elist->end(), tgt ) != elist->end();
		bool inInclude =
			include == 0 ||
			find( include->begin(), include->end(), tgt ) != include->end();
		
		if ( ! inElist && inInclude ) {
			map.push_back( eIndex );
			srcElements.push_back( i->target() );
			srcMsg.push_back( i->targetMsg() );
			if ( !retain )
				dropList.push_back( i->connTainer() );
		}
		i->increment();
	}
	delete i;

	e.dropVec( eFinfo->msg(), dropList );

	for ( unsigned int j = 0; j != srcElements.size(); j++ ) {
		bool ret = srcElements[j].add( srcMsg[j], hub, hubFinfo->msg(),
			ConnTainer::Default );
		assert( ret );
	}
}

/**
 * Here we replace the existing DynamicFinfos and their messages with
 * new ones for the updated access functions.
 *
 * It would be nice to retain everything and only replace the 
 * access functions, but this gets too messy as it requires poking the
 * new funcVecs into the remote Msgs. So instead we delete the 
 * old DynamicFinfos and recreate them.
 */
// Assumption e is a simple element. Replace it with Eref to make it general
void HSolveHub::redirectDynamicMessages( Element* e )
{
	vector< Finfo* > flist;
	// We get a list of DynamicFinfos independent of the Finfo vector on 
	// the Element, because we will be messing up the iterators on the
	// element.
	e->listLocalFinfos( flist );
	vector< Finfo* >::iterator i;

	// Go through flist noting messages, deleting finfo, and rebuilding.
	for( i = flist.begin(); i != flist.end(); ++i )
	{
		const DynamicFinfo *df = dynamic_cast< const DynamicFinfo* >( *i );
		if ( df == 0 )
			continue;
		
		vector< Eref > srcElements;
		vector< const Finfo* > srcFinfos;
		Conn* c = e->targets( ( *i )->msg(), 0 ); //zero index for SE

		// note messages.
		while( c->good() ) {
			srcElements.push_back( c->target() );
			srcFinfos.push_back( 
				c->target().e->findFinfo( c->targetMsg() ) );
			c->increment();
		}
		delete c;
		string name = df->name();
		bool ret = e->dropFinfo( df );
		assert( ret );
		const Finfo* origFinfo = e->findFinfo( name );
		assert( origFinfo );

		unsigned int max = srcFinfos.size();
		for ( unsigned int i =  0; i < max; i++ ) {
			ret = srcElements[ i ].add( srcFinfos[ i ]->name(),
				e, name );
			/*
			ret = srcElements[ i ].add( srcFinfos[ i ]->msg(),
				e, origFinfo->msg(), ConnTainer::Default );
			*/
			// ret = srcFinfos[ i ]->add( srcElements[ i ], e, origFinfo );
			assert( ret );
		}
	}
}

/**
 * Looks up the solver from the zombie element e. Returns the solver
 * element, or null on failure. 
 * It needs the originating Finfo on the solver that connects to the zombie,
 * as the srcFinfo.
 * Also passes back the index of the zombie element on this set of
 * messages. This is NOT the absolute Conn index.
 */
HSolveHub* HSolveHub::getHubFromZombie( Eref e, unsigned int& index )
{
	Conn* c = e.e->targets( "process", e.i );
	if ( c->good() ) {
		index = c->targetIndex();
		HSolveHub* nh = static_cast< HSolveHub* >( c->target().data() );
		c->increment();
		assert( !c->good() ); // Should only be one process incoming.
		return dynamic_cast< HSolveHub* >( nh );
	}
	delete c;
	return 0;
}

/////////////////////////////////////////////////////////////////////////
// Field access functions (Biophysics)
/////////////////////////////////////////////////////////////////////////

void HSolveHub::setVm( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, compartmentVmFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setVm( index, value );
}

double HSolveHub::getVm( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getVm( index );
	
	return 0.0;
}

void HSolveHub::setInject( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, compartmentInjectFieldFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setInject( index, value );
}

double HSolveHub::getInject( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getInject( index );
	
	return 0.0;
}

double HSolveHub::getIm( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getIm( index );
	
	return 0.0;
}

void HSolveHub::setHHChannelGbar( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, channelGbarFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setHHChannelGbar( index, value );
}

double HSolveHub::getHHChannelGbar( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getHHChannelGbar( index );
	
	return 0.0;
}

void HSolveHub::setEk( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, channelEkFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setEk( index, value );
}

double HSolveHub::getEk( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getEk( index );
	
	return 0.0;
}

void HSolveHub::setGk( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, channelGkFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setGk( index, value );
}

double HSolveHub::getGk( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getGk( index );
	
	return 0.0;
}

double HSolveHub::getIk( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getIk( index );
	
	return 0.0;
}

void HSolveHub::setX( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, channelXFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setX( index, value );
}

double HSolveHub::getX( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getX( index );
	
	return 0.0;
}

void HSolveHub::setY( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, channelYFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setY( index, value );
}

double HSolveHub::getY( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getY( index );
	
	return 0.0;
}

void HSolveHub::setZ( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, channelZFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setZ( index, value );
}

double HSolveHub::getZ( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getZ( index );
	
	return 0.0;
}

void HSolveHub::setCaBasal( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, caConcCaFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setCaBasal( index, value );
}

double HSolveHub::getCaBasal( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getCaBasal( index );
	
	return 0.0;
}

void HSolveHub::setCa( const Conn* c, double value )
{
	Eref e = c->target();
	set< double >( e, caConcCaFinfo, value );
	
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		nh->integ_->setCa( index, value );
}

double HSolveHub::getCa( Eref e )
{
	unsigned int index;
	HSolveHub* nh = getHubFromZombie( e, index );
	
	if ( nh )
		return nh->integ_->getCa( index );
	
	return 0.0;
}

/////////////////////////////////////////////////////////////////////////
// Dest functions (Biophysics)
/////////////////////////////////////////////////////////////////////////
void HSolveHub::compartmentInjectMsgFunc( const Conn* c, double value )
{
	Element* hub = c->target().e;
	unsigned int index = c->targetIndex();
	HSolveHub* nh = static_cast< HSolveHub* >( hub->data() );
	
	assert( nh != 0 );
	assert( index < nh->compartmentInjectMap_.size() );
	
	nh->integ_->addInject( nh->compartmentInjectMap_[ index ], value );
}

void HSolveHub::compartmentChannelFunc( const Conn* c, double v1, double v2 )
{
	Element* hub = c->target().e;
	unsigned int index = c->targetIndex();
	HSolveHub* nh = static_cast< HSolveHub* >( hub->data() );
	
	assert( nh != 0 );
	assert( index < nh->compartmentChannelMap_.size() );
	
	nh->integ_->addGkEk(
		nh->compartmentChannelMap_[ index ],
		v1, v2 );
}
