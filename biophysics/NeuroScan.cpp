/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "../element/Neutral.h"
#include "SpikeGen.h"
#include <queue>
#include "SynInfo.h"
#include "RateLookup.h"
#include "HSolveStruct.h"
#include "SynChan.h"
#include "NeuroHub.h"
#include "NeuroScanBase.h"
#include "NeuroScan.h"

#include "HHChannel.h"

const Cinfo* initNeuroScanCinfo()
{
	// Shared message to NeuroHub
	static Finfo* hubShared[] =
	{
		new SrcFinfo( "compartment",
			Ftype2< vector< double >*, vector< Element* >* >::global() ),
		new SrcFinfo( "channel",
			Ftype1< vector< Element* >* >::global() ),
		new SrcFinfo( "spikegen",
			Ftype1< vector< Element* >* >::global() ),
		new SrcFinfo( "synchan",
			Ftype1< vector< Element* >* >::global() ),
	};
	
	static Finfo* gateShared[] =
	{
		new SrcFinfo( "Vm",
			Ftype1< double >::global() ),
		new DestFinfo( "gate",
			Ftype2< double, double >::global(),
			RFCAST( &NeuroScan::gateFunc ) ),
	};
	
	static Finfo* neuroScanFinfos[] = 
	{
	//////////////////////////////////////////////////////////////////
	// Field definitions
	//////////////////////////////////////////////////////////////////
		new ValueFinfo( "VDiv", ValueFtype1< int >::global(),
			GFCAST( &NeuroScan::getVDiv ),
			RFCAST( &NeuroScan::setVDiv )
		),
		new ValueFinfo( "VMin", ValueFtype1< double >::global(),
			GFCAST( &NeuroScan::getVMin ),
			RFCAST( &NeuroScan::setVMin )
		),
		new ValueFinfo( "VMax", ValueFtype1< double >::global(),
			GFCAST( &NeuroScan::getVMax ),
			RFCAST( &NeuroScan::setVMax )
		),
		new ValueFinfo( "CaDiv", ValueFtype1< int >::global(),
			GFCAST( &NeuroScan::getCaDiv ),
			RFCAST( &NeuroScan::setCaDiv )
		),
		new ValueFinfo( "CaMin", ValueFtype1< double >::global(),
			GFCAST( &NeuroScan::getCaMin ),
			RFCAST( &NeuroScan::setCaMin )
		),
		new ValueFinfo( "CaMax", ValueFtype1< double >::global(),
			GFCAST( &NeuroScan::getCaMax ),
			RFCAST( &NeuroScan::setCaMax )
		),
		
	//////////////////////////////////////////////////////////////////
	// SharedFinfo definitions
	//////////////////////////////////////////////////////////////////
		new SharedFinfo( "scan-hub", hubShared,
			sizeof( hubShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "gate", gateShared,
			sizeof( gateShared ) / sizeof( Finfo* ) ),
	//////////////////////////////////////////////////////////////////
	// SrcFinfo definitions
	//////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////
	// DestFinfo definitions
	//////////////////////////////////////////////////////////////////
		new DestFinfo( "hubCreate",
			Ftype0::global(),
			&NeuroScan::hubCreateFunc ),
		new DestFinfo( "readModel",
			Ftype2< Id, double >::global(),
			RFCAST( &NeuroScan::readModelFunc ) ),
	};

	static Cinfo neuroScanCinfo(
		"NeuroScan",
		"Niraj Dudani, 2007, NCBS",
		"NeuroScan: HSolve component for reading in neuronal models from MOOSE object tree.",
		initNeutralCinfo(),
		neuroScanFinfos,
		sizeof( neuroScanFinfos ) / sizeof( Finfo* ),
		ValueFtype1< NeuroScan >::global()
	);

	return &neuroScanCinfo;
}

static const Cinfo* neuroScanCinfo = initNeuroScanCinfo();

static const Slot hubCompartmentSlot =
	initNeuroScanCinfo()->getSlot( "scan-hub.compartment" );
static const Slot hubChannelSlot =
	initNeuroScanCinfo()->getSlot( "scan-hub.channel" );
static const Slot hubSpikegenSlot =
	initNeuroScanCinfo()->getSlot( "scan-hub.spikegen" );
static const Slot hubSynchanSlot =
	initNeuroScanCinfo()->getSlot( "scan-hub.synchan" );
static const Slot gateVmSlot =
	initNeuroScanCinfo()->getSlot( "gate.Vm" );
static const Slot gateSlot =
	initNeuroScanCinfo()->getSlot( "gate" );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void NeuroScan::setVDiv( const Conn* c, int vDiv )
{
	static_cast< NeuroScan* >( c->data() )->vDiv_ = vDiv;
}

int NeuroScan::getVDiv( Eref e )
{
	return static_cast< NeuroScan* >( e.data() )->vDiv_;
}

void NeuroScan::setVMin( const Conn* c, double vMin )
{
	static_cast< NeuroScan* >( c->data() )->vMin_ = vMin;
}

double NeuroScan::getVMin( Eref e )
{
	return static_cast< NeuroScan* >( e.data() )->vMin_;
}

void NeuroScan::setVMax( const Conn* c, double vMax )
{
	static_cast< NeuroScan* >( c->data() )->vMax_ = vMax;
}

double NeuroScan::getVMax( Eref e )
{
	return static_cast< NeuroScan* >( e.data() )->vMax_;
}

void NeuroScan::setCaDiv( const Conn* c, int caDiv )
{
	static_cast< NeuroScan* >( c->data() )->caDiv_ = caDiv;
}

int NeuroScan::getCaDiv( Eref e )
{
	return static_cast< NeuroScan* >( e.data() )->caDiv_;
}

void NeuroScan::setCaMin( const Conn* c, double caMin )
{
	static_cast< NeuroScan* >( c->data() )->caMin_ = caMin;
}

double NeuroScan::getCaMin( Eref e )
{
	return static_cast< NeuroScan* >( e.data() )->caMin_;
}

void NeuroScan::setCaMax( const Conn* c, double caMax )
{
	static_cast< NeuroScan* >( c->data() )->caMax_ = caMax;
}

double NeuroScan::getCaMax( Eref e )
{
	return static_cast< NeuroScan* >( e.data() )->caMax_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

/// Creating hub as solver's child.
void NeuroScan::hubCreateFunc( const Conn* c )
{
	static_cast< NeuroScan* >( c->data() )->
		innerHubCreateFunc( c->target() );
}

void NeuroScan::innerHubCreateFunc( Eref scan )
{
	// Hub element's data field is owned by its parent HSolve
	// structure, so we set it's noDelFlag to 1.
	Id solve = Neutral::getParent( scan );
	Element* hub = initNeuroHubCinfo()->create( 
		Id::scratchId(), "hub",
		static_cast< void* >( &hub_ ), 1 );
	Eref( solve() ).add( "childSrc", hub, "child" );
	
	// Setting up shared msg between scanner and hub.
	Eref( scan ).add( "scan-hub", hub, "scan-hub" );
}

void NeuroScan::readModelFunc( const Conn* c, Id seed, double dt  )
{
	static_cast< NeuroScan* >( c->data() )->
		innerReadModelFunc( c->target(), seed, dt );
}

void NeuroScan::innerReadModelFunc( Eref e, Id seed, double dt  )
{
	scanElm_ = e;
	initialize( seed, dt );
	
	vector< Id >::iterator i;
	vector< Element* > elist;
	for ( i = compartmentId_.begin(); i != compartmentId_.end(); ++i )
		elist.push_back( ( *i )() );
	send2< vector< double >*, vector< Element* >* >(
		scanElm_, hubCompartmentSlot, &V_, &elist );
	
	elist.clear();
	for ( i = channelId_.begin(); i != channelId_.end(); ++i )
		elist.push_back( ( *i )() );
	send1< vector< Element* >* >(
		scanElm_, hubChannelSlot, &elist );
	
	elist.clear();
	vector< SpikeGenStruct >::iterator j;
	for ( j = spikegen_.begin(); j != spikegen_.end(); ++j )
		elist.push_back( j->elm_ );
	send1< vector< Element* >* >(
		scanElm_, hubSpikegenSlot, &elist );
	
	elist.clear();
	vector< SynChanStruct >::iterator k;
	for ( k = synchan_.begin(); k != synchan_.end(); ++k )
		elist.push_back( k->elm_ );
	send1< vector< Element* >* >(
		scanElm_, hubSynchanSlot, &elist );
}

void NeuroScan::gateFunc( const Conn* c, double A, double B )
{
	NeuroScan* ns = static_cast< NeuroScan* >( c->data() );
	ns->A_ = A;
	ns->B_ = B;
}

///////////////////////////////////////////////////
// Portal functions (to scan model)
///////////////////////////////////////////////////

vector< Id > NeuroScan::children( Id self, Id parent )
{
	vector< Id > child = neighbours( self );
	child.erase(
		remove( child.begin(), child.end(), parent ),
		child.end()
	);
	return child;
}

vector< Id > NeuroScan::neighbours( Id compartment )
{
	vector< Id > neighbour;
	targets( compartment, "axial", neighbour );
	targets( compartment, "raxial", neighbour );
	return neighbour;
}

vector< Id > NeuroScan::channels( Id compartment )
{
	vector< Id > channel;
	// Request only for elements of type "HHChannel" since
	// channel messages can lead to synchans as well.
	targets( compartment, "channel", channel, "HHChannel" );
	return channel;
}

int NeuroScan::gates( Id channel, vector< Id >& ret )
{
	vector< Id > gate;
	targets( channel, "xGate", gate );
	targets( channel, "yGate", gate );
	targets( channel, "zGate", gate );
	ret.insert( ret.end(), gate.begin(), gate.end() );
	return gate.size();
}

Id NeuroScan::presyn( Id compartment )
{
	vector< Id > spikegen;
	targets( compartment, "VmSrc", spikegen );
	ProcInfoBase p;
	if ( spikegen.size() > 0 ) {
		SetConn c( spikegen.front()(), 0 );
		SpikeGen::reinitFunc( &c, &p );
		return spikegen[ 0 ];
	}
	else
		return Id::badId();
}

vector< Id > NeuroScan::postsyn( Id compartment )
{
	vector< Id > channel, synchan;
	// "channel" msgs lead to SynChans as well HHChannels, so request
	// explicitly for former.
	targets( compartment, "channel", channel, "SynChan" );
	ProcInfoBase p;
	p.dt_ = dt_;
	vector< Id >::iterator ichan;
	for ( ichan = channel.begin(); ichan != channel.end(); ++ichan ) {
		// Initializing element
		SetConn c( ( *ichan )(), 0 );
		SynChan::reinitFunc( &c, &p );
		// Remembering it
		synchan.push_back( *ichan );
	}
	
	return synchan;
}

int NeuroScan::caTarget( Id channel, vector< Id >& ret )
{
	return targets( channel, "IkSrc", ret );
}

int NeuroScan::caDepend( Id channel, vector< Id >& ret )
{
	return targets( channel, "concen", ret );
}

void NeuroScan::field( Id object, string field, double& value )
{
	get< double >( object(), field, value );
}

void NeuroScan::field( Id object, string field, int& value )
{
	get< int >( object(), field, value );
}

void NeuroScan::synchanFields( Id synchan, SynChanStruct& scs )
{
	SetConn c( synchan(), 0 );
	ProcInfoBase p;
	p.dt_ = dt_;
	
	SynChan::reinitFunc( &c, &p );
	set< SynChanStruct* >( synchan(), "scan", &scs );
}

//~ Meant for small hack below. Temporary.
#include "../builtins/Interpol.h"
#include "HHGate.h"
void NeuroScan::rates(
	Id gate,
	const vector< double >& grid,
	vector< double >& A,
	vector< double >& B )
{
//~ Temporary
HHGate* h = static_cast< HHGate *>( gate()->data() );
	scanElm_.add( "gate", gate(), "gate" );
	
	A.resize( grid.size() );
	B.resize( grid.size() );
	
	//~ Uglier hack to access Interpol's tables directly.
	//~ Strictly for debugging purposes
	//~ const vector< double >& AA = h->A().table();
	//~ const vector< double >& BB = h->B().table();
	//~ for ( unsigned int i = 0; i < grid.size(); i++ ) {
		//~ A[ i ] = AA[ i ];
		//~ B[ i ] = BB[ i ];
	//~ }
	
	vector< double >::const_iterator igrid;
	vector< double >::iterator ia = A.begin();
	vector< double >::iterator ib = B.begin();
	for ( igrid = grid.begin(); igrid != grid.end(); ++igrid ) {
		//~ send1< double >( scanElm_, gateVmSlot, *igrid );
		//~ Temporary
		A_ = h->A().innerLookup( *igrid );
		B_ = h->B().innerLookup( *igrid );
		
		// locals A_ and B_ receive rate values from gate via a callback.
		*ia = A_;
		*ib = B_;
		++ia, ++ib;
	}
	
	scanElm_.dropAll( "gate" );
}

///////////////////////////////////////////////////
// Utility functions
///////////////////////////////////////////////////

int NeuroScan::targets(
	Id object,
	const string& msg,
	vector< Id >& target,
	const string& type ) const
{
	unsigned int oldSize = target.size();
	
	Id found;
	Conn* i = object()->targets( msg, 0 );
	for ( ; i->good(); i->increment() ) {
		found = i->target()->id();
		if ( type != "" && !isType( found, type ) )	// speed this up
			continue;
		
		target.push_back( found );
	}
	delete i;
	
	return target.size() - oldSize;
}

bool NeuroScan::isType( Id object, const string& type ) const
{
	return object()->cinfo()->isA( Cinfo::find( type ) );
}

///////////////////////////////////////////////////
// Unit tests
///////////////////////////////////////////////////
