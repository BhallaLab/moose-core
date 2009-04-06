/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "HHChannel.h"
#include "HHChannel2D.h"

const Cinfo* initHHChannel2DCinfo()
{
	static Finfo* xGateShared[] =
	{
		new SrcFinfo( "lookup", Ftype2< double, double >::global() ),
		new DestFinfo( "gate", Ftype2< double, double >::global(),
				RFCAST( &HHChannel::xGateFunc ) ),
	};

	static Finfo* yGateShared[] =
	{
		new SrcFinfo( "lookup", Ftype2< double, double >::global() ),
		new DestFinfo( "gate", Ftype2< double, double >::global(),
				RFCAST( &HHChannel::yGateFunc ) ),
	};

	static Finfo* zGateShared[] =
	{
		new SrcFinfo( "lookup", Ftype2< double, double >::global() ),
		new DestFinfo( "gate", Ftype2< double, double >::global(),
			RFCAST( &HHChannel::zGateFunc ) ),
	};


	static Finfo* HHChannel2DFinfos[] =
	{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
		new ValueFinfo( "Xindex", ValueFtype1< string >::global(),
			GFCAST( &HHChannel2D::getXindex ),
			RFCAST( &HHChannel2D::setXindex ),
			"  " ),
		new ValueFinfo( "Yindex",
			ValueFtype1< string >::global(),
			GFCAST( &HHChannel2D::getYindex ),
			RFCAST( &HHChannel2D::setYindex ),
			"  " ),
		new ValueFinfo( "Zindex",
			ValueFtype1< string >::global(),
			GFCAST( &HHChannel2D::getZindex ),
			RFCAST( &HHChannel2D::setZindex ),
			"  " ),

///////////////////////////////////////////////////////
// Shared message definitions
///////////////////////////////////////////////////////
		new SharedFinfo( "xGate2D", xGateShared,
			sizeof( xGateShared ) / sizeof( Finfo* ),
			"This is a shared message to communicate with the X gate. "
			"Sends out Vm. "
			"Receives lookedup up values A and B for the Vm. "
			"The A term is the alpha term from HH equations. "
			"The B term is actually alpha + beta, precalculated. " ),
		new SharedFinfo( "yGate2D", yGateShared,
			sizeof( yGateShared ) / sizeof( Finfo* ),
			"Shared message for Y gate. Fields as in X gate."),
		new SharedFinfo( "zGate2D", zGateShared,
			sizeof( zGateShared ) / sizeof( Finfo* ),
			"Shared message for Z gate. Fields as in X gate."),

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
		new DestFinfo( "concen2",
			ValueFtype1< double >::global(),
			RFCAST( &HHChannel2D::conc2Func ),
			"  " ),
	};

	static string doc[] =
	{
		"Name", "HHChannel2D",
		"Author", "Niraj Dudani, 2009, NCBS",
		"Description", "HHChannel2D: Hodgkin-Huxley type voltage-gated Ion channel. Something "
				"like the old tabchannel from GENESIS, but also presents "
				"a similar interface as hhchan from GENESIS. ",
	};

	static Cinfo HHChannel2DCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),
		initHHChannelCinfo(),
		HHChannel2DFinfos,
		sizeof( HHChannel2DFinfos ) / sizeof(Finfo *),
		ValueFtype1< HHChannel2D >::global()
	);

	return &HHChannel2DCinfo;
}

static const Cinfo* HHChannel2DCinfo = initHHChannel2DCinfo();

static const Slot xGateSlot =
	initHHChannel2DCinfo()->getSlot( "xGate.lookup" );
static const Slot yGateSlot =
	initHHChannel2DCinfo()->getSlot( "yGate.lookup" );
static const Slot zGateSlot =
	initHHChannel2DCinfo()->getSlot( "zGate.lookup" );

static const Slot xGate2DSlot =
	initHHChannel2DCinfo()->getSlot( "xGate2D.lookup" );
static const Slot yGate2DSlot =
	initHHChannel2DCinfo()->getSlot( "yGate2D.lookup" );
static const Slot zGate2DSlot =
	initHHChannel2DCinfo()->getSlot( "zGate2D.lookup" );


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
void HHChannel2D::setXindex( const Conn* c, string Xindex )
{
	static_cast< HHChannel2D* >( c->data() )->
		innerSetXindex( c->target(), Xindex );
}

string HHChannel2D::getXindex( Eref e )
{
	return static_cast< HHChannel2D* >( e.data() )->Xindex_;
}

void HHChannel2D::setYindex( const Conn* c, string Yindex )
{
	static_cast< HHChannel2D* >( c->data() )->
		innerSetYindex( c->target(), Yindex );
}

string HHChannel2D::getYindex( Eref e )
{
	return static_cast< HHChannel2D* >( e.data() )->Yindex_;
}

void HHChannel2D::setZindex( const Conn* c, string Zindex )
{
	static_cast< HHChannel2D* >( c->data() )->
		innerSetZindex( c->target(), Zindex );
}

string HHChannel2D::getZindex( Eref e )
{
	return static_cast< HHChannel2D* >( e.data() )->Zindex_;
}

void HHChannel2D::innerSetXindex( Eref e, string Xindex )
{
	if ( Xindex == Xindex_ )
		return;

	Xindex_ = Xindex;
	Xdep0_ = dependency( Xindex, 0 );
	Xdep1_ = dependency( Xindex, 1 );
	
	assert( Xdep0_ >= 0 );
}

void HHChannel2D::innerSetYindex( Eref e, string Yindex )
{
	if ( Yindex == Yindex_ )
		return;
	
	Yindex_ = Yindex;
	Ydep0_ = dependency( Yindex, 0 );
	Ydep1_ = dependency( Yindex, 1 );
	
	assert( Ydep0_ >= 0 );
}

void HHChannel2D::innerSetZindex( Eref e, string Zindex )
{
	if ( Zindex == Zindex_ )
		return;

	Zindex_ = Zindex;
	Zdep0_ = dependency( Zindex, 0 );
	Zdep1_ = dependency( Zindex, 1 );
	
	assert( Zdep0_ >= 0 );
}

int HHChannel2D::dependency( string index, unsigned int dim )
{
	static vector< map< string, int > > dep;
	if ( dep.empty() ) {
		dep.resize( 2 );
		
		dep[ 0 ][ "VOLT_INDEX" ] = 0;
		dep[ 0 ][ "C1_INDEX" ] = 1;
		dep[ 0 ][ "C2_INDEX" ] = 2;
		
		dep[ 0 ][ "VOLT_C1_INDEX" ] = 0;
		dep[ 0 ][ "VOLT_C2_INDEX" ] = 0;
		dep[ 0 ][ "C1_C2_INDEX" ] = 1;
		
		dep[ 1 ][ "VOLT_INDEX" ] = -1;
		dep[ 1 ][ "C1_INDEX" ] = -1;
		dep[ 1 ][ "C2_INDEX" ] = -1;
		
		dep[ 1 ][ "VOLT_C1_INDEX" ] = 1;
		dep[ 1 ][ "VOLT_C2_INDEX" ] = 2;
		dep[ 1 ][ "C1_C2_INDEX" ] = 2;
	}
	
	if ( dep[ dim ].find( index ) == dep[ dim ].end() )
		return -1;
	
	return dep[ dim ][ index ];
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HHChannel2D::conc2Func( const Conn* c, double conc )
{
	static_cast< HHChannel2D* >( c->data( ) )->conc2_ = conc;
}

///////////////////////////////////////////////////
// Virtual function definitions
///////////////////////////////////////////////////
unsigned int HHChannel2D::dimension( string gateType ) const
{
	if ( gateType == "X" )
		return ( Xdep1_ == -1 ) ? 1 : 2;
	else if ( gateType == "Y" )
		return ( Ydep1_ == -1 ) ? 1 : 2;
	else if ( gateType == "Z" )
		return ( Zdep1_ == -1 ) ? 1 : 2;
	else
		assert( 0 );
}

string HHChannel2D::chanFinfo( string gateType ) const
{
	unsigned int dim = dimension( gateType );
	assert( dim == 1 || dim == 2 );
	
	if ( dim == 1 )
		return HHChannel::chanFinfo( gateType );
	
	if ( gateType == "X" )
		return "xGate2D";
	else if ( gateType == "Y" )
		return "yGate2D";
	else if ( gateType == "Z" )
		return "zGate2D";
	else
		assert( 0 );
}

string HHChannel2D::gateFinfo( string gateType ) const
{
	unsigned int dim = dimension( gateType );
	assert( dim == 1 || dim == 2 );
	
	if ( dim == 1 )
		return HHChannel::gateFinfo( gateType );
	else
		return "gate2D";
}

string HHChannel2D::gateClass( string gateType ) const
{
	unsigned int dim = dimension( gateType );
	assert( dim == 1 || dim == 2 );
	
	if ( dim == 1 )
		return HHChannel::gateClass( gateType );
	else
		return "HHGate2D";
}

void HHChannel2D::lookupXrates( Eref e )
{
	double var1;
	double var2;
	unsigned int dim = ( Xdep1_ == -1 ) ? 1 : 2;
	
	switch( Xdep0_ ) {
		case 0:		var1 = Vm_; break;
		case 1:		var1 = conc_; break;
		case 2:		var1 = conc2_; break;
		default:	assert( 0 );
	};
	
	if ( dim == 1 ) {	
		send1< double >( e, xGateSlot, var1 );
		return;
	}
	
	switch( Xdep1_ ) {
		case 0:		var2 = Vm_; break;
		case 1:		var2 = conc_; break;
		case 2:		var2 = conc2_; break;
		default:	assert( 0 );
	};
	
	send2< double, double >( e, xGate2DSlot, var1, var2 );
}

void HHChannel2D::lookupYrates( Eref e )
{
	double var1;
	double var2;
	unsigned int dim = ( Ydep1_ == -1 ) ? 1 : 2;
	
	switch( Ydep0_ ) {
		case 0:		var1 = Vm_; break;
		case 1:		var1 = conc_; break;
		case 2:		var1 = conc2_; break;
		default:	assert( 0 );
	};
	
	if ( dim == 1 ) {	
		send1< double >( e, yGateSlot, var1 );
		return;
	}
	
	switch( Ydep1_ ) {
		case 0:		var2 = Vm_; break;
		case 1:		var2 = conc_; break;
		case 2:		var2 = conc2_; break;
		default:	assert( 0 );
	};
	
	send2< double, double >( e, yGate2DSlot, var1, var2 );
}

void HHChannel2D::lookupZrates( Eref e )
{
	double var1;
	double var2;
	unsigned int dim = ( Zdep1_ == -1 ) ? 1 : 2;
	
	switch( Zdep0_ ) {
		case 0:		var1 = Vm_; break;
		case 1:		var1 = conc_; break;
		case 2:		var1 = conc2_; break;
		default:	assert( 0 );
	};
	
	if ( dim == 1 ) {	
		send1< double >( e, zGateSlot, var1 );
		return;
	}
	
	switch( Zdep1_ ) {
		case 0:		var2 = Vm_; break;
		case 1:		var2 = conc_; break;
		case 2:		var2 = conc2_; break;
		default:	assert( 0 );
	};
	
	send2< double, double >( e, zGate2DSlot, var1, var2 );
}

///////////////////////////////////////////////////
// Unit tests
///////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS

void testHHChannel2D()
{
	;
}
#endif 
