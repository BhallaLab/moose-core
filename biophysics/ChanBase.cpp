/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ChanBase.h"

static SrcFinfo1< double > permeability( "permeability", 
	"Conductance term going out to GHK object" );
static SrcFinfo2< double, double > channelOut( "channelOut", 
	"Sends channel variables Gk and Ek to compartment" );
static SrcFinfo1< double > IkOut( "IkOut", 
	"Channel current. This message typically goes to concen"
	"objects that keep track of ion concentration." );

const Cinfo* ChanBase::initCinfo()
{
	/////////////////////////////////////////////////////////////////////
	// Shared messages
	/////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////
	/// ChannelOut SrcFinfo defined above.
	static DestFinfo Vm( "Vm", 
		"Handles Vm message coming in from compartment",
		new OpFunc1< ChanBase, double >( &ChanBase::handleVm ) );

	static Finfo* channelShared[] =
	{
		&channelOut, &Vm
	};
	static SharedFinfo channel( "channel", 
		"This is a shared message to couple channel to compartment. "
		"The first entry is a MsgSrc to send Gk and Ek to the compartment "
		"The second entry is a MsgDest for Vm from the compartment.",
		channelShared, sizeof( channelShared ) / sizeof( Finfo* )
	);

	///////////////////////////////////////////////////////
	// Here we reuse the Vm DestFinfo declared above.

	/// Permability SrcFinfo defined above.
	static Finfo* ghkShared[] =
	{
		&Vm, &permeability
	};
	static SharedFinfo ghk( "ghk", 
		"Message to Goldman-Hodgkin-Katz object",
		ghkShared, sizeof( ghkShared ) / sizeof( Finfo* ) );

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////

		static ValueFinfo< ChanBase, double > Gbar( "Gbar",
			"Maximal channel conductance",
			&ChanBase::setGbar,
			&ChanBase::getGbar
		);
		static ValueFinfo< ChanBase, double > Ek( "Ek", 
			"Reversal potential of channel",
			&ChanBase::setEk,
			&ChanBase::getEk
		);
		static ValueFinfo< ChanBase, double > Gk( "Gk",
			"Channel conductance variable",
			&ChanBase::setGk,
			&ChanBase::getGk
		);
		static ReadOnlyValueFinfo< ChanBase, double > Ik( "Ik",
			"Channel current variable",
			&ChanBase::getIk
		);

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	// IkOut SrcFinfo defined above.

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	
///////////////////////////////////////////////////////
	static Finfo* ChanBaseFinfos[] =
	{
		&channel,			// Shared
		&ghk,				// Shared
		&Gbar,				// Value
		&Ek,				// Value
		&Gk,				// Value
		&Ik,				// ReadOnlyValue
		&IkOut,				// Src
	};
	
	static string doc[] =
	{
		"Name", "ChanBase",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "ChanBase: Base class for assorted ion channels."
		"Presents a common interface for all of them. ",
	};

	static Cinfo ChanBaseCinfo(
		"ChanBase",
		Neutral::initCinfo(),
		ChanBaseFinfos,
		sizeof( ChanBaseFinfos )/sizeof(Finfo *),
		new Dinfo< ChanBase >()
	);

	return &ChanBaseCinfo;
}

static const Cinfo* chanBaseCinfo = ChanBase::initCinfo();
//////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////
ChanBase::ChanBase()
			:
			Vm_( 0.0 ),
			Gbar_( 0.0 ), Ek_( 0.0 ),
			Gk_( 0.0 ), Ik_( 0.0 ),
			g_( 0.0 )
{
	;
}

ChanBase::~ChanBase()
{;}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void ChanBase::setGbar( double Gbar )
{
	Gbar_ = Gbar;
}
double ChanBase::getGbar() const
{
	return Gbar_;
}

void ChanBase::setEk( double Ek )
{
	Ek_ = Ek;
}
double ChanBase::getEk() const
{
	return Ek_;
}

void ChanBase::setGk( double Gk )
{
	Gk_ = Gk;
}
double ChanBase::getGk() const
{
	return Gk_;
}

double ChanBase::getIk() const
{
	return Ik_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ChanBase::handleVm( double Vm )
{
	Vm_ = Vm;
}
