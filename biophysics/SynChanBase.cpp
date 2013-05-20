/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <queue>
#include "header.h"
#include "Synapse.h"
#include "SynBase.h"
#include "ChanBase.h"
#include "SynChanBase.h"

static SrcFinfo1< double >* permeability()
{
	static SrcFinfo1< double > permeability( "permeability", 
		"Conductance term going out to GHK object" );
	return &permeability;
}

static SrcFinfo2< double, double >* channelOut()
{
	static SrcFinfo2< double, double > channelOut( "channelOut", 
		"Sends channel variables Gk and Ek to compartment" );
	return &channelOut;
}

static SrcFinfo1< double >* IkOut()
{
	static SrcFinfo1< double > IkOut( "IkOut", 
		"Channel current. This message typically goes to concen"
		"objects that keep track of ion concentration." );
	return &IkOut;
}


/**
 * These are the fields carried over from ChanBase: for any object
 * managing synapses. Duplicated here as a hack to accomplish double
 * inheritance in the MOOSE framework. Must keep the two sets of 
 * fields identical.
 */

const Cinfo* SynChanBase::initCinfo()
{
	/////////////////////////////////////////////////////////////////////
	// Shared messages
	/////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////
	/// ChannelOut SrcFinfo defined above.
	static DestFinfo Vm( "Vm", 
		"Handles Vm message coming in from compartment",
		new OpFunc1< SynChanBase, double >( &SynChanBase::handleVm ) );

	static Finfo* channelShared[] =
	{
		channelOut(), &Vm
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
		&Vm, permeability()
	};
	static SharedFinfo ghk( "ghk", 
		"Message to Goldman-Hodgkin-Katz object",
		ghkShared, sizeof( ghkShared ) / sizeof( Finfo* ) );

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////

		static ValueFinfo< SynChanBase, double > Gbar( "Gbar",
			"Maximal channel conductance",
			&SynChanBase::setGbar,
			&SynChanBase::getGbar
		);
		static ValueFinfo< SynChanBase, double > Ek( "Ek", 
			"Reversal potential of channel",
			&SynChanBase::setEk,
			&SynChanBase::getEk
		);
		static ValueFinfo< SynChanBase, double > Gk( "Gk",
			"Channel conductance variable",
			&SynChanBase::setGk,
			&SynChanBase::getGk
		);
		static ReadOnlyValueFinfo< SynChanBase, double > Ik( "Ik",
			"Channel current variable",
			&SynChanBase::getIk
		);

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	// IkOut SrcFinfo defined above.

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	
///////////////////////////////////////////////////////
	static Finfo* SynChanBaseFinfos[] =
	{
		&channel,			// Shared
		&ghk,				// Shared
		&Gbar,				// Value
		&Ek,				// Value
		&Gk,				// Value
		&Ik,				// ReadOnlyValue
		IkOut(),				// Src
	};
	
	static string doc[] =
	{
		"Name", "SynChanBase",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "SynChanBase: Base class for assorted ion channels."
		"Presents a common interface for all of them. ",
	};

	static Cinfo SynChanBaseCinfo(
		"SynChanBase",
		SynBase::initCinfo(),
		SynChanBaseFinfos,
		sizeof( SynChanBaseFinfos )/sizeof(Finfo *),
		new Dinfo< SynChanBase >()
	);

	return &SynChanBaseCinfo;
}

static const Cinfo* synChanBaseCinfo = SynChanBase::initCinfo();
//////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////

SynChanBase::SynChanBase()
{ ; }

SynChanBase::~SynChanBase()
{ ; }

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void SynChanBase::setGbar( double Gbar )
{
	innerSetGbar( Gbar );
}
void SynChanBase::innerSetGbar( double Gbar ) // Virtual func
{
	cb.setGbar( Gbar );
}
double SynChanBase::getGbar() const
{
	return cb.getGbar();
}

void SynChanBase::setEk( double Ek )
{
	cb.setEk( Ek );
}
double SynChanBase::getEk() const
{
	return cb.getEk();
}

void SynChanBase::setGk( double Gk )
{
	cb.setGk( Gk );
}
double SynChanBase::getGk() const
{
	return cb.getGk();
}

void SynChanBase::setIk( double Ik )
{
	cb.setIk( Ik );
}
double SynChanBase::getIk() const
{
	return cb.getIk();
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void SynChanBase::handleVm( double Vm )
{
	cb.handleVm( Vm );
}

///////////////////////////////////////////////////
// Looks like a dest function, but it is only called
// from the child class. Sends out various messages.
///////////////////////////////////////////////////

void SynChanBase::process(  const Eref& e, const ProcPtr info )
{
	channelOut()->send( e, info->threadIndexInGroup, cb.getGk(), cb.getEk() );
	// This is used if the channel connects up to a conc pool and
	// handles influx of ions giving rise to a concentration change.
	IkOut()->send( e, info->threadIndexInGroup, cb.getIk() );
	// Needed by GHK-type objects
	permeability()->send( e, info->threadIndexInGroup, cb.getGk() );
}


void SynChanBase::reinit(  const Eref& e, const ProcPtr info )
{
	channelOut()->send( e, info->threadIndexInGroup, cb.getGk(), cb.getEk() );
	// Needed by GHK-type objects
	permeability()->send( e, info->threadIndexInGroup, cb.getGk() );
}

void SynChanBase::updateIk()
{
	cb.updateIk();
}

double SynChanBase::getVm() const
{
	return cb.getVm();
}
