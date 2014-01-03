/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SpikeRingBuffer.h"
#include "Synapse.h"
#include "SynHandler.h"

const Cinfo* Synapse::initCinfo()
{
		static ValueFinfo< Synapse, double > weight(
			"weight",
			"Synaptic weight",
			&Synapse::setWeight,
			&Synapse::getWeight
		);

		static ValueFinfo< Synapse, double > delay(
			"delay",
			"Axonal propagation delay to this synapse",
			&Synapse::setDelay,
			&Synapse::getDelay
		);

		static DestFinfo addSpike( "addSpike",
			"Handles arriving spike messages, inserts into event queue.",
			new EpFunc1< Synapse, double >( &Synapse::addSpike ) );

	static Finfo* synapseFinfos[] = {
		&weight,		// Field
		&delay,			// Field
		&addSpike,		// DestFinfo
	};

	static Dinfo< Synapse > dinfo;
	static Cinfo synapseCinfo (
		"Synapse",
		Neutral::initCinfo(),
		synapseFinfos,
		sizeof( synapseFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &synapseCinfo;
}

static const Cinfo* synapseCinfo = Synapse::initCinfo();

Synapse::Synapse()
	: weight_( 1.0 ), delay_( 0.0 ), buffer_( 0 )
{
	;
}

void Synapse::setWeight( const double v )
{
	weight_ = v;
}

void Synapse::setDelay( const double v )
{
	delay_ = v;
}

double Synapse::getWeight() const
{
	return weight_;
}

double Synapse::getDelay() const
{
	return delay_;
}

void Synapse::setBuffer( SpikeRingBuffer* buf )
{
	buffer_ = buf;
}


void Synapse::addSpike( const Eref& e, double time )
{
	static bool report = false;
	static unsigned int tgtDataIndex = 0;
	// static unsigned int tgtFieldIndex = 0;
	if ( report && e.dataIndex() == tgtDataIndex ) {
		cout << "	" << time << "," << e.fieldIndex();
	}
	buffer_->addSpike( time + delay_, weight_ );
}

/////////////////////////////////////////////////////////////
// Callbacks for message add/drop
/////////////////////////////////////////////////////////////

// static function, executed by the Synapse Element when a message is
// added to the Element. Expands the parent synapse array to fit.
void Synapse::addMsgCallback(
				const Eref& e, const string& finfoName, 
			    ObjId msg, unsigned int msgLookup )
{
	if ( finfoName == "addSpike" ) {
		ObjId pa = Neutral::parent( e );
		SynHandler* sh = reinterpret_cast< SynHandler* >( pa.data() );
		unsigned int synapseNumber = sh->addSynapse();
		SetGet2< unsigned int, unsigned int >::set( 
						msg, "fieldIndex", msgLookup, synapseNumber );
	}
}

// static function, executed by the Synapse Element when a message is
// dropped from the Element. Contracts the parent synapse array to fit.
// Typically the SynHandler won't resize, easier to just leave an
// unused entry. Could even reuse if a synapse is added later, but all
// this policy is independent of the Synapse class.
void Synapse::dropMsgCallback(
				const Eref& e, const string& finfoName, 
			    ObjId msg, unsigned int msgLookup )
{
	if ( finfoName == "addSpike" ) {
		ObjId pa = Neutral::parent( e );
		SynHandler* sh = reinterpret_cast< SynHandler* >( pa.data() );
		sh->dropSynapse( msgLookup );
	}
}

