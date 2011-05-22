/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <queue>
#include "header.h"
#include "Synapse.h"
#include "SynHandler.h"
#include "Dinfo.h"
#include "UpFunc.h"

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
			"Handles arriving spike messages, by redirecting up to parent "
			"SynHandler object",
			new UpFunc1< SynHandler, double >( &SynHandler::addSpike ) );

	static Finfo* synapseFinfos[] = {
		// Fields
		&weight,
		&delay,
		// DestFinfo
		&addSpike,
	};

	static Cinfo synapseCinfo (
		"Synapse",
		Neutral::initCinfo(),
		synapseFinfos,
		sizeof( synapseFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Synapse >()
	);

	return &synapseCinfo;
}

static const Cinfo* synapseCinfo = Synapse::initCinfo();

Synapse::Synapse()
	: weight_( 1.0 ), delay_( 0.0 )
{
	;
}

Synapse::Synapse( double w, double d ) 
	: weight_( w ), delay_( d )
{
	;
}

Synapse::Synapse( const Synapse& other, double time )
	: weight_( other.weight_ ), delay_( time + other.delay_ )
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
