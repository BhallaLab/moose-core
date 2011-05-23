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
#include "SynHandler.h"

/// This is bigger than # synapses for any neuron I know of: should suffice.
const unsigned int SynHandler::MAX_SYNAPSES = 1000000;

const Cinfo* SynHandler::initCinfo()
{
		static ValueFinfo< SynHandler, unsigned int > numSynapses(
			"numSynapses",
			"Number of synapses on SynHandler",
			&SynHandler::setNumSynapses,
			&SynHandler::getNumSynapses
		);

		//////////////////////////////////////////////////////////////
		// FieldElementFinfo definition for Synapses
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< SynHandler, Synapse > synFinfo( "synapse",
			"Sets up field Elements for synapse",
			Synapse::initCinfo(),
			&SynHandler::getSynapse,
			&SynHandler::setNumSynapses,
			&SynHandler::getNumSynapses
		);

	static Finfo* synHandlerFinfos[] = {
		&numSynapses,	// Value
		&synFinfo		// FieldElementFinfo for synapses.
	};

	static Cinfo synHandlerCinfo (
		"SynHandler",
		Neutral::initCinfo(),
		synHandlerFinfos,
		sizeof( synHandlerFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SynHandler >()
	);

	return &synHandlerCinfo;
}

static const Cinfo* synHandlerCinfo = SynHandler::initCinfo();

SynHandler::SynHandler()
{ ; }

SynHandler::~SynHandler()
{ ; }

/**
 * Inserts an event into the pendingEvents queue for spikes.
 * Note that this function lives on the Element managing the Synapses,
 * and gets redirected to the SynHandler.
 * This is called by UpFunc1< double >
 */
void SynHandler::addSpike( DataId index, const double time )
{
	this->innerAddSpike( index, time );
}

void SynHandler::innerAddSpike( DataId index, const double time )
{
	cout << "Warning: SynHandler::innerAddSpike: Should not get here\n";
}


void SynHandler::setNumSynapses( const unsigned int v )
{
	assert( v < MAX_SYNAPSES );
	synapses_.resize( v );
	// threadReduce< unsigned int >( er, getMaxNumSynapses, v );
}

unsigned int SynHandler::getNumSynapses() const
{
	return synapses_.size();
}

Synapse* SynHandler::getSynapse( unsigned int i )
{
	static Synapse dummy;
	if ( i < synapses_.size() )
		return &synapses_[i];
	cout << "Warning: SynHandler::getSynapse: index: " << i <<
		" is out of range: " << synapses_.size() << endl;
	return &dummy;
}
