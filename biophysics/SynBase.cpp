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

/// This is bigger than # synapses for any neuron I know of: should suffice.
const unsigned int SynBase::MAX_SYNAPSES = 1000000;

/**
 * These are the base set of fields for any object managing synapses.
 * Note that these are duplicated in SynChanBase: if you change anything
 * here it must also be reflected there.
 */
const Cinfo* SynBase::initCinfo()
{
		static ValueFinfo< SynBase, unsigned int > numSynapses(
			"numSynapses",
			"Number of synapses on SynBase",
			&SynBase::setNumSynapses,
			&SynBase::getNumSynapses
		);

		//////////////////////////////////////////////////////////////
		// FieldElementFinfo definition for Synapses
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< SynBase, Synapse > synFinfo( "synapse",
			"Sets up field Elements for synapse",
			Synapse::initCinfo(),
			&SynBase::getSynapse,
			&SynBase::setNumSynapses,
			&SynBase::getNumSynapses
		);

	static Finfo* synBaseFinfos[] = {
		&numSynapses,	// Value
		&synFinfo		// FieldElementFinfo for synapses.
	};

	static Cinfo synBaseCinfo (
		"SynBase",
		Neutral::initCinfo(),
		synBaseFinfos,
		sizeof( synBaseFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SynBase >()
	);

	return &synBaseCinfo;
}

static const Cinfo* synBaseCinfo = SynBase::initCinfo();

SynBase::SynBase()
{ ; }

SynBase::~SynBase()
{ ; }

/**
 * Inserts an event into the pendingEvents queue for spikes.
 * Note that this function lives on the Element managing the Synapses,
 * and gets redirected to the SynBase.
 * This is called by UpFunc1< double >
 */
void SynBase::addSpike( unsigned int index, const double time )
{
	this->innerAddSpike( index, time );
}

void SynBase::innerAddSpike( DataId index, const double time )
{
	cout << "Warning: SynBase::innerAddSpike: Should not get here\n";
}


void SynBase::setNumSynapses( const unsigned int v )
{
	assert( v < MAX_SYNAPSES );
	synapses_.resize( v );
	// threadReduce< unsigned int >( er, getMaxNumSynapses, v );
}

unsigned int SynBase::getNumSynapses() const
{
	return synapses_.size();
}

Synapse* SynBase::getSynapse( unsigned int i )
{
	static Synapse dummy;
	if ( i < synapses_.size() )
		return &synapses_[i];
	cout << "Warning: SynBase::getSynapse: index: " << i <<
		" is out of range: " << synapses_.size() << endl;
	return &dummy;
}
