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

	static LookupValueFinfo< SynBase, double, unsigned int > weight(
		"weight",
		"Synaptic weight",
		&SynBase::setWeight,
		&SynBase::getWeight,
	);

	static LookupValueFinfo< SynBase, double, unsigned int > delay(
		"delay",
		"Synaptic delay",
		&SynBase::setDelay,
		&SynBase::getDelay,
	);

	static DestFinfo addSpike( 
		"addSpike",
		"Handles arriving spike messages. "
		"Single argument specifies timestamp of action potential",
		new EpFunc1< SynBase, double >( &SynBase::addSpike )
	);


	//////////////////////////////////////////////////////////////////////
	static Finfo* synBaseFinfos[] = {
		&numSynapses,	// Value
		&weight			// LookupValue
		&delay			// LookupValue
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
void SynBase::addSpike( const Eref& e, const double time )
{
	this->innerAddSpike( e.fieldIndex(), time );
}

void SynBase::innerAddSpike( unsigned int index, const double time )
{
	cout << "Warning: SynBase::innerAddSpike: Should not get here\n";
	assert( 0 );
}


void SynBase::setNumSynapses( const unsigned int v )
{
	assert( v < MAX_SYNAPSES );
	synapses_.resize( v );
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

void SynBase::setWeight( unsigned int i, double weight ) 
{
	if ( i < synapses_.size() )
		synapses_[i].setWeight( weight );
}

void SynBase::setDelay( unsigned int i, double delay ) 
{
	if ( i < synapses_.size() )
		synapses_[i].setDelay( delay );
}


double SynBase::getWeight( unsigned int i ) const
{
	if ( i < synapses_.size() )
		return synapses_[i].getWeight();
	return 0.0;
}

double SynBase::getDelay( unsigned int i ) const
{
	if ( i < synapses_.size() )
		return synapses_[i].getDelay();
	return 0.0;
}
