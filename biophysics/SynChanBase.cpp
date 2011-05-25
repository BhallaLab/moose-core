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

/**
 * These are the fields carried over from SynBase: for any object
 * managing synapses. Duplicated here as a hack to accomplish double
 * inheritance in the MOOSE framework. Must keep the two sets of 
 * fields identical.
 */
const Cinfo* SynChanBase::initCinfo()
{
		static ValueFinfo< SynChanBase, unsigned int > numSynapses(
			"numSynapses",
			"Number of synapses on object",
			&SynChanBase::setNumSynapses,
			&SynChanBase::getNumSynapses
		);

		//////////////////////////////////////////////////////////////
		// FieldElementFinfo definition for Synapses
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< SynChanBase, Synapse > synFinfo( "synapse",
			"Sets up field Elements for synapse",
			Synapse::initCinfo(),
			&SynChanBase::getSynapse,
			&SynChanBase::setNumSynapses,
			&SynChanBase::getNumSynapses
		);

	static Finfo* synChanBaseFinfos[] = {
		&numSynapses,	// Value
		&synFinfo		// FieldElementFinfo for synapses.
	};

	static Cinfo synChanBaseCinfo (
		"SynChanBase",
		ChanBase::initCinfo(),
		synChanBaseFinfos,
		sizeof( synChanBaseFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SynChanBase >()
	);

	return &synChanBaseCinfo;
}

static const Cinfo* synChanBaseCinfo = SynChanBase::initCinfo();

SynChanBase::SynChanBase()
{ ; }

SynChanBase::~SynChanBase()
{ ; }


/*
void SynChanBase::innerAddSpike( DataId index, const double time )
{
	cout << "Warning: SynChanBase::innerAddSpike: Should not get here\n";
}


void SynChanBase::setNumSynapses( const unsigned int v )
{
	assert( v < MAX_SYNAPSES );
	synapses_.resize( v );
	// threadReduce< unsigned int >( er, getMaxNumSynapses, v );
}

unsigned int SynChanBase::getNumSynapses() const
{
	return synapses_.size();
}

Synapse* SynChanBase::getSynapse( unsigned int i )
{
	static Synapse dummy;
	if ( i < synapses_.size() )
		return &synapses_[i];
	cout << "Warning: SynChanBase::getSynapse: index: " << i <<
		" is out of range: " << synapses_.size() << endl;
	return &dummy;
}
*/
