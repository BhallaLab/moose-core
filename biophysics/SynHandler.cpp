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
#include "SpikeRingBuffer.h"
#include "Synapse.h"
#include "SynHandler.h"

/// This is bigger than # synapses for any neuron I know of: should suffice.
const unsigned int SynHandler::MAX_SYNAPSES = 1000000;

/**
 * These are the base set of fields for any object managing synapses.
 * Note that these are duplicated in SynChanBase: if you change anything
 * here it must also be reflected there.
 */
const Cinfo* SynHandler::initCinfo()
{
	static ValueFinfo< SynHandler, unsigned int > numSynapses(
		"numSynapses",
		"Number of synapses on SynHandler. Duplicate field for num_synapse",
		&SynHandler::setNumSynapses,
		&SynHandler::getNumSynapses
	);
	//////////////////////////////////////////////////////////////////////
	static FieldElementFinfo< SynHandler, Synapse > synFinfo( 
		"synapse",
		"Sets up field Elements for synapse",
		Synapse::initCinfo(),
		&SynHandler::getSynapse,
		&SynHandler::setNumSynapses,
		&SynHandler::getNumSynapses
	);

	//////////////////////////////////////////////////////////////////////
	static Finfo* synHandlerFinfos[] = {
		&numSynapses,	// Value
		&synFinfo,		// FieldElement
	};

	static Dinfo< SynHandler > dinfo;
	static Cinfo synHandlerCinfo (
		"SynHandler",
		Neutral::initCinfo(),
		synHandlerFinfos,
		sizeof( synHandlerFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &synHandlerCinfo;
}

static const Cinfo* synHandlerCinfo = SynHandler::initCinfo();

SynHandler::SynHandler()
{ ; }

SynHandler::~SynHandler()
{ ; }

void SynHandler::setNumSynapses( const unsigned int v )
{
	assert( v < MAX_SYNAPSES );
	unsigned int prevSize = synapses_.size();
	synapses_.resize( v );
	for ( unsigned int i = prevSize; i < v; ++i )
		synapses_[i].setBuffer( &buf_ );
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

void SynHandler::reinitBuffer( double dt )
{
		buf_.reinit( dt );
}

double SynHandler::popBuffer()
{
	return buf_.pop();
}

unsigned int SynHandler::addSynapse()
{
	synapses_.resize( synapses_.size() + 1 );
	return synapses_.size() - 1;
}


void SynHandler::dropSynapse( unsigned int msgLookup )
{
	assert( msgLookup < synapses_.size() );
	synapses_[msgLookup].setWeight( -1.0 );
}
