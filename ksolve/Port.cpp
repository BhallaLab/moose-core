/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"

static SrcFinfo1< vector< Id > > availableMols(
	"availableMols",
	"Sends out the full set of molecule Ids that are available for data transfer"
);

static SrcFinfo1< vector< Id > > matchedMols(
	"matchedMols",
	"Sends out the set of molecule Ids that match between both ports"
);

static SrcFinfo1< vector< double > > efflux(
	"efflux",
	"Molecule #s going out"
);


const Cinfo* Port::initCinfo()
{
		static ValueFinfo< Port, double > scaleOutRate(
			"scaleOutRate",
			"Scaling factor for outgoing rates. Applies to the RateTerms"
			"controlled by this port. Represents a diffusion related term,"
			"or the permeability of the port",
			&Port::setScaleOutRate,
			&Port::getScaleOutRate
		);

		static ReadOnlyValueFinfo< Port, unsigned int > inStart(
			"inStart",
			"Start index to S_ vector into which incoming molecules should add.",
			&Port::getInStart
		);

		static ReadOnlyValueFinfo< Port, unsigned int > inEnd(
			"inEnd",
			"End index to S_ vector into which incoming molecules should add.",
			&Port::getInEnd
		);

		static ReadOnlyValueFinfo< Port, unsigned int > outStart(
			"outStart",
			"Start index to S_ vector from where outgoing molecules come.",
			&Port::getOutStart
		);

		static ReadOnlyValueFinfo< Port, unsigned int > outEnd(
			"outEnd",
			"End index to S_ vector from where outgoing molecules come.",
			&Port::getOutEnd
		);

		static DestFinfo handleAvailableMols( "handleAvailableMols",
			"Handles list of all molecules that the other port cares about",
			new UpFunc1< Stoich, vector< Id > >( &Stoich::handleAvailableMols ) );

		static DestFinfo handleMatchedMols( "handleMatchedMols",
			"Handles list of matched molecules worked out by the other port",
			new UpFunc1< Stoich, vector< Id > >( &Stoich::handleMatchedMols ) );

		static DestFinfo influx( "influx",
			"Molecule #s coming back in",
			new UpFunc1< Stoich, vector< double > >( &Stoich::influx ) );

		////////////////////////////////////////////////////////////
		// SharedFinfo definitions
		////////////////////////////////////////////////////////////
		/*
		static Finfo* procShared[] = {
			&process, &reinit
		}
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);
		*/

		static Finfo* port1array[] = {
			&availableMols, &handleMatchedMols, &efflux, &influx
		};

		static SharedFinfo port1( "port1",
			"Shared message for port. This one initiates the request for"
			"setting up the communications between the ports"
			"The shared message also handles the runtime data transfer",
			port1array, sizeof( port1array ) / sizeof( const Finfo* )
		);

		static Finfo* port2array[] = {
			&handleAvailableMols, &matchedMols, &influx, &efflux, 
		};

		static SharedFinfo port2( "port2",
			"Shared message for port. This one responds to the request for"
			"setting up the communications between the ports"
			"The shared message also handles the runtime data transfer",
			port2array, sizeof( port2array ) / sizeof( const Finfo* )
		);

		////////////////////////////////////////////////////////////
	static Finfo* portFinfos[] = {
		// Fields
		&scaleOutRate,		// Value
		&inStart,			// ReadOnly Value
		&inEnd,				// ReadOnly Value
		&outStart,			// ReadOnly Value
		&outEnd,			// ReadOnly Value
		// &proc,				// SharedFinfo
		&port1,				// SharedFinfo
		&port2,				// SharedFinfo
	};

	static Cinfo portCinfo (
		"Port",
		Neutral::initCinfo(),
		portFinfos,
		sizeof( portFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Port >()
	);

	return &portCinfo;
}

static const Cinfo* portCinfo = Port::initCinfo();

Port::Port()
	: 
		inStart_( 0 ),
		inEnd_( 0 ),
		outStart_( 0 ),
		outEnd_( 0 ),
		scaleOutRate_( 1.0 ),
		parent_( 0 )
{
	;
}

Port::~Port()
{;}

void Port::setScaleOutRate( double v )
{
	scaleOutRate_ = v;
}

double Port::getScaleOutRate() const
{
	return scaleOutRate_;
}

unsigned int Port::getInStart() const
{
	return inStart_;
}

unsigned int Port::getInEnd() const
{
	return inEnd_;
}

unsigned int Port::getOutStart() const
{
	return outStart_;
}

unsigned int Port::getOutEnd() const
{
	return outEnd_;
}

