/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Stats.h"

const Cinfo* Stats::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< Stats, double > mean(
			"mean",
			"Mean of all sampled values.",
			&Stats::getMean
		);
		static ReadOnlyValueFinfo< Stats, double > sdev(
			"sdev",
			"Standard Deviation of all sampled values.",
			&Stats::getSdev
		);
		static ReadOnlyValueFinfo< Stats, double > sum(
			"sum",
			"Sum of all sampled values.",
			&Stats::getSum
		);
		static ReadOnlyValueFinfo< Stats, unsigned int > num(
			"num",
			"Number of all sampled values.",
			&Stats::getNum
		);
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Stats >( &Stats::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< Stats >( &Stats::reinit ) );

		//////////////////////////////////////////////////////////////
		// SharedFinfo Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* statsFinfos[] = {
		&mean,	// ReadOnlyValue
		&sdev,	// ReadOnlyValue
		&sum,	// ReadOnlyValue
		&num,	// ReadOnlyValue
		&process,		// DestFinfo
		&reinit,		// DestFinfo
		&proc		// SharedFinfo
	};

	static Dinfo< Stats > dinfo;
	static Cinfo statsCinfo (
		"Stats",
		Neutral::initCinfo(),
		statsFinfos,
		sizeof( statsFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &statsCinfo;
}

static const Cinfo* statsCinfo = Stats::initCinfo();

///////////////////////////////////////////////////////////////////////////
// Inner class funcs
///////////////////////////////////////////////////////////////////////////

Stats::Stats()
	: 
	mean_( 0.0 ), sdev_( 0.0 ), sum_( 0.0 ), num_( 0 )
{
	;
}

///////////////////////////////////////////////////////////////////////////
// Process stuff.
///////////////////////////////////////////////////////////////////////////

void Stats::process( const Eref& e, ProcPtr p )
{
	;
}

void Stats::reinit( const Eref& e, ProcPtr p )
{
	mean_ = 0.0;
	sdev_ = 0.0;
	sum_ = 0.0;
	num_ = 0;
}
///////////////////////////////////////////////////////////////////////////
// DestFinfos
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// Fields
///////////////////////////////////////////////////////////////////////////

double Stats::getMean() const
{
	return mean_;
}

double Stats::getSdev() const
{
	return sdev_;
}

double Stats::getSum() const
{
	return sum_;
}

unsigned int Stats::getNum() const
{
	return num_;
}
