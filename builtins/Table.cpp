/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "TableEntry.h"
#include "Table.h"

static SrcFinfo1< double > output (
	"output",
	"Sends out single pass of data in vector"
);

static SrcFinfo1< double > outputLoop (
	"outputLoop",
	"Sends data in vector in a loop, repeating as often as run continues"
);

const Cinfo* Table::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Table, double > threshold(
			"threshold",
			"threshold used when Table acts as a buffer for spikes",
			&Table::setThreshold,
			&Table::getThreshold
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo input( "input",
			"Fills data into the Table.",
			new OpFunc1< Table, double >( &Table::input ) );

		static DestFinfo spike( "spike",
			"Fills spike timings into the Table. Signal has to exceed thresh",
			new OpFunc1< Table, double >( &Table::spike ) );

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// Field Element for the vector data
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< Table, double > tableEntryFinfo( 
			"table", 
			"Field Element for Table entries",
			TableEntry::initCinfo(),
			&Table::lookupVec,
			&Table::setVecSize,
			&Table::getVecSize
		);

	static Finfo* tableFinfos[] = {
		&threshold,		// Value
		&group,			// DestFinfo
		&input,			// DestFinfo
		&spike,			// DestFinfo
		&output,		// SrcFinfo
		&outputLoop,		// SrcFinfo
		&tableEntryFinfo,	// FieldElementFinfo
	};

	static Cinfo tableCinfo (
		"Table",
		Neutral::initCinfo(),
		tableFinfos,
		sizeof( tableFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Table >()
	);

	return &tableCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* tableCinfo = Table::initCinfo();

Table::Table()
	: threshold_( 0.0 ), lastTime_( 0.0 ), outputIndex_( 0 )
{
	;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Table::process( const ProcInfo* p, const Eref& e )
{
	lastTime_ = p->currTime;
	if ( vec_.size() == 0 ) {
		output.send( e, p, 0.0 );
		outputLoop.send( e, p, 0.0 );
		return;
	}

	if ( outputIndex_ < vec_.size() )
		output.send( e, p, vec_[ outputIndex_ ] );
	else
		output.send( e, p, vec_.back() );

	outputLoop.send( e, p, vec_[ outputIndex_ % vec_.size() ] );

	outputIndex_++;
}

void Table::reinit()
{
	outputIndex_ = 0;
	lastTime_ = 0;
}

void Table::input( double v )
{
	vec_.push_back( v );
}

void Table::spike( double v )
{
	if ( v > threshold_ )
		vec_.push_back( lastTime_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Table::setThreshold( double v )
{
	threshold_ = v;
}

double Table::getThreshold() const
{
	return threshold_;
}

//////////////////////////////////////////////////////////////
// Element Field Definitions
//////////////////////////////////////////////////////////////

double* Table::lookupVec( unsigned int index )
{
	if ( index < vec_.size() )
		return &( vec_[index] );
	cout << "Error: Table::lookupTableEntry: Index " << index << 
		" >= vector size " << vec_.size() << endl;
	return 0;
}

void Table::setVecSize( unsigned int num )
{
	assert( num < 1000 ); // Pretty unlikely upper limit
	vec_.resize( num );
}

unsigned int Table::getVecSize() const
{
	return vec_.size();
}
