/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <fstream>
#include "TableBase.h"
#include "Table.h"

static SrcFinfo1< vector< double >* > *requestOut() {
	static SrcFinfo1< vector< double >* > requestOut(
			"requestOut",
			"Sends request for a field to target object"
			);
	return &requestOut;
}

static DestFinfo *handleInput() {
	static DestFinfo input( "input",
		"Fills data into table. Also handles data sent back following request",
			new OpFunc1< Table, double >( &Table::input )
			);
	return &input;
}

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

		static DestFinfo spike( "spike",
			"Fills spike timings into the Table. Signal has to exceed thresh",
			new OpFunc1< Table, double >( &Table::spike ) );

		static DestFinfo process( "process",
			"Handles process call, updates internal time stamp.",
			new ProcOpFunc< Table >( &Table::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call.",
			new ProcOpFunc< Table >( &Table::reinit ) );
		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

		//////////////////////////////////////////////////////////////
		// Field Element for the vector data
		// Use a limit of 2^20 entries for the tables, about 1 million.
		//////////////////////////////////////////////////////////////

	static Finfo* tableFinfos[] = {
		&threshold,		// Value
		handleInput(),		// DestFinfo
		&spike,			// DestFinfo
		requestOut(),		// SrcFinfo
		&proc,			// SharedFinfo
	};

	static string doc[] = 
	{
			"Name", "Table",
			"Author", "Upi Bhalla",
			"Description", 
			"Table for accumulating data values, or spike timings. "
			"Can either receive incoming doubles, or can explicitly "
			"request values from fields provided they are doubles. "
			"The latter mode of use is preferable if you wish to have "
			"independent control of how often you sample from the output "
			"variable. \n"
			"Typically used for storing simulation output into memory. \n"
			"There are two functionally identical variants of the Table "
			"class: Table and Table2. Their only difference is that the "
			"default scheduling of the Table (Clock Tick 8, dt = 0.1 ms ) "
			"makes it suitable for "
			"tracking electrical compartmental models of neurons and "
			"networks. \n"
			"Table2 (Clock Tick 18, dt = 1.0 s) is good for tracking "
			"biochemical signaling pathway outputs. \n"
			"These are just the default values and Tables can be assigned"
			" to any Clock Tick and timestep in the usual manner.",
	};
	static Dinfo< Table > dinfo;
	static Cinfo tableCinfo (
		"Table",
		TableBase::initCinfo(),
		tableFinfos,
		sizeof( tableFinfos ) / sizeof ( Finfo* ),
		&dinfo,
		doc,
		sizeof( doc ) / sizeof( string )
	);
	static string doc2[] = {doc[0], "Table2", doc[2], doc[3], 
			doc[4], doc[5] };
	doc2[1] = "Table2";
	static Cinfo table2Cinfo (
		"Table2",
		TableBase::initCinfo(),
		tableFinfos,
		sizeof( tableFinfos ) / sizeof ( Finfo* ),
		&dinfo,
		doc2,
		sizeof( doc2 ) / sizeof( string )
	);

	return &tableCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* tableCinfo = Table::initCinfo();

Table::Table()
	: threshold_( 0.0 ), lastTime_( 0.0 ), input_( 0.0 )
{
	;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Table::process( const Eref& e, ProcPtr p )
{
	lastTime_ = p->currTime;
	vector< double > ret;
	requestOut()->send( e, &ret );
	vec().insert( vec().end(), ret.begin(), ret.end() );
}

void Table::reinit( const Eref& e, ProcPtr p )
{
	input_ = 0.0;
	vec().resize( 0 );
	lastTime_ = 0;
	// cout << "tabReinit on :" << p->groupId << ":" << p->threadIndexInGroup << endl << flush;
	// requestOut()->send( e, handleInput()->getFid());
	vector< double > ret;
	requestOut()->send( e, &ret );
	vec().insert( vec().end(), ret.begin(), ret.end() );
}

//////////////////////////////////////////////////////////////
// Used to handle direct messages into the table, or 
// returned plot data from queried objects.
//////////////////////////////////////////////////////////////
void Table::input( double v )
{
	vec().push_back( v );
}

void Table::spike( double v )
{
	if ( v > threshold_ )
		vec().push_back( lastTime_ );
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

