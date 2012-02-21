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
#include "TableEntry.h"
#include "TableBase.h"
#include "Table.h"

static SrcFinfo1< FuncId > *requestData() {
	static SrcFinfo1< FuncId > requestData(
			"requestData",
			"Sends request for a field to target object"
			);
	return &requestData;
}

static DestFinfo *recvDataBuf() {
	static DestFinfo recvDataBuf( "recvData",
			"Handles data sent back following request",
			new OpFunc1< Table, PrepackedBuffer >( &Table::recvData )
			);
	return &recvDataBuf;
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

		static DestFinfo input( "input",
			"Fills data into the Table.",
			new OpFunc1< Table, double >( &Table::input ) );

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
		&input,			// DestFinfo
		&spike,			// DestFinfo
		recvDataBuf(),	// DestFinfo
		requestData(),		// SrcFinfo
		&proc,			// SharedFinfo
	};

	static Cinfo tableCinfo (
		"Table",
		TableBase::initCinfo(),
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
	// send out a request for data. This magically comes back in the
	// RecvDataBuf and is handled.
	requestData()->send( e, p->threadIndexInGroup, recvDataBuf()->getFid());
}

void Table::reinit( const Eref& e, ProcPtr p )
{
	input_ = 0.0;
	vec().resize( 0 );
	lastTime_ = 0;
	// cout << "tabReinit on :" << p->groupId << ":" << p->threadIndexInGroup << endl << flush;
	requestData()->send( e, p->threadIndexInGroup, recvDataBuf()->getFid());
}

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

//////////////////////////////////////////////////////////////
// Element Field Definitions
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
// Used to handle returned plot data from queried objects.
//////////////////////////////////////////////////////////////

void Table::recvData( PrepackedBuffer pb )
{
	assert ( pb.dataSize() == 1 );
	double ret = *reinterpret_cast< const double* >( pb.data() );

	vec().push_back( ret );
}

//////////////////////////////////////////////////////////////
