/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Sched.h"

const Cinfo* initSched()
{
	static Finfo* schedFinfos[] =
	{
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	//	new NSrc1Finfo< ProcInfo >(
	//		"processOut", &SchedWrapper::getProcessSrc, "" ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "start", Ftype2< string, string >::global(), 
			RFCAST( &SchedWrapper::startFunc ),
			),
		new Dest1Finfo< string >(
			"stopIn", &SchedWrapper::stopFunc,
			&SchedWrapper::getStopInConn, "" ),
	};
}

const Cinfo SchedWrapper::cinfo_(
	"Sched",
	"Upinder S. Bhalla, Nov 2005, NCBS",
	"Sched: Scheduler root class. Controls multiple jobs for I/O and processing.",
	"Neutral",
	SchedWrapper::fieldArray_,
	sizeof(SchedWrapper::fieldArray_)/sizeof(Finfo *),
	&SchedWrapper::create
);


///////////////////////////////////////////////////////
// Function definitions
///////////////////////////////////////////////////////

// All well and good, but I really should call the send from an
// event loop. This means that the ProcInfo should be on a synapse-like
// array at the msg src.
void SchedWrapper::startFuncLocal(
	const string& job, const string& shell )
{
	ProcInfoBase p( shell );
	Field j( job );
	if ( j.good() ) { // Assign value to it
		// Field s = cinfo()->field( "processOut" );
		Ftype1< ProcInfo >::set( j.getElement(), j.operator->(), &p );
	}
}
