/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"

#include "tryConn.h"

// Here's all the Sender class stuff:

SrcFinfo1< double >* Sender::XOut() {
	static SrcFinfo1< double > XOut( "XOut", 
									 "Value of random field X" );
		// This XOut is an object, not a function declaration!
	return &XOut;
}

static const Cinfo* Sender::initCinfo()
{
	static DestFinfo process( "process", 
		"Handles process call",
		new ProcOpFunc< Sender >( &Sender::process ) );
	static DestFinfo reinit( "reinit", 
		"Handles reinit call",
		new ProcOpFunc< Sender >( &Sender::reinit ) );

	static Finfo* processShared[] =
	{
		&process, &reinit
	};

	static SharedFinfo proc( "proc", 
		"Shared message to receive Process message from scheduler",
		processShared, sizeof( processShared ) / sizeof( Finfo* ) );

	static ValueFinfo< Sender, double > X( "X",
		"Random value field for testing",
        &Sender::setX,
		&Sender::getX
	);
	
	static Finfo* SenderFinfos[] =
	{
		&proc,		// Shared
		&X,			// Value
		XOut(),		// Src
	};
	
	static string doc[] =
	{
		"Name", "Sender",
		"Author", "Praveen Venkatesh, 2013, NCBS",
		"Description", "Sender: A class for trying out creation of a C++ class"
					   "accessible in python",
	};

	static Cinfo SenderCinfo(
		"Sender",
		Neutral::initCinfo(),
		SenderFinfos,
		sizeof( SenderFinfos )/sizeof(Finfo *),
		new Dinfo< Sender >()
	);
	
	return &SenderCinfo;
}

// This statement does the all-important task of making this class visible to
// pymoose. pymoose automatically searches for files that have Cinfo pointers
// declared in them and uses the information present in these Cinfo's to
// create the corresponding python classes.
static const Cinfo* tryConnCinfo = Sender::initCinfo();

Sender::Sender()
{
	X_ = 10;		// default
}

double Sender::getX() const
{
	return X_;
}

void Sender::setX( double x )
{
	X_ = x;
}

void Sender::reinit( const Eref& e, ProcPtr p )
{
	XOut()->send( e, p->threadIndexInGroup, X_ );
}

void Sender::process( const Eref& e, ProcPtr p )
{
	// Let's say that X gets multiplied by 2 every time this thing is processed
	// I mean, _something_ should happen every time step, right?
	X_ *= 2;
	XOut()->send( e, p->threadIndexInGroup, X_ );
}

///////////////////////////////////////////////////////////////////////////////

// Now for the Receiver class...

static const Cinfo* Receiver::initCinfo()
{
	static DestFinfo handleX( "handleX", 
		"Prints out X as and when it is received",
		new OpFunc1< Receiver, double >( &Receiver::handleX )
	);
	
	static Finfo* ReceiverFinfos[] =
	{
		&handleX,	// Dest
	};

	static string doc[] =
	{
		"Name", "Receiver",
		"Author", "Praveen Venkatesh, 2013, NCBS",
		"Description", "Receiver: Acts as the receiving end for whenever Sender"
					   "sends out X values. Needs to be connected by the user.",
	};

	static Cinfo ReceiverCinfo(
		"Receiver",
		Neutral::initCinfo(),
		ReceiverFinfos,
		sizeof( ReceiverFinfos ) / sizeof(Finfo *),
		new Dinfo< Receiver >()
	);

	return &ReceiverCinfo;
}

// Once again, this statement allows pymoose to find the Receiver class during
// compile time.
static const Cinfo* ReceiverCinfo = Receiver::initCinfo();

void Receiver::handleX( double X )
{
	cout<<X<<endl;
}
