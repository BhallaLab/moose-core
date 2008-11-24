/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Kintegrator.h"

const Cinfo* initKintegratorCinfo()
{
	static Finfo* integrateShared[] =
	{
		new SrcFinfo( "reinitSrc", Ftype0::global() ),
		new SrcFinfo( "integrateSrc",
			Ftype2< vector< double >* , double >::global() ),
		new DestFinfo( "allocate",
			Ftype1< vector< double >* >::global(),
			RFCAST( &Kintegrator::allocateFunc )
			),
	};

	static Finfo* processShared[] =
	{
		new DestFinfo( "process",
			Ftype1< ProcInfo >::global(),
			RFCAST( &Kintegrator::processFunc )),
		new DestFinfo( "reinit",
			Ftype1< ProcInfo >::global(),
			RFCAST( &Kintegrator::reinitFunc )),
	};

	static Finfo* kintegratorFinfos[] =
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		new ValueFinfo( "isInitiatilized", 
			ValueFtype1< bool >::global(),
			GFCAST( &Kintegrator::getIsInitialized ), 
			&dummyFunc
		),
		new ValueFinfo( "method", 
			ValueFtype1< string >::global(),
			GFCAST( &Kintegrator::getMethod ), 
			RFCAST( &Kintegrator::setMethod )
		),
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		new SharedFinfo( "integrate", integrateShared, 
				sizeof( integrateShared )/ sizeof( Finfo* ) ),
		new SharedFinfo( "process", processShared, 
				sizeof( processShared )/ sizeof( Finfo* ) ),
	};

	static string doc[] =
	{
		"Name", "Kintegrator",
		"Author", "Upinder S. Bhalla, June 2006, NCBS",
		"Description","Kintegrator: Kinetic Integrator base class for setting up numerical solvers. "
			       "This is currently set up to work only with the Stoich class, which represents "
			       "biochemical networks.The goal is to have a standard interface so different "
			       "solvers can work with different kinds of calculation.",
	};	
	static  Cinfo kintegratorCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		kintegratorFinfos,
		sizeof(kintegratorFinfos)/sizeof(Finfo *),
		ValueFtype1< Kintegrator >::global()
	);

	return &kintegratorCinfo;
}

static const Cinfo* kintegratorCinfo = initKintegratorCinfo();

static const Slot integrateSlot =
	initKintegratorCinfo()->getSlot( "integrate.integrateSrc" );
static const Slot reinitSlot =
	initKintegratorCinfo()->getSlot( "integrate.reinitSrc" );


///////////////////////////////////////////////////
// Basic class function definitions
///////////////////////////////////////////////////
Kintegrator::Kintegrator()
{
	isInitialized_ = 0;
	method_ = "Euler";
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

bool Kintegrator::getIsInitialized( Eref e )
{
	return static_cast< const Kintegrator* >( e.data() )->isInitialized_;
}

string Kintegrator::getMethod( Eref e )
{
	return static_cast< const Kintegrator* >( e.data() )->method_;
}
void Kintegrator::setMethod( const Conn* c, string method )
{
	static_cast< Kintegrator* >( c->data() )->innerSetMethod( method );
}

void Kintegrator::innerSetMethod( const string& method )
{
	method_ = method;
	cout << "in void Kintegrator::innerSetMethod( string method ) \n";
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Kintegrator::allocateFunc( const Conn* c, vector< double >* y )
{
	static_cast< Kintegrator* >( c->data() )->allocateFuncLocal( y );
}
void Kintegrator::allocateFuncLocal( vector< double >*  y )
{
			y_ = y;
			if ( !isInitialized_ || yprime_.size() != y->size() )
				yprime_.resize( y->size(), 0.0 );
			isInitialized_ = 1;
}

void Kintegrator::processFunc( const Conn* c, ProcInfo info )
{
	static_cast< Kintegrator* >( c->data() )->innerProcessFunc( 
		c->target(), info );
}

void Kintegrator::innerProcessFunc( Eref e, ProcInfo info )
{
		vector< double >::iterator i;
		vector< double >::const_iterator j = yprime_.begin();
		send2< vector< double >*, double >( e, integrateSlot, 
			&yprime_, info->dt_ );

		// Here we do the simple Euler method.
		for ( i = y_->begin(); i != y_->end(); i++ )
			*i += *j++;
		/*
		long currT = static_cast< long >( info->currTime_ );
		if ( info->currTime_ - currT < info->dt_ )
			cout << info->currTime_ << "    " << y_->front() << "\n";
			*/
}

void Kintegrator::reinitFunc( const Conn* c, ProcInfo info )
{
	send0( c->target(), reinitSlot );
}
