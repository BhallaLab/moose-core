/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"
#include "HSolve.h"
#include "../biophysics/Compartment.h"
#include "ZombieCompartment.h"

const Cinfo* HSolve::initCinfo()
{
	static DestFinfo process(
		"process",
		"Handles 'process' call: Solver advances by one time-step.",
		new ProcOpFunc< HSolve >( &HSolve::process )
	);
	
	static DestFinfo reinit(
		"reinit",
		"Handles 'reinit' call: Solver reads in model.",
		new ProcOpFunc< HSolve >( &HSolve::reinit )
	);
	
	static Finfo* processShared[] =
	{
		&process,
		&reinit
	};
	
	static SharedFinfo proc(
		"proc",
		"Handles 'reinit' and 'process' calls from a clock.",
		processShared,
		sizeof( processShared ) / sizeof( Finfo* )
	);
	
	static ValueFinfo< HSolve, Id > seed(
		"seed",
		"Use this field to specify path to a 'seed' compartment, that is, "
		"any compartment within a neuron. The HSolve object uses this seed as "
		"a handle to discover the rest of the neuronal model, which means all "
		"the remaining compartments, channels, synapses, etc.",
		&HSolve::setSeed,
		&HSolve::getSeed
	);
	
	static ValueFinfo< HSolve, string > path(
		"path",
		"Specifies the path containing a compartmental model to be taken over.",
		&HSolve::setPath,
		&HSolve::getPath
	);
	
	static ValueFinfo< HSolve, int > caAdvance(
		"caAdvance",
		"This flag determines how current flowing into a calcium pool "
		"is computed. A value of 0 means that the membrane potential at the "
		"beginning of the time-step is used for the calculation. This is how "
		"GENESIS does its computations. A value of 1 means the membrane "
		"potential at the middle of the time-step is used. This is the "
		"correct way of integration, and is the default way.",
		&HSolve::setCaAdvance,
		&HSolve::getCaAdvance
	);
	
	static ValueFinfo< HSolve, int > vDiv(
		"vDiv",
		"Specifies number of divisions for lookup tables of voltage-sensitive "
		"channels.",
		&HSolve::setVDiv,
		&HSolve::getVDiv
	);
	
	static ValueFinfo< HSolve, double > vMin(
		"vMin",
		"Specifies the lower bound for lookup tables of voltage-sensitive "
		"channels. Default is to automatically decide based on the tables of "
		"the channels that the solver reads in.",
		&HSolve::setVMin,
		&HSolve::getVMin
	);
	
	static ValueFinfo< HSolve, double > vMax(
		"vMax",
		"Specifies the upper bound for lookup tables of voltage-sensitive "
		"channels. Default is to automatically decide based on the tables of "
		"the channels that the solver reads in.",
		&HSolve::setVMax,
		&HSolve::getVMax
	);
	
	static ValueFinfo< HSolve, int > caDiv(
		"caDiv",
		"Specifies number of divisions for lookup tables of calcium-sensitive "
		"channels.",
		&HSolve::setCaDiv,
		&HSolve::getCaDiv
	);
	
	static ValueFinfo< HSolve, double > caMin(
		"caMin",
		"Specifies the lower bound for lookup tables of calcium-sensitive "
		"channels. Default is to automatically decide based on the tables of "
		"the channels that the solver reads in.",
		&HSolve::setCaMin,
		&HSolve::getCaMin
	);
	
	static ValueFinfo< HSolve, double > caMax(
		"caMax",
		"Specifies the upper bound for lookup tables of calcium-sensitive "
		"channels. Default is to automatically decide based on the tables of "
		"the channels that the solver reads in.",
		&HSolve::setCaMax,
		&HSolve::getCaMax
	);
	
	static Finfo* hsolveFinfos[] = 
	{
		&seed,              // Value
		&caAdvance,         // Value
		&vDiv,              // Value
		&vMin,              // Value
		&vMax,              // Value
		&caDiv,             // Value
		&caMin,             // Value
		&caMax,             // Value
		&proc,              // Shared
	};
	
	static string doc[] =
	{
		"Name",             "HSolve",
		"Author",           "Niraj Dudani, 2007, NCBS",
		"Description",      "HSolve: Hines solver, for solving "
		                    "branching neuron models.",
	};
	
	static Cinfo hsolveCinfo(
		"HSolve",
		Neutral::initCinfo(),
		hsolveFinfos,
		sizeof( hsolveFinfos ) / sizeof( Finfo* ),
		new Dinfo< HSolve >()
	);
	
	return &hsolveCinfo;
}

static const Cinfo* hsolveCinfo = HSolve::initCinfo();

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HSolve::process( const Eref& hsolve, ProcPtr p )
{
	this->HSolveActive::step( p );
}

void HSolve::reinit( const Eref& hsolve, ProcPtr p )
{
	if ( seed_ == Id() )
		return;
	
	// Setup solver.
	this->HSolveActive::setup( seed_, p->dt );
	
	zombify( hsolve );
	mapIds();
}

void HSolve::zombify( Eref hsolve ) const
{
	vector< Id >::const_iterator i;
	for ( i = compartmentId_.begin(); i != compartmentId_.end(); ++i )
		ZombieCompartment::zombify( hsolve.element(), i->eref().element() );
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void HSolve::setSeed( Id seed )
{
	if ( seed()->cinfo()->name() != "Compartment" ) {
		cerr << "Error: HSolve::setSeed(): Seed object '" << seed.path()
		     << "' is not of type 'Compartment'." << endl;
		return;
	}
	
	seed_ = seed;
}

Id HSolve::getSeed() const
{
	return seed_;
}

void HSolve::setPath( string path )
{
	;
}

string HSolve::getPath() const
{
	return "";
}

void HSolve::setCaAdvance( int caAdvance )
{
	if ( caAdvance != 0 && caAdvance != 1 ) {
		cout << "Error: HSolve: caAdvance should be either 0 or 1.\n";
		return;
	}
	
	caAdvance_ = caAdvance;
}

int HSolve::getCaAdvance() const
{
	return caAdvance_;
}

void HSolve::setVDiv( int vDiv )
{
	vDiv_ = vDiv;
}

int HSolve::getVDiv() const
{
	return vDiv_;
}

void HSolve::setVMin( double vMin )
{
	vMin_ = vMin;
}

double HSolve::getVMin() const
{
	return vMin_;
}

void HSolve::setVMax( double vMax )
{
	vMax_ = vMax;
}

double HSolve::getVMax() const
{
	return vMax_;
}

void HSolve::setCaDiv( int caDiv )
{
	caDiv_ = caDiv;
}

int HSolve::getCaDiv() const
{
	return caDiv_;
}

void HSolve::setCaMin( double caMin )
{
	caMin_ = caMin;
}

double HSolve::getCaMin() const
{
	return caMin_;
}

void HSolve::setCaMax( double caMax )
{
	caMax_ = caMax;
}

double HSolve::getCaMax() const
{
	return caMax_;
}
