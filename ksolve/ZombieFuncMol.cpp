/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "StoichHeaders.h"
#include "ElementValueFinfo.h"
#include "DataHandlerWrapper.h"

#include "Mol.h"
#include "FuncMol.h"
#include "ZombieMol.h"
#include "ZombieFuncMol.h"
#include "ZombieSumFunc.h"

// Derived from ZombieMol.
const Cinfo* ZombieFuncMol::initCinfo()
{
	static DestFinfo input( "input",
		"Handles input to control value of n_",
		new OpFunc1< ZombieFuncMol, double >( &ZombieFuncMol::input ) );
	
	static Finfo* zombieFuncMolFinfos[] = {
		&input,             // DestFinfo
	};

	static Cinfo zombieFuncMolCinfo (
		"ZombieFuncMol",
		ZombieMol::initCinfo(),
		zombieFuncMolFinfos,
		sizeof( zombieFuncMolFinfos ) / sizeof( const Finfo* ),
		new Dinfo< ZombieFuncMol >()
	);

	return &zombieFuncMolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieFuncMolCinfo = ZombieFuncMol::initCinfo();

ZombieFuncMol::ZombieFuncMol()
{;}

void ZombieFuncMol::input( double v )
{;}

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
// This is more involved as it also has to zombify the Func.
void ZombieFuncMol::zombify( Element* solver, Element* orig )
{
	Element temp( orig->id(), zombieFuncMolCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieFuncMol* z = reinterpret_cast< ZombieFuncMol* >( zer.data() );
	FuncMol* m = reinterpret_cast< FuncMol* >( oer.data() );

	z->setN( zer, 0, m->getN() );
	z->setNinit( zer, 0, m->getNinit() );
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
	orig->zombieSwap( zombieFuncMolCinfo, dh );

	// Later change name just to 'func'
	Id funcId = Neutral::child( oer, "sumFunc" );
	if ( funcId != Id() ) {
		if ( funcId()->cinfo()->isA( "SumFunc" ) )
			ZombieSumFunc::zombify( solver, funcId(), orig->id() );
			// The additional Id argument helps the system to look up
			// what molecule is involved. Could of course get it as target.
		/*
		else if ( funcId()->cinfo().isA( "MathFunc" ) )
			ZombieMathFunc::zombify( solver, funcId() );
		else if ( funcId()->cinfo().isA( "Table" ) )
			ZombieTable::zombify( solver, funcId() );
		*/
	}
}

// Static func
void ZombieFuncMol::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieFuncMol* z = reinterpret_cast< ZombieFuncMol* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( FuncMol::initCinfo(), dh );

	FuncMol* m = reinterpret_cast< FuncMol* >( oer.data() );

	m->setN( z->getN( zer, 0 ) );
	m->setNinit( z->getNinit( zer, 0 ) );
}
