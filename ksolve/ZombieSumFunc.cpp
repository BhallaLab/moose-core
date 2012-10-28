/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include "FuncBase.h"
#include "../kinetics/SumTotalTerm.h"
#include "SumFunc.h"
#include "ZombieSumFunc.h"
#include "ZombieSumFunc.h"
#include "ElementValueFinfo.h"
#include "DataHandlerWrapper.h"

const Cinfo* ZombieSumFunc::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ReadOnlyElementValueFinfo< ZombieSumFunc, double > result(
			"result",
			"outcome of summation",
			&ZombieSumFunc::getResult
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< ZombieSumFunc >( &ZombieSumFunc::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< ZombieSumFunc >( &ZombieSumFunc::reinit ) );
		static DestFinfo input( "input",
			"Handles input values",
			new OpFunc1< ZombieSumFunc, double >( &ZombieSumFunc::input ) );

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////

		static SrcFinfo1< double > output(
				"output", 
				"Sends out sum on each timestep"
		);

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

	static Finfo* zombieSumFuncFinfos[] = {
		&result,	// Value
		&input,				// DestFinfo
		&output,			// SrcFinfo
		&proc,				// SharedFinfo
	};

	static Cinfo zombieSumFuncCinfo (
		"ZombieSumFunc",
		Neutral::initCinfo(),
		zombieSumFuncFinfos,
		sizeof( zombieSumFuncFinfos ) / sizeof ( Finfo* ),
		new Dinfo< ZombieSumFunc >()
	);

	return &zombieSumFuncCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieSumFuncCinfo = ZombieSumFunc::initCinfo();

ZombieSumFunc::ZombieSumFunc()
	: result_( 0.0 )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ZombieSumFunc::process( const Eref& e, ProcPtr p )
{;}

void ZombieSumFunc::reinit( const Eref& e, ProcPtr p )
{;}

void ZombieSumFunc::input( double v )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

double ZombieSumFunc::getResult( const Eref& e, const Qinfo* q ) const
{
	return S_[ e.index().value() ][ convertIdToFuncIndex( e.id() ) + numVarPools_ + numBufPools_ ];
}

//////////////////////////////////////////////////////////////
// Zombie conversion functions
//////////////////////////////////////////////////////////////

// static func
void ZombieSumFunc::zombify( Element* solver, Element* orig, Id molId )
{
	static const DestFinfo* finfo = dynamic_cast< const DestFinfo* >(
		SumFunc::initCinfo()->findFinfo( "input" ) );
	assert( finfo );

	Element temp( orig->id(), zombieSumFuncCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieSumFunc* z = reinterpret_cast< ZombieSumFunc* >( zer.data() );

	vector< Id > srcPools;
	unsigned int funcIndex = z->convertIdToFuncIndex( orig->id() );
	unsigned int numSrc = orig->getNeighbours( srcPools, finfo );
	assert( numSrc > 0 );
	vector< unsigned int > poolIndex( numSrc );
	for ( unsigned int i = 0; i < numSrc; ++i )
		poolIndex[i] = z->convertIdToPoolIndex( srcPools[i] );
	SumTotalTerm * stt = new SumTotalTerm();
	stt->setReactants( poolIndex );
	z->funcs_[ funcIndex ] = stt;

	DataHandler* dh = new ZombieHandler( solver->dataHandler(),
		orig->dataHandler() );
	orig->zombieSwap( zombieSumFuncCinfo, dh );
}

// Static func
void ZombieSumFunc::unzombify( Element* zombie )
{
	DataHandler* oldHandler = zombie->dataHandler();
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieHandler* zh = dynamic_cast< ZombieHandler* >( oldHandler );
	assert( zh );

	DataHandler* sumFuncHandler = zh->origHandler();

	zombie->zombieSwap( SumFunc::initCinfo(), sumFuncHandler );

	// Delete the old handler, which is a ZombieHandler and thus
	// leaves its contents intact.
	delete zh;
}

