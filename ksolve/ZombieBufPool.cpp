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

#include "Pool.h"
#include "BufPool.h"
#include "ZombiePool.h"
#include "ZombieBufPool.h"

// Entirely derived from ZombiePool. Only the zombification routines differ.
const Cinfo* ZombieBufPool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieBufPool, double > n(
			"n",
			"Number of molecules in pool",
			&ZombieBufPool::setN,
			&ZombieBufPool::getN
		);

		static ElementValueFinfo< ZombieBufPool, double > nInit(
			"nInit",
			"Initial value of number of molecules in pool",
			&ZombieBufPool::setNinit,
			&ZombieBufPool::getNinit
		);

		static ElementValueFinfo< ZombieBufPool, double > conc(
			"conc",
			"Concentration of molecules in pool",
			&ZombieBufPool::setConc,
			&ZombieBufPool::getConc
		);

		static ElementValueFinfo< ZombieBufPool, double > concInit(
			"concInit",
			"Initial value of molecular concentration in pool",
			&ZombieBufPool::setConcInit,
			&ZombieBufPool::getConcInit
		);

	static Finfo* zombieBufPoolFinfos[] = {
		&n,				// Value
		&nInit,			// Value
		&conc,			// Value
		&concInit,		// Value
	};

	static Cinfo zombieBufPoolCinfo (
		"ZombieBufPool",
		ZombiePool::initCinfo(),
		zombieBufPoolFinfos,
		sizeof( zombieBufPoolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< ZombieBufPool >()
	);

	return &zombieBufPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieBufPoolCinfo = ZombieBufPool::initCinfo();

ZombieBufPool::ZombieBufPool()
{;}


//////////////////////////////////////////////////////////////
// Field functions
//////////////////////////////////////////////////////////////

void ZombieBufPool::setN( const Eref& e, const Qinfo* q, double v )
{
	this->innerSetN( e.index().value(), e.id(), v );
	this->innerSetNinit( e.index().value(), e.id(), v );

	/*
	unsigned int i = convertIdToPoolIndex( e.id() );
	S_[ e.index().value() ][ i ] = v;
	Sinit_[ e.index().value() ][ i ] = v;
	*/
}

double ZombieBufPool::getN( const Eref& e, const Qinfo* q ) const
{
	return S_[ e.index().value() ][ convertIdToPoolIndex( e.id() ) ];
}

void ZombieBufPool::setNinit( const Eref& e, const Qinfo* q, double v )
{
	setN( e, q, v );
}

double ZombieBufPool::getNinit( const Eref& e, const Qinfo* q ) const
{
	return Sinit_[ e.index().value() ][ convertIdToPoolIndex( e.id() ) ];
}

void ZombieBufPool::setConc( const Eref& e, const Qinfo* q, double conc )
{
	static const Finfo* req = 
		ZombiePool::initCinfo()->findFinfo( "requestSize" );
	static const SrcFinfo1< double >* requestSize = dynamic_cast< 
		const SrcFinfo1< double >* >( req );
	assert( requestSize );

	double n = NA * conc * lookupSizeFromMesh( e, requestSize );
	setN( e, q, n );
	/*
	unsigned int pool = convertIdToPoolIndex( e.id() );
	S_[ e.index().value() ][ pool ] = n;
	Sinit_[ e.index().value() ][ pool ] = n;
	*/
}

double ZombieBufPool::getConc( const Eref& e, const Qinfo* q ) const
{
	static const Finfo* req = 
		ZombiePool::initCinfo()->findFinfo( "requestSize" );
	static const SrcFinfo1< double >* requestSize = dynamic_cast< 
		const SrcFinfo1< double >* >( req );
	assert( requestSize );

	unsigned int pool = convertIdToPoolIndex( e.id() );
	return S_[ e.index().value() ][ pool ] / ( NA * lookupSizeFromMesh( e, requestSize ) );
}

void ZombieBufPool::setConcInit( const Eref& e, const Qinfo* q, double conc )
{
	setConc( e, q, conc );
}

double ZombieBufPool::getConcInit( const Eref& e, const Qinfo* q ) const
{
	static const Finfo* req = 
		ZombiePool::initCinfo()->findFinfo( "requestSize" );
	static const SrcFinfo1< double >* requestSize = dynamic_cast< 
		const SrcFinfo1< double >* >( req );
	assert( requestSize );

	unsigned int pool = convertIdToPoolIndex( e.id() );
	return Sinit_[ e.index().value() ][ pool ] / ( NA * lookupSizeFromMesh( e, requestSize ) );
}


//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
void ZombieBufPool::zombify( Element* solver, Element* orig )
{
	Element temp( orig->id(), zombieBufPoolCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );
	unsigned int numEntries = orig->dataHandler()->localEntries();

	ZombieBufPool* z = reinterpret_cast< ZombieBufPool* >( zer.data() );
	BufPool* m = reinterpret_cast< BufPool* >( oer.data() );

	unsigned int poolIndex = z->convertIdToPoolIndex( orig->id() );
	z->concInit_[ poolIndex ] = m->getConcInit();
	z->setN( zer, 0, m->getN() );
	z->setNinit( zer, 0, m->getNinit( oer, 0 ) );
	z->setDiffConst( zer, 0, m->getDiffConst() );
	/*
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler(),
		orig->dataHandler() );
	*/
	DataHandler* dh = new ZombieHandler( solver->dataHandler(),
		orig->dataHandler(), 0, numEntries );
	orig->zombieSwap( zombieBufPoolCinfo, dh );
}

// Static func
void ZombieBufPool::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieBufPool* z = reinterpret_cast< ZombieBufPool* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( BufPool::initCinfo(), dh );

	BufPool* m = reinterpret_cast< BufPool* >( oer.data() );

	m->setN( z->getN( zer, 0 ) );
	m->setNinit( oer, 0, z->getNinit( zer, 0 ) );
}
