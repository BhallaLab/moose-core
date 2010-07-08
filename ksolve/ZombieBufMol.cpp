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
#include "BufMol.h"
#include "ZombieMol.h"
#include "ZombieBufMol.h"

// Entirely derived from ZombieMol. Only the zombification routines differ.
const Cinfo* ZombieBufMol::initCinfo()
{
	static Cinfo zombieBufMolCinfo (
		"ZombieBufMol",
		ZombieMol::initCinfo(),
		0,
		0,
		new Dinfo< ZombieBufMol >()
	);

	return &zombieBufMolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieBufMolCinfo = ZombieBufMol::initCinfo();

ZombieBufMol::ZombieBufMol()
{;}


//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
void ZombieBufMol::zombify( Element* solver, Element* orig )
{
	Element temp( orig->id(), zombieBufMolCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieBufMol* z = reinterpret_cast< ZombieBufMol* >( zer.data() );
	BufMol* m = reinterpret_cast< BufMol* >( oer.data() );

	z->setN( zer, 0, m->getN() );
	z->setNinit( zer, 0, m->getNinit() );
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
	orig->zombieSwap( zombieBufMolCinfo, dh );
}

// Static func
void ZombieBufMol::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieBufMol* z = reinterpret_cast< ZombieBufMol* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( BufMol::initCinfo(), dh );

	BufMol* m = reinterpret_cast< BufMol* >( oer.data() );

	m->setN( z->getN( zer, 0 ) );
	m->setNinit( z->getNinit( zer, 0 ) );
}
