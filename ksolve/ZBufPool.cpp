/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "StoichHeaders.h"
#include "StoichPools.h"
#include "ZPool.h"
#include "ZBufPool.h"

// Entirely derived from ZombiePool. Only the zombification routines differ.
const Cinfo* ZBufPool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions: use virtual functions to deal with, the
		// moose definitions are inherited.
		//////////////////////////////////////////////////////////////
	static Cinfo zombieBufPoolCinfo (
		"ZBufPool",
		ZPool::initCinfo(),
		0,
		0,
		new Dinfo< ZBufPool >()
	);

	return &zombieBufPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieBufPoolCinfo = ZBufPool::initCinfo();

static const SrcFinfo1< double >* requestSize =
	dynamic_cast< const SrcFinfo1< double >* >(
	zombieBufPoolCinfo->findFinfo( "requestSize" ) );


ZBufPool::ZBufPool()
{;}

ZBufPool::~ZBufPool()
{;}

//////////////////////////////////////////////////////////////
// Field functions
//////////////////////////////////////////////////////////////

void ZBufPool::vSetN( const Eref& e, const Qinfo* q, double v )
{
	stoich_->setN( e, v );
	stoich_->setNinit( e, v );
}

void ZBufPool::vSetNinit( const Eref& e, const Qinfo* q, double v )
{
	vSetN( e, q, v );
}

void ZBufPool::vSetConc( const Eref& e, const Qinfo* q, double conc )
{
	double n = NA * conc * lookupSizeFromMesh( e, requestSize );
	vSetN( e, q, n );
}

void ZBufPool::vSetConcInit( const Eref& e, const Qinfo* q, double conc )
{
	vSetConc( e, q, conc );
}

