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
		// Field Definitions: use virtual functions to deal with, the
		// moose definitions are inherited.
		//////////////////////////////////////////////////////////////
	static Cinfo zombieBufPoolCinfo (
		"ZombieBufPool",
		ZombiePool::initCinfo(),
		0,
		0,
		new Dinfo< ZombieBufPool >()
	);

	return &zombieBufPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieBufPoolCinfo = ZombieBufPool::initCinfo();

static const SrcFinfo1< double >* requestSize =
	dynamic_cast< const SrcFinfo1< double >* >(
	zombieBufPoolCinfo->findFinfo( "requestSize" ) );


ZombieBufPool::ZombieBufPool()
{;}

ZombieBufPool::~ZombieBufPool()
{;}

//////////////////////////////////////////////////////////////
// Field functions
//////////////////////////////////////////////////////////////

void ZombieBufPool::vSetN( const Eref& e, const Qinfo* q, double v )
{
	stoich_->innerSetN( e.index().value(), e.id(), v );
	stoich_->innerSetNinit( e.index().value(), e.id(), v );
}

void ZombieBufPool::vSetNinit( const Eref& e, const Qinfo* q, double v )
{
	vSetN( e, q, v );
}

void ZombieBufPool::vSetConc( const Eref& e, const Qinfo* q, double conc )
{
	double n = NA * conc * lookupSizeFromMesh( e, requestSize );
	vSetN( e, q, n );
}

void ZombieBufPool::vSetConcInit( const Eref& e, const Qinfo* q, double conc )
{
	vSetConc( e, q, conc );
}

