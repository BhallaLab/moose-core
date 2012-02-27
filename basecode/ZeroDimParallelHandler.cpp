/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../shell/Shell.h"

/// Generic constructor
ZeroDimParallelHandler::ZeroDimParallelHandler( const DinfoBase* dinfo, 
	const vector< DimInfo >&dims, unsigned short pathDepth, bool isGlobal )
	: ZeroDimHandler( dinfo, dims, pathDepth, isGlobal )
{;}

/// Special constructor using in Cinfo::makeCinfoElements
ZeroDimParallelHandler::ZeroDimParallelHandler( const DinfoBase* dinfo, char* data )
	: ZeroDimHandler( dinfo, data )
{;}

/// Copy constructor
ZeroDimParallelHandler::ZeroDimParallelHandler( const ZeroDimParallelHandler* other )
	: ZeroDimHandler( other )
{;}

ZeroDimParallelHandler::~ZeroDimParallelHandler()
{;}

///////////////////////////////////////////////////////////////////////
// Information functions: all inherited
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// Load balancing
///////////////////////////////////////////////////////////////////////

bool ZeroDimParallelHandler::execThread( ThreadId thread, DataId di ) const
{
	return ( di == DataId::globalField || data( di ) != 0 );
}
///////////////////////////////////////////////////////////////////////
// Process
///////////////////////////////////////////////////////////////////////

void ZeroDimParallelHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{
	if ( data( 0 ) ) {
		const OpFunc* f = e->cinfo()->getOpFunc( fid );
		const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
		assert( pf );
		pf->proc( data( 0 ), Eref( e, 0 ), p );
	}
}

void ZeroDimParallelHandler::forall( const OpFunc* f, Element* e, const Qinfo* q,
	const double* arg, unsigned int argSize, unsigned int numArgs ) const
{
	if ( data( 0 ) ) // Call it on all threads. The object has to figure out
	// what to do on each thread.
		f->op( Eref( e, 0 ), q, arg );
}

///////////////////////////////////////////////////////////////////////
// Data reallocation and copy
///////////////////////////////////////////////////////////////////////

DataHandler* ZeroDimParallelHandler::copy( 
	unsigned short newParentDepth, 
	unsigned short copyRootDepth, 
	bool toGlobal, unsigned int n ) const
{
	if ( toGlobal ) {
		if ( !isGlobal() ) {
			cout << "Warning: ZeroDimParallelHandler::copy: Cannot copy from nonGlobal to global\n";
			return 0;
		}
	}
	if ( n > 1 ) {
		cout << "Error: ZeroDimParallelHandler::copy: cannot scale to arrays";

		return 0;
	} else {
		ZeroDimParallelHandler* ret = new ZeroDimParallelHandler( this );
		if ( !ret->changeDepth( pathDepth() + 1 + newParentDepth - copyRootDepth ) ) {
			delete ret;
			return 0;
		}
		return ret;
	}
	return 0;
}

DataHandler* ZeroDimParallelHandler::copyUsingNewDinfo( 
	const DinfoBase* dinfo ) const
{
	ZeroDimParallelHandler* ret = new ZeroDimParallelHandler( dinfo, 
		dims_, pathDepth_, isGlobal_ );
	return ret;
}
