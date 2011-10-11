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

ZeroDimHandler::ZeroDimHandler( const DinfoBase* dinfo, bool isGlobal )
	: DataHandler( dinfo, isGlobal )
{
	myThread_ = 0;
	if ( isGlobal || Shell::myNode() == 0 ) {
		data_ = dinfo->allocData( 1 );
	} else {
		data_ = 0;
	}

	if ( data_ && Shell::numProcessThreads() > 1 ) {
		myThread_ = 1 + Id::numIds() % Shell::numProcessThreads();
	}
}

ZeroDimHandler::ZeroDimHandler( const DinfoBase* dinfo, char* data )
	: DataHandler( dinfo, 1 ), data_( data )
{
	myThread_ = 0;
	if ( data_ && Shell::numProcessThreads() >= 2 )
		myThread_ = 1 + Id::numIds() % Shell::numProcessThreads();
}

ZeroDimHandler::ZeroDimHandler( const ZeroDimHandler* other )
	: DataHandler( other->dinfo(), other->isGlobal() )
{
	myThread_ = 0;
	if ( other->isGlobal() || Shell::myNode() == 0 ) {
		data_ = dinfo()->allocData( 1 ); 
		if ( other->data_ )
			dinfo()->assignData( data_, 1, other->data_, 1 );
	} else {
		data_ = 0;
	}

	if ( data_ && Shell::numProcessThreads() > 1 ) {
		myThread_ = 1 + Id::numIds() % Shell::numProcessThreads();
	}
}

ZeroDimHandler::~ZeroDimHandler()
{
	// The base class ZeroDimGlobalHandler does the destruction.
}

///////////////////////////////////////////////////////////////////////
// Information functions
///////////////////////////////////////////////////////////////////////

char* ZeroDimHandler::data( DataId index ) const {
	return data_;
}

unsigned int ZeroDimHandler::localEntries() const {
	return ( data_ != 0 );
}

vector< unsigned int > ZeroDimHandler::dims() const {
	static vector< unsigned int > ret( 0 );
	return ret;
}

bool ZeroDimHandler::isDataHere( DataId index ) const
{
	return ( data_ != 0 );
}

///////////////////////////////////////////////////////////////////////
// Load balancing
///////////////////////////////////////////////////////////////////////

bool ZeroDimHandler::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

///////////////////////////////////////////////////////////////////////
// Process
///////////////////////////////////////////////////////////////////////

void ZeroDimHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{
	if ( data_ && ( p->threadIndexInGroup == myThread_ || 
		p->threadIndexInGroup == 0 ) ) {
		const OpFunc* f = e->cinfo()->getOpFunc( fid );
		const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
		assert( pf );
		pf->proc( data_, Eref( e, 0 ), p );
	}
}

void ZeroDimHandler::foreach( const OpFunc* f, Element* e, const Qinfo* q,
	const double* arg, unsigned int argSize, unsigned int numArgs ) const
{
	if ( data_ && ( q->threadNum() == myThread_ || q->threadNum() == 0 ) )
		f->op( Eref( e, 0 ), q, arg );
}

unsigned int ZeroDimHandler::getAllData( vector< char* >& dataVec ) const
{
	dataVec.resize( 0 );
	if ( data_ )
		dataVec.push_back( data_ );
	return dataVec.size();
}

///////////////////////////////////////////////////////////////////////
// Data reallocation and copy
///////////////////////////////////////////////////////////////////////

void ZeroDimHandler::globalize( const char* data, unsigned int numEntries )
{
	if ( !data_ )
		data_ = dinfo()->copyData( data, numEntries, 1 );
	isGlobal_ = 1;
}

void ZeroDimHandler::unGlobalize()
{
	if ( !isGlobal_ ) return;

	isGlobal_ = 0;
	if ( Shell::myNode() != 0 ) { // Clear it out
		dinfo()->destroyData( data_ );
		data_ = 0;
	}
}

DataHandler* ZeroDimHandler::copy( bool toGlobal, unsigned int n ) const
{
	if ( toGlobal ) {
		if ( !isGlobal() ) {
			cout << "Warning: ZeroDimHandler::copy: Cannot copy from nonGlobal to global\n";
			return 0;
		}
	}
	if ( n > 1 ) {
		OneDimHandler* ret = new OneDimHandler( dinfo(), toGlobal, n );
		if ( data_ )
			ret->assign( data_, 1 );
		return ret;
	} else {
		return ( new ZeroDimHandler( this ) );
	}
	return 0;
}

DataHandler* ZeroDimHandler::copyUsingNewDinfo( 
	const DinfoBase* dinfo ) const
{
	ZeroDimHandler* ret = new ZeroDimHandler( dinfo, isGlobal_ );
	return ret;
}

bool ZeroDimHandler::resize( unsigned int dimension, unsigned int size )
{
	return 0; // Cannot resize a zero dim array!
}

void ZeroDimHandler::assign( const char* orig, unsigned int numOrig )
{
	if ( data_ && numOrig > 0 ) {
		dinfo()->assignData( data_, 1, orig, 1 );
	}
}

/*
//////////////////////////////////////////////////////////////////////
// Iterator functions.
//////////////////////////////////////////////////////////////////////

DataHandler::iterator ZeroDimHandler::begin( ThreadId threadNum ) const
{
	if ( data_ && ( threadNum + 1U == Shell::numProcessThreads() ) )
		return iterator( this, 0 );
	else
		return iterator();
}

DataHandler::iterator ZeroDimHandler::end( ThreadId threadNum ) const
{
	return iterator();
}

void ZeroDimHandler::rolloverIncrement( DataHandler::iterator* i ) const
{
	i->setData( 0 );
}
*/
