/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

ZeroDimGlobalHandler::ZeroDimGlobalHandler( const DinfoBase* dinfo )
	: DataHandler( dinfo ), data_( 0 )
{;}

// Special constructor for use by Cinfo Elements.
ZeroDimGlobalHandler::ZeroDimGlobalHandler( const DinfoBase* dinfo, 
	char* data )
	: DataHandler( dinfo ), data_( data )
{;}

ZeroDimGlobalHandler::ZeroDimGlobalHandler( const ZeroDimGlobalHandler* other )
	: DataHandler( other->dinfo() ),
		data_( other->dinfo()->copyData( other->data_, 1, 1 ) )
{;}

ZeroDimGlobalHandler::~ZeroDimGlobalHandler()
{
	// This is a hack to avoid deleting the data for the Cinfo 
	// elements, which are statically allocated.
	static Dinfo< Cinfo > ref;
	// assert( data_ != 0 );
	// Cannot use this assertion, because derived classes may or may
	// not have any allocated data depending on node decomposition.
	// Instead just check for availability of data ptr to delete.
	if ( data_ && !dinfo()->isA( &ref ) ) {
		dinfo()->destroyData( data_ );
	}
	data_ = 0;
}

DataHandler* ZeroDimGlobalHandler::globalize() const
{
	return new ZeroDimGlobalHandler( this );
}

DataHandler* ZeroDimGlobalHandler::unGlobalize() const
{
	return 0;
}

bool ZeroDimGlobalHandler::innerNodeBalance( unsigned int size,
	unsigned int myNode, unsigned int numNodes )
{
	return 0;
}

DataHandler* ZeroDimGlobalHandler::copy() const
{
	return ( new ZeroDimGlobalHandler( this ) );
}

DataHandler* ZeroDimGlobalHandler::copyUsingNewDinfo( 
	const DinfoBase* dinfo) const
{
	ZeroDimGlobalHandler* ret = new ZeroDimGlobalHandler( dinfo );
	ret->data_ = dinfo->allocData( 1 );
	return ret;
}

DataHandler* ZeroDimGlobalHandler::copyExpand( unsigned int copySize ) const
{
	OneDimGlobalHandler* ret = new OneDimGlobalHandler( dinfo() );
	vector< unsigned int > dims( 1, copySize );
	ret->resize( dims );
	for ( iterator i = ret->begin(); i != ret->end(); ++i ) {
		char* temp = *i;
		memcpy( temp, data_, dinfo()->size() );
	}
	return ret;
}

DataHandler* ZeroDimGlobalHandler::copyToNewDim( unsigned int newDimSize ) const
{
	return copyExpand( newDimSize );
}


void ZeroDimGlobalHandler::process( const ProcInfo* p, Element* e, FuncId fid ) const
{
	// We only want one thread to deal with this.
	// In principle we could subdivide the zeroDim cases using
	// the Element Id:
	// if ( p->threadIndexInGroup == e->id()->value() % p->numThreadsinGroup)
	if ( p->threadIndexInGroup == p->numThreadsInGroup - 1 ) {
		const OpFunc* f = e->cinfo()->getOpFunc( fid );
		const ProcOpFuncBase* pf = dynamic_cast< const ProcOpFuncBase* >( f );
		assert( pf );
		pf->proc( data_, Eref( e, 0 ), p );
	//	reinterpret_cast< Data* >( data_ )->process( p, Eref( e, 0 ) );
	}
}

bool ZeroDimGlobalHandler::resize( vector< unsigned int > dims )
{
	if ( !data_ )
		data_ = dinfo()->allocData( 1 );
	return 1;
}

vector< unsigned int > ZeroDimGlobalHandler::dims() const
{
	static vector< unsigned int > ret;
	return ret;
}

bool ZeroDimGlobalHandler::isAllocated() const {
	return data_ != 0;
}

DataHandler::iterator ZeroDimGlobalHandler::begin() const
{
	//cout << Shell::myNode() << ": ZeroDimGlobalHandler::begin\n";
	return iterator( this, 0, 0 );
}

DataHandler::iterator ZeroDimGlobalHandler::end() const
{
	//cout << Shell::myNode() << ": ZeroDimGlobalHandler::end\n";
	return iterator( this, 1, 1 );
}

bool ZeroDimGlobalHandler::setDataBlock(
	const char* data, unsigned int numData,
	const vector< unsigned int >& startIndex ) const
{
	if ( !isAllocated() ) return 0;
	if ( numData != 1 ) return 0;
	if ( startIndex.size() != 0 ) return 0;

	memcpy( data_, data, dinfo()->size() );
	return 1;
}


bool ZeroDimGlobalHandler::setDataBlock(
	const char* data, unsigned int numData,
	DataId startIndex ) const
{
	if ( !isAllocated() ) return 0;
	if ( numData != 1 ) return 0;
	if ( startIndex.data() != 0 ) return 0;
	memcpy( data_, data, dinfo()->size() );
	return 1;
}

void ZeroDimGlobalHandler::nextIndex( DataId& index, 
	unsigned int& linearIndex ) const
{
	index.incrementDataIndex();
	++linearIndex;
}
