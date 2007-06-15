/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"

//////////////////////////////////////////////////////////////////
// Global  functions
//////////////////////////////////////////////////////////////////

FunctionDataManager* getFunctionDataManager()
{
	static FunctionDataManager fdm;

	return &fdm;
}

const FunctionData* lookupFunctionData( RecvFunc rf )
{
	return getFunctionDataManager()->find( rf );
}

const FunctionData* lookupFunctionData( unsigned int index )
{
	return getFunctionDataManager()->find( index );
}


//////////////////////////////////////////////////////////////////
// FunctionData functions
//////////////////////////////////////////////////////////////////

const Ftype* FunctionData::funcType() const
{
	return info_->ftype();
}

FunctionData::FunctionData( 
	RecvFunc func, const Finfo* info, unsigned int index )
	: func_( func ), info_( info ), index_( index )
{;}

//////////////////////////////////////////////////////////////////
// FunctionDataManager functions
//////////////////////////////////////////////////////////////////

/**
 * creates a new FunctionData and inserts into the map and vector.
 * Returns the new FunctionData.
 */
const FunctionData* FunctionDataManager::add( RecvFunc func, const Finfo* info )
{
	vector< const FunctionData* >::const_iterator i;
	for ( i = funcVec_.begin(); i != funcVec_.end(); i++ )
		if ( (*i)->func() == func )
			break; 
	if ( i == funcVec_.end() ) { // make a new one.
		FunctionData* fd = new FunctionData( func, info, funcVec_.size() );
		funcVec_.push_back( fd );
		funcMap_[func] = fd;
		// cout << "# FunctionData = " << funcVec_.size() << endl;
		return fd;
	} else {
		return *i; // already exists.
	}
}

const FunctionData* FunctionDataManager::find( RecvFunc rf )
{
	map< RecvFunc, const FunctionData* >::iterator i = 
		funcMap_.find( rf );
	if ( i != funcMap_.end() )
		return i->second; 
	return 0;
}

const FunctionData* FunctionDataManager::find( unsigned int index )
{
	if ( index < funcVec_.size() )
		return funcVec_[ index ];
	return 0;
}
