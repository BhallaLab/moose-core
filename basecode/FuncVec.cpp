/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <algorithm>
#include <fstream>
#include "header.h"
#include "ProcInfo.h"
#include "SetConn.h"
#include "Ftype.h"
#include "Ftype0.h"
/*
#include <string>
#include <vector>
#include <cassert>
#include "RecvFunc.h"
#include "Ftype.h"
#include "SetConn.h"
#include "FuncVec.h"
*/

using namespace std;

static const unsigned int EMPTY_ID = 0;

/**
 * This function manages a static vector of FuncVecs.
 */
static vector< FuncVec* >& funcVecLookup()
{
	static vector< FuncVec* > fv;

	return fv;
}

FuncVec::FuncVec( const string& className, const string& finfoName )
	: id_( 0 ), isDest_( 0 ), trigFuncVec_( 0 ), lookupFuncVec_( 0 )
{
	name_ = className + "." + finfoName;
	funcVecLookup().push_back( this );
}


/// addFunc pushes a new function onto the FuncVec.
void FuncVec::addFunc( RecvFunc func, const Ftype* ftype )
{
	func_.push_back( func );
	funcType_.push_back( ftype );
	id_ = 1; // Temporary hack to set up non-zero id when func is live.
	// parFuncSync_.push_back( ftype->parFuncSync() );
	// parFuncAsync_.push_back( ftype->parFuncAsync() );
}

/**
 * Makes a trigger FuncVec from the current one. Used by ValueFinfos.
 */
void FuncVec::makeTrig( )
{
	// Checks that this is set up from a ValueFinfo.
	assert( func_.size() == 1 );
	assert( funcType_.size() == 1 );
	assert( funcType_[0]->nValues() == 1 );
	trigFuncVec_ = new FuncVec( name_, "trig" );
	trigFuncVec_->addFunc( funcType_[0]->trigFunc(), Ftype0::global() );
	trigFuncVec_->isDest_ = 1;

	// Just to be sure that we've set it up.
	isDest_ = 1;
}

/**
 * Makes a LookupFinfo FuncVec from the current one. Used by LookupFinfos.
 */
void FuncVec::makeLookup()
{
	// Checks that this is set up from a LookupFinfo.
	assert( func_.size() == 1 );
	assert( funcType_.size() == 1 );
	assert( funcType_[0]->nValues() == 1 );
	lookupFuncVec_ = new FuncVec( name_, "lookup" );
	lookupFuncVec_->addFunc( funcType_[0]->recvFunc(), funcType_[0] );
	lookupFuncVec_->isDest_ = 1;

	// Just to be sure that we've set it up.
	isDest_ = 1;
}

/// func returns the indexed function.
RecvFunc FuncVec::func( unsigned int funcNum ) const
{
	assert( funcNum < func_.size() );
	return func_[ funcNum ];
}

/**
* parFuncSync returns a destination function that will handle
* the identical arguments and package them for sending to a
* remote node on a parallel message call.
* The Sync means that this func is for synchronous data.
*/
RecvFunc FuncVec::parFuncSync( unsigned int funcNum ) const
{
	assert( funcNum < func_.size() );
	return parFuncSync_[ funcNum ];
}

/**
* parFuncAsync returns a destination function that will handle
* the identical arguments and package them for sending to a
* remote node on a parallel message call.
* The Async means that this func is for synchronous data.
*/
RecvFunc FuncVec::parFuncAsync( unsigned int funcNum ) const
{
	assert( funcNum < func_.size() );
	return parFuncAsync_[ funcNum ];
}

// trigId returns the identifier of the trigFuncVec if it exists.
unsigned int FuncVec::trigId() const 
{
	if ( trigFuncVec_ )
		return trigFuncVec_->id();

	return 0;
}

// trigId returns the identifier of the trigFuncVec if it exists.
unsigned int FuncVec::lookupId() const 
{
	if ( lookupFuncVec_ )
		return lookupFuncVec_->id();

	return 0;
}

/**
* fType returns a vector of Ftypes for the functions
*/
const vector< const Ftype* >& FuncVec::fType() const
{
	return funcType_;
}

/**
 * This function returns the FuncVec belonging to the specified id
 */
const FuncVec* FuncVec::getFuncVec( unsigned int id )
{
	assert( funcVecLookup().size() > id );

	return funcVecLookup()[id];
}

const string& FuncVec::name() const
{
	return name_;
}

static bool fvcmp( const FuncVec* a, const FuncVec* b )
{
	return ( a->name() < b->name() );
}
/**
 * sortFuncVec puts them in order and assigns ids.
 * Must be called before any messaging is begun, because we'll need
 * the FuncVecs for that.
 * It is OK (if wasteful) to call it again later, provided no internode
 * messages have been sent and no-one is using the funcIds elsewhere,
 * as they will change.
 */
void FuncVec::sortFuncVec( )
{
	vector< FuncVec* >& fv = funcVecLookup();
	// Check if it has already been done. 
	if ( fv.size() > 2 && 
		fv[0]->name() == "empty.empty" && fv[1]->name() == "dummy.dummy" ) {
		sort( fv.begin() + 2, fv.end(), fvcmp );
		// cout << fv.size() << " FuncVecs rebuilt \n";
	} else {
		sort( fv.begin(), fv.end(), fvcmp );

		// Put it in at the beginning
		FuncVec* empty = new FuncVec( "empty", "empty" );
		fv[ fv.size() - 1 ] = fv[0];
		fv[0] = empty;

		// Put it in at number 1.
		FuncVec* dummy = new FuncVec( "dummy", "dummy" );
		fv[ fv.size() - 1 ] = fv[1];
		fv[1] = dummy;

		dummy->addFunc( &dummyFunc, Ftype0::global() );
		dummy->setDest();
		cout << fv.size() << " FuncVecs built for the first time\n";
	}
	// Note that 'empty' is at zero and 'dummy' at one.
	for ( unsigned int i = 0; i < fv.size(); i++ ) {
		if ( fv[i]->size() == 0 ) {
			fv[i]->id_ = EMPTY_ID;
		} else {
			fv[i]->id_ = i;
			// cout << "FuncVec # " << i << " = " << fv[i]->name() << endl;
		}
	}
	
	/*
	 * Uncomment if you want a list of FuncVecs stored in a text file. Commented
	 * out because it keeps creating the file everywhere you run moose.
	 */
//~ #ifndef NDEBUG /* If compiling in DEBUG mode. */
	//~ // Printing list of FuncVecs
	//~ string filename = "funcvec.txt";
	//~ ofstream fout( filename.c_str() );
	//~ for ( unsigned int i = 0; i < fv.size(); i++ )
		//~ fout << fv[ i ]->name() << "\n";
	//~ fout << flush;
	//~ cout << "Wrote list of sorted FuncVecs to " << filename << ".\n";
//~ #endif // NDEBUG
}
