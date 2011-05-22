/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SetGet.h"
#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"

/*
Eref SetGet::shelle_( 0, 0 );
Element* SetGet::shell_;

void SetGet::setShell()
{
	Id shellid;
	shelle_ = shellid.eref();
	shell_ = shelle_.element();
}
*/

/**
 * completeSet: Confirms that the target function has been executed.
 * Later this has to be much more complete: perhaps await return of a baton
 * to ensure that the operation is complete through varous threading,
 * nesting, and multinode operations.
 * Even single thread, single node this current version is dubious: 
 * suppose the function called by 'set' issues a set command of its own?
 */
void SetGet::completeSet() const
{
	// e_.element()->clearQ();
	// Qinfo::clearQ( Shell::procInfo() );
}

//////////////////////////////////////////////////////////////////////
// A group of functions to forward dispatch commands to the Shell.
//////////////////////////////////////////////////////////////////////

void SetGet::dispatchSet( const ObjId& oid, FuncId fid, 
	const char* args, unsigned int size )
{
	Shell::dispatchSet( oid, fid, args, size );
}

void SetGet::dispatchSetVec( const ObjId& oid, FuncId fid, 
	const PrepackedBuffer& arg )
{
	Shell::dispatchSetVec( oid, fid, arg );
}

/*
const vector< char* >& SetGet::dispatchGet( 
	const ObjId& oid, const string& field,
	const SetGet* sg, unsigned int& numGetEntries )
{
	return Shell::dispatchGet( oid, field, sg, numGetEntries );
}
*/


const vector< char* >& SetGet::dispatchGet( 
	const ObjId& dest, FuncId fid, const PrepackedBuffer& buf )
{
	Eref sheller = Id().eref();
	Shell* s = reinterpret_cast< Shell* >( sheller.data() );
	return s->dispatchGet( sheller, dest, fid, buf );
}

/*
const vector< char* >& SetGet::dispatchLookupGet( 
	const ObjId& oid, const string& field, char* indexBuf,
	const SetGet* sg, unsigned int& numGetEntries )
{
	// return Shell::dispatchGet( oid, field, sg, numGetEntries );
}
*/

/*
const vector< char* >& SetGet::dispatchGetVec( 
	const SetGet* sg, const ObjId& dest, const string& field,
	const char* args, unsigned int size)
{
	static vector< char* > badRet( 0 );
	ObjId tgt( dest );
	FuncId fid;

	unsigned int numRetEntries = sg->checkSet( field, tgt, fid );
	if ( numRetEntries > 0 ) {
		FuncId retFuncId = receiveGet()->getFid();
		Conv< FuncId > conv( retFuncId );
		char *temp = new char[ conv.size() ];
		conv.val2buf( temp );
		const vector< char* >& ret = 
			Shell::dispatchGet( tgt, fid, temp, conv.size() );
		delete[] temp;
		return ret;
	}
	return badRet;
}
*/

//////////////////////////////////////////////////////////////////////

unsigned int SetGet::checkSet( 
	const string& field, ObjId& tgt, FuncId& fid ) const
{
	// string field = "set_" + destField;
	const Finfo* f = oid_.element()->cinfo()->findFinfo( field );
	if ( !f ) { // Could be a child element? Note that field name will 
		// change from set_<name> to just <name>
		string f2 = field.substr( 4 );
		Id child = Neutral::child( oid_.eref(), f2 );
		if ( child == Id() ) {
			cout << "Error: SetGet:checkSet:: No field or child named '" <<
				field << "' was found on\n" << tgt.id.path() << endl;
		} else {
			if ( field.substr( 0, 4 ) == "set_" )
				f = child()->cinfo()->findFinfo( "set_this" );
			else if ( field.substr( 0, 4 ) == "get_" )
				f = child()->cinfo()->findFinfo( "get_this" );
			assert( f ); // should always work as Neutral has the field.
			if ( child()->dataHandler()->totalEntries() == 
				oid_.element()->dataHandler()->totalEntries() ) {
				tgt = ObjId( child, oid_.dataId );
				if ( !tgt.isDataHere() )
					return 0;
			} else if ( child()->dataHandler()->totalEntries() <= 1 ) {
				tgt = ObjId( child, 0 );
				if ( !tgt.isDataHere() )
					return 0;
			} else {
				cout << "SetGet::checkSet: child index mismatch\n";
				return 0;
			}
		}
	} else {
		tgt = oid_;
	}
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( f );
	if ( !df )
		return 0;
	
	fid = df->getFid();
	const OpFunc* func = df->getOpFunc();

// 	fid = oid_.element()->cinfo()->getOpFuncId( field );
//	const OpFunc* func = oid_.element()->cinfo()->getOpFunc( fid );
	if ( !func ) {
		cout << "set::Failed to find " << oid_ << "." << field << endl;
		return 0;
	}

	// This is the crux of the function: typecheck for the field.
	if ( func->checkSet( this ) ) {
		return tgt.element()->dataHandler()->totalEntries();
	} else {
		cout << "set::Type mismatch" << oid_ << "." << field << endl;
		return 0;
	}
}

/////////////////////////////////////////////////////////////////////////

// Static function
bool SetGet::strGet( const ObjId& tgt, const string& field, string& ret )
{
	const Finfo* f = tgt.element()->cinfo()->findFinfo( field );
	if ( !f ) {
		cout << Shell::myNode() << ": Error: SetGet::strGet: Field " <<
			field << " not found on Element " << tgt.element()->getName() <<
			endl;
		return 0;
	}
	return f->strGet( tgt.eref(), field, ret );
}

bool SetGet::strSet( const ObjId& tgt, const string& field, const string& v)
{
	const Finfo* f = tgt.element()->cinfo()->findFinfo( field );
	if ( !f ) {
		cout << Shell::myNode() << ": Error: SetGet::strSet: Field " <<
			field << " not found on Element " << tgt.element()->getName() <<
			endl;
		return 0;
	}
	return f->strSet( tgt.eref(), field, v );
}
