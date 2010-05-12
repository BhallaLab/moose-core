/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Message.h"


/*
bool addMsgToFunc( Element* src, const Finfo* finfo, Element* dest, 
	FuncId fid )
{
	const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >( finfo );
	if ( !sf ) {
		cout << "Warning: addMsgToFunc:: Source of message '" << 
			finfo->name() << "' is not a SrcFinfo\n";
		return 0;
	}
	// Msg* m = new SingleMsg( Eref( src, 0 ), Eref( dest, 0 ) );
	Msg* m = new OneToOneMsg( src, dest );
	src->addMsgAndFunc( m->mid(), fid, sf->getBindIndex() );
	return 1;
}

bool addSharedMsg( Element* src, const Finfo* f1, Element* dest, 
	const Finfo* f2 )
{
	return 0;
}
*/

/**
 * Normal message adds should be from srcFinfo to named opFunc.
 * Shared message adds are from srcFinfo to destfinfo, where both are
 * 	shared Finfos.
bool add( Element* src, const string& srcField, 
	Element* dest, const string& destField )
{
	const Finfo* f1 = src->cinfo()->findFinfo( srcField );
	if ( !f1 ) {
		cout << "add: Error: Failed to find field " << srcField << 
			" on src:\n"; // Put name here.
		return 0;
	}
	const Finfo* f2 = dest->cinfo()->findFinfo( destField );
	if ( !f2 ) {
		cout << "add: Error: Failed to find field " << destField << 
			" on dest:\n"; // Put name here.
		return 0;
	}

	Msg* m = new OneToOneMsg( src, dest );
	if ( !f1->addMsg( f2, m->mid(), src )  ) {
		cout << "add: Error: Finfo type mismatch for " << 
			destField << " on dest:\n"; // Put name here.
		delete m;
		return 0;
	}
	return 1;

}
 */

/*
const SrcFinfo* validateMsg( Element* src, const string& srcField, 
	Element* dest, const string& destField, FuncId& fid )
{
	const Finfo* f1 = src->cinfo()->findFinfo( srcField );
	if ( !f1 ) {
		cout << "add: Error: Failed to find field '" << srcField << 
			"' on src: " << src->name() << "\n"; // Put name here.
		return 0;
	}

	const Finfo* f2 = dest->cinfo()->findFinfo( destField );
	if ( !f2 ) {
		cout << "add: Error: Failed to find field '" << destField << 
			"' on dest: " << dest->name() << "\n"; // Put name here.
		return 0;
	}

	if ( f1->checkTarget( f2 ) ) {
		const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >( f1 );
		const DestFinfo* df = dynamic_cast< const DestFinfo* >( f2 );
		if ( sf && df ) {
			fid = df->getFid();
			return sf;
		} else {
			cout << "Warning: validateMsg:: Source of message '" << 
				sf->name() << "' is not a SrcFinfo\n";
			return 0;
		}
	}
	return 0;
}
*/
