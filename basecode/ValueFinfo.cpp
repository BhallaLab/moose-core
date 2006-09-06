/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

void solverUpdate( const Element* e, const Finfo* f, SolverOp mode )
{
	e->solverUpdate( f, mode );
}

// Returns an index of 0 if the name is matched without any braces.
// Otherwise looks for index in square braces. 
// Returns -1 if it fails.
long findIndex(const string& s, const string& name)
{
	if (s == name)
		return 0;
	if (s.find(name) == 0) {
		string temp = s.substr(name.length());
		if (temp[0] == '[') {
			unsigned long close_brace = temp.find(']');
			if (close_brace == string::npos)
				return -1;
			temp = temp.substr(1, close_brace - 1);
			char* endptr;
			long i = strtol( temp.c_str(), &endptr, 10);
			if (endptr == temp.c_str())
				return -1;
			if (i > 0) {
				return i;
			} else if (i == 0) {
				return 0;
			} else {
				return -1;
			}
		}
	}
	return -1;
}

// The add and respondToAdd functions have to handle a number
// of cases.
// Case 1: Assigning a value to this field. Here, respondToAdd
// is called and returns a a RelayFinfo1< T >.
// Case 2: Sending a value message from the field to a target.
//   Here, add is called and either creates or looks up a 
//   ValueRelayFinfo
//   This must be done as a pair with receiving a trigger 
//   message, because both operations use the same
//   ValueRelayFinfo.
// Case 3: Receiving a trigger message requesting a value.
//   Here, respondToAdd is called from the trigger.
//   It needs a ValueRelayInfo to recieve the trigger, and also
//   to send out the value.
// In cases 2 and 3, we need to check if there is an existing
// ValueRelayFinfo that awaits the other member of the pair.
// Otherwise we make a new ValueRelayFinfo.
// Note that the existing ValueRelayFinfo need not be 
// sending and getting info to the same object. So we 
// cannot use the target identity as a check.

bool valueFinfoAdd( 
	Finfo* vf, Finfo *( *createValueRelayFinfo )( Field& ),
	Element* e, Field& destfield, bool useSharedConn )
{
	Finfo* f = destfield.respondToAdd( vf );
	if ( f ) {
		// Look for a relay already receiving a trigger message
		Finfo* rf = e->findVacantValueRelay( vf, 0 );
		if ( !rf ) {
			Field temp( vf, e );
			rf = createValueRelayFinfo( temp );
			e->appendRelay( rf );
		}
		return rf->add( e, destfield );
	}
	return 0;
}

Finfo* valueFinfoRespondToAdd( 
	Finfo* vf, Finfo *( *createValueRelayFinfo )( Field& ),
	Finfo *( *createRelayFinfo )( Field& ),
	Element* e, const Finfo* sender )
{
	static Ftype0 f0;
	Finfo* rf;
	Finfo* ret = 0;
	Field temp( vf, e );
	if ( vf->isSameType( sender ) ) {
		ret = createRelayFinfo( temp );
		e->appendRelay( ret );
		return ret;
	} else if ( sender->isSameType( &f0 ) ) {
		// Type check trigger
		// Look for a relay sending out a value message
		rf = e->findVacantValueRelay( vf, 1 );
		if ( !rf ) {
			// Field temp( vf, e );
			rf = createValueRelayFinfo( temp );
			e->appendRelay( rf );
		}
		return rf;
	} else { // set up shared conn stuff.
		vector< Finfo* > finfos;
		FinfoDummy dummy( "dummy" );
		finfos.push_back( vf );
		finfos.push_back( &dummy );
		MultiFtype shared( finfos );
		if ( sender->isSameType( &shared ) ) {
			Field temp( vf, e );
			rf = createValueRelayFinfo( temp );
			e->appendRelay( rf );
			return rf;
		}
	}
	return 0;
}


Finfo* obtainValueRelay( 
	Finfo *( *createValueRelayFinfo )( Field& ),
	Finfo* f, Element* e, bool isSending )
{
	Finfo* rf = e->findVacantValueRelay( f, isSending );
	if ( !rf ) {
		Field temp( f, e );
		rf = createValueRelayFinfo( temp );
		if ( rf )
			e->appendRelay( rf );
	}
	return rf;
}

