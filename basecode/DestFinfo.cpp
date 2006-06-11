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
//////////////////////////////////////////////////////////////////
// Utility functions
//////////////////////////////////////////////////////////////////

// Goes through trigger list and connects up FinfoConns on the
// trigger and triggeree to be tied to each other as if they were
// a regular message.
// The triggers string is a comma/space separated list of finfo names
// on the current Cinfo.
// Called from initialize of DestFinfos to set outgoing msgsrcs
// that they feed into. Essentially connects messages between
// finfos.

void parseTriggerList(
	const Cinfo* c, const string& triggers, vector< Finfo* >& list )
{
	string s = triggers;
	size_t i = s.find_first_not_of( ", " );
	if (i > 0 && i != string::npos)
		s = s.substr( i );
	while ( s.length() > 0 ) {
		i = s.find_first_of( ", " );
		string t = s.substr( 0, i );
		Field f = c->field( t );
		if ( f.good() ) {
			list.push_back( f->copy() );
		} else {
			cerr << "Error: parseTriggerList: Could not find field '" <<
				t << "' on class '" << c->name() << "'\n";
		}

		if (i != string::npos ) {
			s = s.substr( i + 1 );
			i = s.find_first_not_of( ", " );
			s = s.substr( i );
		} else {
			break;
		}
	}
}

//////////////////////////////////////////////////////////////////
// DestFinfo functions
//////////////////////////////////////////////////////////////////

// Returns nonzero if the remote func belongs to one of
// its targets. In this case this means that if the func
// has come from one of the SrcFinfos triggered by this dest,
// return 1.
unsigned long DestFinfo::matchRemoteFunc(
	Element* e, RecvFunc func ) const
{
	unsigned long ret = 0;
	for ( unsigned long i = 0; i < internalDest_.size(); i++ ) {
		if ( internalDest_[ i ]->recvFunc() == func )
			ret++;
	}
	return ret;
}

RecvFunc DestFinfo::targetFunc( Element* e, unsigned long i ) const
{
	if ( i < internalDest_.size() )
		return internalDest_[ i ]->recvFunc();
	return 0;
}

void DestFinfo::addRecvFunc( Element* e, RecvFunc rf,
			unsigned long position ) {
	cerr << "DestFinfo::addRecFunc on " << e->path() << 
		": Cannot add func to msg dest\n";
}

void DestFinfo::dest( vector< Field >& list, Element* e ) {
	vector < Finfo* >::iterator i;
	for ( i = internalDest_.begin();
		i != internalDest_.end(); i++)
		list.push_back( Field( *i, e ) );
}

Finfo* DestFinfo::respondToAdd( Element* e, const Finfo* sender ) {
	if ( sharesConn_ ) {
	// Need to insert dummy RecvFuncs in any shared MsgSrcs.
	// Locate Shared Finfo
	// Scan through its list of SharedOut, put dummies on them.
	// Or, as here, simply create a RelayFinfo to deal with it.
		cerr << "Warning:DestFinfo::respondToAdd::" << e->name() <<
			"." << name() << ", " << sender->name() << "\n";
		cout << "Shared message handling not yet implementd, trying relay\n";
		return makeRelayFinfo( e );
	}
	if ( isSameType( sender ) )
		return this;
	return 0;
}

bool DestFinfo::strGet( Element* e, string& val )
{
	vector< Field > list;
	src( list, e );
	val = "";
	for (unsigned int i = 0; i < list.size(); i++) {
		val += list[ i ].path();
		if ( i < list.size() - 1 )
			val += ", ";
	}
	return 1;
}

//////////////////////////////////////////////////////////////////
// Synapse functions
//////////////////////////////////////////////////////////////////
// Here we override the src function for Synapses. We provide
// either the single source or all of them, depending on the
// field status.
void synSrc( 
	vector< Field >& list, 
	vector< Conn* >& conns, 
	RecvFunc f,
	unsigned long index,
	bool isIndex
)
{
	if ( isIndex ) {
		if ( conns.size() > index ) {
			Conn* t = conns[ index ]->target( 0 );
			if ( t )
				list.push_back( t->parent()->lookupSrcField( t, f ));
		}
	} else {
		vector< Conn* >::iterator i;
		for ( i = conns.begin(); i != conns.end(); i++) {
			Conn* t = ( *i )->target( 0 );
			if ( t )
				list.push_back( t->parent()->lookupSrcField( t, f ));
		}
	}
}

/*
unsigned long matchRemoteFunc( Element* e, RecvFunc func ) const
{
	unsigned long ret = 0;
	for ( unsigned long i = 0; i < internalDest_.size(); i++ ) {
		if ( internalDest_[ i ]->recvFunc() == func )
			ret++;
	}
	return ret;
}
*/
