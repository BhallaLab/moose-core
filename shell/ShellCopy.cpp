/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "OneToAllMsg.h"
#include "Shell.h"

/// Returns the Id of the root of the copied tree.
Id Shell::doCopy( Id orig, Id newParent, string newName, unsigned int n, bool copyExtMsg )
{
	static const Finfo* reqf = 
		Shell::initCinfo()->findFinfo( "requestCopy" );
	static const SrcFinfo4< vector< Id >, string, unsigned int, bool >* 
		requestCopy = dynamic_cast< const SrcFinfo4< vector< Id >, string, unsigned int, bool >* >( reqf );

	if ( Neutral::isDescendant( newParent, orig ) ) {
		cout << "Error: Shell::doCopy: Cannot copy object to descendant in tree\n";
		return Id();
	}

	initAck();
	Eref sheller( shelle_, 0 );
	Id newElm = Id::nextId();
	vector< Id > args;
	args.push_back( orig );
	args.push_back( newParent );
	args.push_back( newElm );
	requestCopy->send( sheller, &p_, args, newName , n, copyExtMsg);
	while ( isAckPending() ) {
		Qinfo::mpiClearQ( &p_ );
	}

	return newElm;
}

Element* innerCopyElements( Id orig, Id newParent, Id newElm, 
	unsigned int n, map< Id, Id >& tree )
{
	static const Finfo* pf = Neutral::initCinfo()->findFinfo( "parentMsg" );
	static const Finfo* cf = Neutral::initCinfo()->findFinfo( "childMsg" );

	Element* e = new Element( newElm, orig(), n );
	Msg* m = new OneToAllMsg( newParent.eref(), e );
	assert( m );
	if ( !cf->addMsg( pf, m->mid(), newParent() ) ) {
		cout << "copy: Error: unable to add parent->child msg from " <<
			newParent()->getName() << " to " << e->getName() << "\n";
		return 0;
	}
	tree[ orig ] = e->id();

	const Neutral* origData = reinterpret_cast< const Neutral* >(
		orig.eref().data() );
	vector< Id > kids = origData->getChildren( orig.eref(), 0 );

	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i ) {
		innerCopyElements( *i, e->id(), Id::nextId(), n, tree );
	}
	return e;
}

/*
void Shell::innerCopyData( Id orig, Id newParent )
{
}

void Shell::innerCopyData( Id orig, Id newParent )
{
}

void Shell::innerCopyMsgs( shelle, Id orig )
{
}
*/

void Shell::handleCopy( vector< Id > args, string newName,
	unsigned int n, bool copyExtMsgs )
{
	static const Finfo* ackf = 
		Shell::initCinfo()->findFinfo( "ack" );
	static const SrcFinfo2< unsigned int, unsigned int >* 
		ack = dynamic_cast< const SrcFinfo2< unsigned int, unsigned int >* >( ackf );

	map< Id, Id > tree;
	// args are orig, newParent, newElm.
	assert( args.size() == 3 );
	Element* e = innerCopyElements( args[0], args[1], args[2], n, tree );
	if ( !e ) {
		ack->send( 
			Eref( shelle_, 0 ), &p_, Shell::myNode(), ErrorStatus, 0 );
		return;
	}
	if ( newName != "" )
		e->setName( newName );
	//innerCopyData( orig, newParent );
	// innerCopyMsgs( tree, copyExtMsgs );
	ack->send( Eref( shelle_, 0 ), &p_, Shell::myNode(), OkStatus, 0 );
}
