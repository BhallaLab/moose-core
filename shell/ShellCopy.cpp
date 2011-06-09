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
		Shell::initCinfo()->findFinfo( "copy" );
	static const SrcFinfo4< vector< Id >, string, unsigned int, bool >* 
		requestCopy = dynamic_cast< const SrcFinfo4< vector< Id >, string, unsigned int, bool >* >( reqf );
	assert( reqf );
	assert( requestCopy );

	if ( n > 1 && numNodes() > 1 && 
		( !orig()->dataHandler()->isGlobal() ) ) {
		cout << "Error: Shell::doCopy( " << orig.path() << " to " <<
			newParent.path() << 
			":\nCannot array copy local object across nodes (yet)\n";
		return Id();
	}
	if ( Neutral::isDescendant( newParent, orig ) ) {
		cout << "Error: Shell::doCopy: Cannot copy object to descendant in tree\n";
		return Id();
	}

	Eref sheller( shelle_, 0 );
	Id newElm = Id::nextId();
	vector< Id > args;
	args.push_back( orig );
	args.push_back( newParent );
	args.push_back( newElm );
	initAck();
		requestCopy->send( sheller, &p_, args, newName , n, copyExtMsg);
	waitForAck();

	return newElm;
}

/// Runs in parallel on all nodes.
Element* innerCopyElements( Id orig, Id newParent, Id newElm, 
	unsigned int n, map< Id, Id >& tree )
{
	// static const Finfo* pf = Neutral::initCinfo()->findFinfo( "parentMsg" );
	// static const Finfo* cf = Neutral::initCinfo()->findFinfo( "childMsg" );

	Element* e = new Element( newElm, orig(), n );
	assert( e );
	Shell::adopt( newParent, newElm );

	// Nasty post-creation fix for FieldElements, which need to assign
	// parent DataHandlers.
	FieldDataHandlerBase* fdh =
		dynamic_cast< FieldDataHandlerBase* >( e->dataHandler() );
	if ( fdh ) {
		fdh->assignParentDataHandler( newParent()->dataHandler() );
	}


	// cout << Shell::myNode() << ": Copy: orig= " << orig << ", newParent = " << newParent << ", newElm = " << newElm << endl;
	tree[ orig ] = e->id();

	// cout << Shell::myNode() << ": ice, pa = " << newParent << ", pair= (" << orig << "," << e->id() << ")\n";

	/*
	const Neutral* origData = reinterpret_cast< const Neutral* >(
		orig.eref().data() );
	vector< Id > kids = origData->getChildren( orig.eref(), 0 );
	*/
	vector< Id > kids;
	Neutral::children( orig.eref(), kids );

	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i ) {
		innerCopyElements( *i, e->id(), Id::nextId(), n, tree );
	}
	return e;
}

void innerCopyMsgs( map< Id, Id >& tree, unsigned int n, bool copyExtMsgs )
{
	static const Finfo* cf = Neutral::initCinfo()->findFinfo( "childMsg" );
	static const SrcFinfo1< int >* cf2 = 
		dynamic_cast< const SrcFinfo1< int >* >( cf );
	assert( cf );
	assert( cf2 );

	/*
	cout << endl << Shell::myNode() << ": innerCopyMsg ";
	for ( map< Id, Id >::const_iterator i = tree.begin(); 
		i != tree.end(); ++i ) {
		cout << " (" << i->first << "," << i->second << ") ";
	}
	cout << endl;
	*/
	for ( map< Id, Id >::const_iterator i = tree.begin(); 
		i != tree.end(); ++i ) {
		Element *e = i->first.operator()();
		unsigned int j = 0;
		const vector< MsgFuncBinding >* b = e->getMsgAndFunc( j );
		while ( b ) {
			if ( j != cf2->getBindIndex() ) {
				for ( vector< MsgFuncBinding >::const_iterator k = 
					b->begin();
					k != b->end(); ++k ) {
					MsgId mid = k->mid;
					const Msg* m = Msg::getMsg( mid );
					assert( m );
	/*
	cout << endl << Shell::myNode() << ": innerCopyMsg orig = (" <<
		e->id() << ", " << e->getName() << "), e1 = (" <<
		m->e1()->id() << ", " << m->e1()->getName() << "), e2 = (" <<
		m->e2()->id() << ", " << m->e2()->getName() << "), fid = " <<
		k->fid << ", mid = " << k->mid << endl;
		*/
					map< Id, Id >::const_iterator tgt;
					if ( m->e1() == e ) {
						tgt = tree.find( m->e2()->id() );
					} else if ( m->e2() == e ) {
						tgt = tree.find( m->e1()->id() );
					} else {
						assert( 0 );
					}
					if ( tgt != tree.end() )
						m->copy( e->id(), i->second, tgt->second, 
							k->fid, j, n );
				}
			}
			b = e->getMsgAndFunc( ++j );
		}
	}
}

// #define CHECK_TREE

void Shell::handleCopy( const Eref& er, const Qinfo* q,
	vector< Id > args, string newName,
	unsigned int n, bool copyExtMsgs )
{
	static const Finfo* ackf = 
		Shell::initCinfo()->findFinfo( "ack" );
	static const SrcFinfo2< unsigned int, unsigned int >* 
		ack = dynamic_cast< const SrcFinfo2< unsigned int, unsigned int >* >( ackf );
	assert( ackf );
	assert( ack );

	if ( q->addToStructuralQ() )
		return;

	/// Test here for tree structure on different nodes
#ifdef CHECK_TREE
	vector< Id > temp;
	Eref baser = args[0].eref();
	const Neutral* basen = reinterpret_cast< const Neutral *>( 
		baser.data() );
	unsigned int nret = basen->buildTree( baser, q, temp );
	assert( nret == temp.size() );
	cout << endl << Shell::myNode() << 
		": handleCopy: ntree= " << nret << endl;
	for ( unsigned int i = 0; i < nret; ++i )
		cout << " (" << i << ", " << temp[i] << ") " << 
		temp[i]()->getName() << "\n";
#endif

	map< Id, Id > tree;
	// args are orig, newParent, newElm.
	assert( args.size() == 3 );
	Element* e = innerCopyElements( args[0], args[1], args[2], n, tree );
	if ( !e ) {
		ack->send( 
			Eref( shelle_, 0 ), &p_, Shell::myNode(), ErrorStatus );
		return;
	}
	if ( newName != "" )
		e->setName( newName );
	//innerCopyData( orig, newParent );
	innerCopyMsgs( tree, n, copyExtMsgs );
	ack->send( Eref( shelle_, 0 ), &p_, Shell::myNode(), OkStatus );
}
