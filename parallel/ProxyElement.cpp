/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "ProxyElement.h"

// #include "../shell/Shell.h"

ProxyElement::ProxyElement( Id id, unsigned int node, 
	unsigned int proxyFuncId, unsigned int size )
	: 	Element( id ), node_( node ),
		proxyVec_( FuncVec::getFuncVec( proxyFuncId ) ),
		numEntries_( size )
{
	id.setNode( node );
	// cout << "Making proxy for " << id << "." << node << " on node " << Shell::myNode() << endl << flush;
}

/**
 * The return value is undefined if msgNum is greater the numSrc but
 * below the # of entries in msg_
 * I could do a check on the cinfo, but that brings in too many dependencies
 * outside the SimpleElement data structures.
 */
unsigned int ProxyElement::numTargets( int msgNum ) const
{
	if ( msgNum >= 0 ) {
		if ( msgNum == 0 );
			return msg_.numTargets( this );
	} else {
		cout << "Proxy Element cannot yet handle incoming.\n";
		assert( 0 ); // Don't allow incoming, yet.
	}
	return 0;
}

void ProxyElement::sendData( unsigned int funcIndex, const char* data, 
	unsigned int eIndex )
{
	SetConn c( this, eIndex );
	ProxyFunc pf = reinterpret_cast< ProxyFunc >( 
		proxyVec_->func( funcIndex ) ); 
	pf( &c, data, Slot( 0, funcIndex ) );
}

void* ProxyElement::data( unsigned int eIndex ) const
{
	Eref pe = Id::postId( node_ ).eref();
	assert( pe.e != 0);
	return pe.data( );
}
