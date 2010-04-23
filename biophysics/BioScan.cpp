/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "BioScan.h"

#include "Compartment.h"
#include "HHChannel.h"
#include "CaConc.h"

void BioScan::initialize( Id object )
{
	ProcInfoBase p;
	SetConn c( object(), 0 );
	
	if ( object()->className() == "Compartment" )
		moose::Compartment::reinitFunc( &c, &p );
	else if ( object()->className() == "HHChannel" )
		HHChannel::reinitFunc( &c, &p );
	else if ( object()->className() == "CaConc" )
		CaConc::reinitFunc( &c, &p );
}

int BioScan::adjacent( Id compartment, Id exclude, vector< Id >& ret )
{
	int size = ret.size();
	adjacent( compartment, ret );
	ret.erase(
		remove( ret.begin(), ret.end(), exclude ),
		ret.end()
	);
	return ret.size() - size;
}

int BioScan::adjacent( Id compartment, vector< Id >& ret )
{
	int size = 0;
	size += targets( compartment, "axial", ret, "Compartment" );
	size += targets( compartment, "raxial", ret, "Compartment" );
	return size;
}

int BioScan::children( Id compartment, vector< Id >& ret )
{
	return targets( compartment, "axial", ret, "Compartment" );
}

/**
 * Gives all channels (hhchannels, synchans, any other) attached to a given
 * compartment.
 */
int BioScan::channels( Id compartment, vector< Id >& ret )
{
	return targets( compartment, "channel", ret );
}

int BioScan::hhchannels( Id compartment, vector< Id >& ret )
{
	// Request for elements of type "HHChannel" only since
	// channel messages can lead to synchans as well.
	return targets( compartment, "channel", ret, "HHChannel" );
}

int BioScan::gates( Id channel, vector< Id >& ret )
{
	vector< Id > gate;
	targets( channel, "xGate", gate, "HHGate" );
	targets( channel, "yGate", gate, "HHGate" );
	targets( channel, "zGate", gate, "HHGate" );
	ret.insert( ret.end(), gate.begin(), gate.end() );
	return gate.size();
}

int BioScan::spikegens( Id compartment, vector< Id >& ret )
{
	return targets( compartment, "VmSrc", ret, "SpikeGen" );
}

int BioScan::synchans( Id compartment, vector< Id >& ret )
{
	// "channel" msgs lead to SynChans as well HHChannels, so request
	// explicitly for former.
	return targets( compartment, "channel", ret, "SynChan" );
}

int BioScan::leakageChannels( Id compartment, vector< Id >& ret )
{
    return targets( compartment, "channel", ret, "Leakage" );
}
int BioScan::caTarget( Id channel, vector< Id >& ret )
{
	return targets( channel, "IkSrc", ret, "CaConc" );
}

int BioScan::caDepend( Id channel, vector< Id >& ret )
{
	return targets( channel, "concen", ret, "CaConc" );
}

/*
 * Functions for accessing gates' lookup tables.
 */

/**
 * Finds the xmin and xmax for the lookup tables (A and B) belonging to a gate.
 * 
 * 'min' will be the smaller of the 2 mins.
 * 'max' will be the greater of the 2 maxs.
 */
int BioScan::domain(
	Id gate,
	double& min,
	double& max )
{
	Id A;
	Id B;
	
	bool success;
	success = lookupGet< Id, string >( gate(), "lookupChild", A, "A" );
	if ( ! success ) {
		cerr << "Error: Interpol A not found as child of " << gate()->name();
		return 0;
	}
	
	success = lookupGet< Id, string >( gate(), "lookupChild", B, "B" );
	if ( ! success ) {
		cerr << "Error: Interpol B not found as child of " << gate()->name();
		return 0;
	}
	
	double Amin, Amax;
	double Bmin, Bmax;
	get< double >( A(), "xmin", Amin );
	get< double >( A(), "xmax", Amax );
	get< double >( B(), "xmin", Bmin );
	get< double >( B(), "xmax", Bmax );
	
	min = Amin < Bmin ? Amin : Bmin;
	max = Amax > Bmax ? Amax : Bmax;
	
	return 1;
}

void BioScan::rates(
	Id gate,
	const vector< double >& grid,
	vector< double >& A,
	vector< double >& B )
{
	A.resize( grid.size() );
	B.resize( grid.size() );
	
	vector< double >::const_iterator igrid;
	vector< double >::iterator ia = A.begin();
	vector< double >::iterator ib = B.begin();
	for ( igrid = grid.begin(); igrid != grid.end(); ++igrid ) {
		lookupGet< double, double >( gate(), "A", *ia, *igrid );
		lookupGet< double, double >( gate(), "B", *ib, *igrid );
		
		++ia, ++ib;
	}
}

int BioScan::modes( Id gate, int& AMode, int& BMode )
{
	Id A;
	Id B;
	
	bool success;
	success = lookupGet< Id, string >( gate(), "lookupChild", A, "A" );
	if ( ! success ) {
		cerr << "Error: Interpol A not found as child of " << gate()->name();
		return 0;
	}
	
	success = lookupGet< Id, string >( gate(), "lookupChild", B, "B" );
	if ( ! success ) {
		cerr << "Error: Interpol B not found as child of " << gate()->name();
		return 0;
	}
	
	get< int >( A(), "mode", AMode );
	get< int >( B(), "mode", BMode );
	return 1;
}

///////////////////////////////////////////////////
// Utility functions
///////////////////////////////////////////////////

int BioScan::targets(
	Id object,
	const string& msg,
	vector< Id >& target,
	const string& include,    // default value: ""
	const string& exclude )   // default value: ""
{
	vector< string > include_v;
	vector< string > exclude_v;
	
	if ( include != "" )
		include_v.push_back( include );
	
	if ( exclude != "" )
		exclude_v.push_back( exclude );
	
	return targets( object, msg, target, include_v, exclude_v );
}

int BioScan::targets(
	Id object,
	const string& msg,
	vector< Id >& target,
	const vector< string >& include,    // Mandatory to provide
	const vector< string >& exclude )   // default value: empty vector
{
	unsigned int oldSize = target.size();
	
	Id found;
	Conn* i = object()->targets( msg, 0 );
	for ( ; i->good(); i->increment() ) {
		found = i->target()->id();
		
		bool inInclude =
			find( include.begin(), include.end(), found()->className() ) != include.end();
		if ( ! include.empty() && ! inInclude )
			continue;
		
		bool inExclude =
			find( exclude.begin(), exclude.end(), found()->className() ) != exclude.end();
		if ( inExclude )
			continue;
		
		initialize( found );
		
		target.push_back( found );
	}
	delete i;
	
	return target.size() - oldSize;
}

////////////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"

void testBioScan( )
{
	cout << "\nTesting Bioscan" << flush;
	bool success;
	
	Element* n =
		Neutral::create( "Neutral", "n", Element::root()->id(), Id::scratchId() );
	
	/**
	 *  First we test the functions which return the compartments linked to a
	 *  given compartment: adjacent(), and children().
	 *  
	 *  A small tree is created for this:
	 *  
	 *               c0
	 *                L c1
	 *                   L c2
	 *                   L c3
	 *                   L c4
	 *                   L c5
	 *  
	 *  (c0 is the parent of c1. c1 is the parent of c2, c3, c4, c5.)
	 */
	Element* c[ 6 ];
	c[ 0 ] = Neutral::create( "Compartment", "c0", n->id(), Id::scratchId() );
	c[ 1 ] = Neutral::create( "Compartment", "c1", n->id(), Id::scratchId() );
	c[ 2 ] = Neutral::create( "Compartment", "c2", n->id(), Id::scratchId() );
	c[ 3 ] = Neutral::create( "Compartment", "c3", n->id(), Id::scratchId() );
	c[ 4 ] = Neutral::create( "Compartment", "c4", n->id(), Id::scratchId() );
	c[ 5 ] = Neutral::create( "Compartment", "c5", n->id(), Id::scratchId() );
	
	success = Eref( c[ 0 ] ).add( "axial", c[ 1 ], "raxial" );
	ASSERT( success, "Linking compartments" );
	success = Eref( c[ 1 ] ).add( "axial", c[ 2 ], "raxial" );
	ASSERT( success, "Linking compartments" );
	success = Eref( c[ 1 ] ).add( "axial", c[ 3 ], "raxial" );
	ASSERT( success, "Linking compartments" );
	success = Eref( c[ 1 ] ).add( "axial", c[ 4 ], "raxial" );
	ASSERT( success, "Linking compartments" );
	success = Eref( c[ 1 ] ).add( "axial", c[ 5 ], "raxial" );
	ASSERT( success, "Linking compartments" );
	
	vector< Id > found;
	unsigned int nFound;
	
	/** Testing version 1 of BioScan::adjacent.
	 *  It finds all neighbours of given compartment.
	 */
	// Neighbours of c0
	nFound = BioScan::adjacent( c[ 0 ]->id(), found );
	ASSERT( nFound == found.size(), "Finding adjacent compartments" );
	// c1 is adjacent
	ASSERT( nFound == 1, "Finding adjacent compartments" );
	ASSERT( found[ 0 ] == c[ 1 ]->id(), "Finding adjacent compartments" );
	
	// Neighbours of c1
	found.clear();
	nFound = BioScan::adjacent( c[ 1 ]->id(), found );
	ASSERT( nFound == 5, "Finding adjacent compartments" );
	// c0 is adjacent
	success =
		find( found.begin(), found.end(), c[ 0 ]->id() ) != found.end();
	ASSERT( success, "Finding adjacent compartments" );
	// c2 - c5 are adjacent
	for ( int i = 2; i < 6; i++ ) {
		success =
			find( found.begin(), found.end(), c[ i ]->id() ) != found.end();
		ASSERT( success, "Finding adjacent compartments" );
	}
	
	// Neighbours of c2
	found.clear();
	nFound = BioScan::adjacent( c[ 2 ]->id(), found );
	// c1 is adjacent
	ASSERT( nFound == 1, "Finding adjacent compartments" );
	ASSERT( found[ 0 ] == c[ 1 ]->id(), "Finding adjacent compartments" );
	
	/** Testing version 2 of BioScan::adjacent.
	 *  It finds all but one neighbours of given compartment.
	 *  The the second argument to 'adjacent' is the one that is excluded.
	 */
	// Neighbours of c1 (excluding c0)
	found.clear();
	nFound = BioScan::adjacent( c[ 1 ]->id(), c[ 0 ]->id(), found );
	ASSERT( nFound == 4, "Finding adjacent compartments" );
	// c2 - c5 are adjacent
	for ( int i = 2; i < 6; i++ ) {
		success =
			find( found.begin(), found.end(), c[ i ]->id() ) != found.end();
		ASSERT( success, "Finding adjacent compartments" );
	}
	
	// Neighbours of c1 (excluding c2)
	found.clear();
	nFound = BioScan::adjacent( c[ 1 ]->id(), c[ 2 ]->id(), found );
	ASSERT( nFound == 4, "Finding adjacent compartments" );
	// c0 is adjacent
	success =
		find( found.begin(), found.end(), c[ 0 ]->id() ) != found.end();
	ASSERT( success, "Finding adjacent compartments" );
	// c3 - c5 are adjacent
	for ( int i = 3; i < 6; i++ ) {
		success =
			find( found.begin(), found.end(), c[ i ]->id() ) != found.end();
		ASSERT( success, "Finding adjacent compartments" );
	}
	
	// Neighbours of c2 (excluding c1)
	found.clear();
	nFound = BioScan::adjacent( c[ 2 ]->id(), c[ 1 ]->id(), found );
	// None adjacent, if c1 is excluded
	ASSERT( nFound == 0, "Finding adjacent compartments" );
	
	// Neighbours of c2 (excluding c3)
	found.clear();
	nFound = BioScan::adjacent( c[ 2 ]->id(), c[ 3 ]->id(), found );
	// c1 is adjacent, while c3 is not even connected
	ASSERT( nFound == 1, "Finding adjacent compartments" );
	ASSERT( found[ 0 ] == c[ 1 ]->id(), "Finding adjacent compartments" );
	
	/** Testing BioScan::children.
	 *  It finds all compartments which are dests for the "axial" message.
	 */
	// Children of c0
	found.clear();
	nFound = BioScan::children( c[ 0 ]->id(), found );
	ASSERT( nFound == 1, "Finding child compartments" );
	// c1 is a child
	ASSERT( found[ 0 ] == c[ 1 ]->id(), "Finding child compartments" );
	
	// Children of c1
	found.clear();
	nFound = BioScan::children( c[ 1 ]->id(), found );
	ASSERT( nFound == 4, "Finding child compartments" );
	// c2 - c5 are c1's children
	for ( int i = 2; i < 6; i++ ) {
		success =
			find( found.begin(), found.end(), c[ i ]->id() ) != found.end();
		ASSERT( success, "Finding child compartments" );
	}
	
	// Children of c2
	found.clear();
	nFound = BioScan::children( c[ 2 ]->id(), found );
	// c2 has no children
	ASSERT( nFound == 0, "Finding child compartments" );
	
	// Clean up
	set( n, "destroy" );
}

#endif // DO_UNIT_TESTS
