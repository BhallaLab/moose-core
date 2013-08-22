/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Shell.h"
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "../scheduling/Tick.h"
#include "../scheduling/TickMgr.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"
#include "../scheduling/testScheduling.h"

#include "../builtins/Arith.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "SingleMsg.h"
#include "OneToAllMsg.h"
#include "Wildcard.h"


/**
 * Tests Create and Delete calls issued through the parser interface,
 * which internally sets up blocking messaging calls.
 */
void testShellParserCreateDelete()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	vector< int > dimensions;
	dimensions.push_back( 1 );
	Id child = shell->doCreate( "Neutral", Id(), "test", dimensions );

	shell->doDelete( child );
	cout << "." << flush;
}

/**
 * Tests traversal through parents and siblings.
 */
void testTreeTraversal()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	vector< int > dimensions;
	dimensions.push_back( 1 );
	Id f1 = shell->doCreate( "Neutral", Id(), "f1", dimensions );
	assert( f1 != Id() );
	Id f2a = shell->doCreate( "Neutral", f1, "f2a", dimensions );
	assert( f2a != Id() );
	Id f2b = shell->doCreate( "Neutral", f1, "f2b", dimensions );
	assert( f2b != Id() );
	Id f2c = shell->doCreate( "Neutral", f1, "f2c", dimensions );
	assert( f2c != Id() );
	Id f3aa = shell->doCreate( "Neutral", f2a, "f3aa", dimensions );
	assert( f3aa != Id() );
	Id f3ab = shell->doCreate( "Neutral", f2a, "f3ab", dimensions );
	assert( f3ab != Id() );
	Id f3ba = shell->doCreate( "Neutral", f2b, "f3ba", dimensions );
	assert( f3ba != Id() );

	////////////////////////////////////////////////////////////////
	// Checking for own Ids
	////////////////////////////////////////////////////////////////
	ObjId me = Field< ObjId >::get( f3aa, "me" );
	assert( me == ObjId( f3aa, 0 ) );
	me = Field< ObjId >::get( f3ba, "me" );
	assert( me == ObjId( f3ba, 0 ) );
	me = Field< ObjId >::get( f2c, "me" );
	assert( me == ObjId( f2c, 0 ) );

	////////////////////////////////////////////////////////////////
	// Checking for parent Ids
	////////////////////////////////////////////////////////////////
	ObjId pa = Field< ObjId >::get( f3aa, "parent" );
	assert( pa == ObjId( f2a, 0 ) );
	pa = Field< ObjId >::get( f3ab, "parent" );
	assert( pa == ObjId( f2a, 0 ) );
	pa = Field< ObjId >::get( f2b, "parent" );
	assert( pa == ObjId( f1, 0 ) );
	pa = Field< ObjId >::get( f1, "parent" );
	assert( pa == ObjId( Id(), 0 ) );

	cout << "." << flush;

	////////////////////////////////////////////////////////////////
	// Checking for child Id lists
	////////////////////////////////////////////////////////////////
	vector< Id > kids = Field< vector< Id > >::get( f1, "children" );
	assert( kids.size() == 3 );
	assert( kids[0] == f2a );
	assert( kids[1] == f2b );
	assert( kids[2] == f2c );

	kids = Field< vector< Id > >::get( f2a, "children" );
	assert( kids.size() == 2 );
	assert( kids[0] == f3aa );
	assert( kids[1] == f3ab );
	
	kids = Field< vector< Id > >::get( f2b, "children" );
	assert( kids.size() == 1 );
	assert( kids[0] == f3ba );

	kids = Field< vector< Id > >::get( f2c, "children" );
	assert( kids.size() == 0 );

	cout << "." << flush;

	////////////////////////////////////////////////////////////////
	// Checking path string generation.
	////////////////////////////////////////////////////////////////
	string path = Field< string >::get( f3aa, "path" );
	assert( path == "/f1/f2a/f3aa" );
	path = Field< string >::get( f3ab, "path" );
	assert( path == "/f1/f2a/f3ab" );
	path = Field< string >::get( f3ba, "path" );
	assert( path == "/f1/f2b/f3ba" );

	path = Field< string >::get( f2a, "path" );
	assert( path == "/f1/f2a" );
	path = Field< string >::get( f2b, "path" );
	assert( path == "/f1/f2b" );
	path = Field< string >::get( f2c, "path" );
	assert( path == "/f1/f2c" );
	path = Field< string >::get( f1, "path" );
	assert( path == "/f1" );
	path = Field< string >::get( Id(), "path" );
	assert( path == "/" );

	cout << "." << flush;
	////////////////////////////////////////////////////////////////
	// Checking finding Ids from path
	////////////////////////////////////////////////////////////////

	shell->setCwe( Id() );
	assert( shell->doFind( "/f1/f2a/f3aa" ) == f3aa );
	assert( shell->doFind( "/f1/f2a/f3ab" ) == f3ab );
	assert( shell->doFind( "/f1/f2b/f3ba" ) == f3ba );
	assert( shell->doFind( "/f1/f2b/f3ba/" ) == f3ba );
	assert( shell->doFind( "/f1/f2b/f3ba/.." ) == f2b );
	assert( shell->doFind( "f1/f2b/f3ba" ) == f3ba );
	assert( shell->doFind( "./f1/f2b/f3ba" ) == f3ba );
	assert( shell->doFind( "./f1/f2b/../f2a/f3aa" ) == f3aa );

	shell->setCwe( f2b );
	assert( shell->doFind( "." ) == f2b );
	assert( shell->doFind( "f3ba" ) == f3ba );
	assert( shell->doFind( "f3ba/" ) == f3ba );
	assert( shell->doFind( "f3ba/.." ) == f2b );
	assert( shell->doFind( "f3ba/../../f2a/f3aa" ) == f3aa );
	assert( shell->doFind( "../f2a/f3ab" ) == f3ab );
	
	cout << "." << flush;
	////////////////////////////////////////////////////////////////
	// Checking getChild
	////////////////////////////////////////////////////////////////
	// Neutral* f1data = reinterpret_cast< Neutral* >( f1.eref().data() );
	assert( f2a == Neutral::child( f1.eref(), "f2a" ) );
	assert( f2b == Neutral::child( f1.eref(), "f2b" ) );
	assert( f2c == Neutral::child( f1.eref(), "f2c" ) );
        shell->setCwe( Id() );
	shell->doDelete( f1 );
	cout << "." << flush;
}

/// Test the Neutral::isDescendant
void testDescendant()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	vector< int > dimensions;
	dimensions.push_back( 1 );
	Id f1 = shell->doCreate( "Neutral", Id(), "f1", dimensions );
	Id f2a = shell->doCreate( "Neutral", f1, "f2a", dimensions );
	Id f2b = shell->doCreate( "Neutral", f1, "f2b", dimensions );
	Id f3aa = shell->doCreate( "Neutral", f2a, "f3aa", dimensions );

	assert( Neutral::isDescendant( f3aa, Id() ) );
	assert( Neutral::isDescendant( f3aa, f1 ) );
	assert( Neutral::isDescendant( f3aa, f2a ) );
	assert( Neutral::isDescendant( f3aa, f3aa ) );
	assert( !Neutral::isDescendant( f3aa, f2b ) );

	assert( Neutral::isDescendant( f2b, Id() ) );
	assert( Neutral::isDescendant( f2b, f2b ) );
	assert( !Neutral::isDescendant( f2b, f2a ) );
	assert( !Neutral::isDescendant( f2b, f3aa ) );

	assert( Neutral::isDescendant( f1, Id() ) );
	assert( Neutral::isDescendant( f1, f1 ) );
	assert( !Neutral::isDescendant( f1, f2a ) );
	assert( !Neutral::isDescendant( f1, f2b ) );
	assert( !Neutral::isDescendant( f1, f3aa ) );

	shell->doDelete( f1 );
	cout << "." << flush;
}

/// Utility function to check a commonly used tree structure
void verifyKids( Id f1, Id f2a, Id f2b, Id f3, Id f4a, Id f4b )
{
	Neutral* f1Data = reinterpret_cast< Neutral* >( f1.eref().data() );
	vector< Id > kids;
 	f1Data->children( f1.eref(), kids );
	assert( kids.size() == 2 );
	assert( kids[0] == f2a );
	assert( kids[1] == f2b );

	vector< Id > tree;
	Qinfo q;
	// Neutral::buildTree is depth-first.
	int num = f1Data->buildTree( f1.eref(), &q, tree );
	assert( num == 6 );
	assert( tree.size() == 6 );
	assert( tree[5] == f1 );
	assert( tree[3] == f2a );
	assert( tree[2] == f3 );
	assert( tree[0] == f4a );
	assert( tree[1] == f4b );
	assert( tree[4] == f2b );
}

/// Test the Neutral::children and buildTree
void testChildren()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	vector< int > dimensions;
	dimensions.push_back( 1 );
	Id f1 = shell->doCreate( "Neutral", Id(), "f1", dimensions );
	Id f2a = shell->doCreate( "Neutral", f1, "f2a", dimensions );
	Id f2b = shell->doCreate( "Neutral", f1, "f2b", dimensions );
	Id f3 = shell->doCreate( "Neutral", f2a, "f3", dimensions );
	Id f4a = shell->doCreate( "Neutral", f3, "f4a", dimensions );
	Id f4b = shell->doCreate( "Neutral", f3, "f4b", dimensions );

	verifyKids( f1, f2a, f2b, f3, f4a, f4b );

	shell->doDelete( f1 );
	cout << "." << flush;
}

void testMove()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	vector< int > dimensions;
	dimensions.push_back( 1 );
	/*
	Id f1 = shell->doCreate( "Neutral", Id(), "f1", dimensions );
	Id f2a = shell->doCreate( "Neutral", f1, "f2a", dimensions );
	Id f2b = shell->doCreate( "Neutral", f1, "f2b", dimensions );
	Id f3aa = shell->doCreate( "Neutral", f2a, "f3aa", dimensions );
	*/

	Id f1 = shell->doCreate( "Neutral", Id(), "f1", dimensions );
	Id f2a = shell->doCreate( "Neutral", f1, "f2a", dimensions );
	Id f2b = shell->doCreate( "Neutral", f1, "f2b", dimensions );
	Id f3 = shell->doCreate( "Neutral", f2a, "f3", dimensions );
	Id f4a = shell->doCreate( "Neutral", f3, "f4a", dimensions );
	Id f4b = shell->doCreate( "Neutral", f3, "f4b", dimensions );
	verifyKids( f1, f2a, f2b, f3, f4a, f4b );

	ObjId pa = Field< ObjId >::get( f4a, "parent" );
	assert( pa == ObjId( f3, 0 ) );
	pa = Field< ObjId >::get( f2a, "parent" );
	assert( pa == ObjId( f1, 0 ) );
	string path = Field< string >::get( f4a, "path" );
	assert( path == "/f1/f2a/f3/f4a" );
	Neutral* f1data = reinterpret_cast< Neutral* >( f1.eref().data() );

	vector< Id > kids = f1data->getChildren( f1.eref(), 0 );
	assert( kids.size() == 2 );
	assert( kids[0] == f2a );
	assert( kids[1] == f2b );

	Neutral* f3data = reinterpret_cast< Neutral* >( f3.eref().data() );
	kids = f3data->getChildren( f3.eref(), 0 );
	assert( kids.size() == 2 );
	assert( kids[0] == f4a );
	assert( kids[1] == f4b );

	//////////////////////////////////////////////////////////////////
	shell->doMove( f4a, f1 );
	//////////////////////////////////////////////////////////////////

	pa = Field< ObjId >::get( f4a, "parent" );
	assert( pa == ObjId( f1, 0 ) );

	kids = f1data->getChildren( f1.eref(), 0 );
	assert( kids.size() == 3 );
	assert( kids[0] == f2a );
	assert( kids[1] == f2b );
	assert( kids[2] == f4a );
	kids = f3data->getChildren( f3.eref(), 0 );
	assert( kids.size() == 1 );
	assert( kids[0] == f4b );

	//////////////////////////////////////////////////////////////////
	shell->doMove( f2a, f4a );
	//////////////////////////////////////////////////////////////////
	pa = Field< ObjId >::get( f2a, "parent" );
	assert( pa == ObjId( f4a, 0 ) );
	path = Field< string >::get( f4b, "path" );
	assert( path == "/f1/f4a/f2a/f3/f4b" );

	kids = f1data->getChildren( f1.eref(), 0 );
	assert( kids[0] == f2b );
	assert( kids[1] == f4a );

	shell->doDelete( f1 );
	cout << "." << flush;
}

void testCopy()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	vector< int > dimensions;
	dimensions.push_back( 1 );
	Id f1 = shell->doCreate( "Neutral", Id(), "f1", dimensions );
	Id f2a = shell->doCreate( "Neutral", f1, "f2a", dimensions );
	Id f2b = shell->doCreate( "Neutral", f1, "f2b", dimensions );
	Id f3 = shell->doCreate( "Neutral", f2a, "f3", dimensions );
	Id f4a = shell->doCreate( "Neutral", f3, "f4a", dimensions );
	Id f4b = shell->doCreate( "Neutral", f3, "f4b", dimensions );

	verifyKids( f1, f2a, f2b, f3, f4a, f4b );

	ObjId pa = Field< ObjId >::get( f3, "parent" );
	assert( pa == ObjId( f2a, 0 ) );
	pa = Field< ObjId >::get( f2a, "parent" );
	assert( pa == ObjId( f1, 0 ) );
	string path = Field< string >::get( f3, "path" );
	assert( path == "/f1/f2a/f3" );

	//////////////////////////////////////////////////////////////////
	Id dupf2a = shell->doCopy( f2a, Id(), "TheElephantsAreLoose", 1, false, false );
	//////////////////////////////////////////////////////////////////

	verifyKids( f1, f2a, f2b, f3, f4a, f4b );

	assert( dupf2a != Id() );
	// pa = Field< ObjId >::get( dupf2a, "parent" );
	// assert( pa == ObjId( f1, 0 ) );
	assert( dupf2a()->getName() == "TheElephantsAreLoose" );
	Neutral* f2aDupData = reinterpret_cast< Neutral* >( dupf2a.eref().data() );
	Id dupf3 = f2aDupData->child( dupf2a.eref(), "f3" );
	assert( dupf3 != Id() );
	assert( dupf3 != f3 );
	assert( dupf3()->getName() == "f3" );
	vector< Id > kids = f2aDupData->getChildren( dupf2a.eref(), 0 );
	assert( kids.size() == 1 );
	assert( kids[0] == dupf3 );

	Neutral* f3DupData = reinterpret_cast< Neutral* >( dupf3.eref().data());
	assert( f3DupData->getParent( dupf3.eref(), 0 ) == ObjId( dupf2a, 0 ));
	kids = f3DupData->getChildren( dupf3.eref(), 0 );
	assert( kids.size() == 2 );
	assert( kids[0]()->getName() == "f4a" );
	assert( kids[1]()->getName() == "f4b" );

	shell->doDelete( f1 );
	shell->doDelete( dupf2a );
	cout << "." << flush;
}

void testCopyFieldElement()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	unsigned int size = 10;
	vector< int > dims( 1, size );
	Id origId = shell->doCreate( "IntFire", Id(), "f1", dims );
	Id origSynId( origId.value() + 1 );
	
	Element* syn = origSynId();
	assert( syn != 0 );
	assert( syn->getName() == "synapse" );
	assert( syn->dataHandler()->totalEntries() == size * 65536 );
	assert( syn->dataHandler()->localEntries() == 0 );
	vector< unsigned int > vec(size);
	for ( unsigned int i = 0; i < size; ++i )
		vec[i] = i;
	bool ret = Field< unsigned int >::setVec( origId, "numSynapses", vec );
	assert( ret );

	unsigned int numHere = 0;
	if ( shell->numNodes() == 1 ) {
		numHere = size * (size - 1) / 2;
	} else {
		for ( unsigned int i = 0; i < size; ++i ) {
			if ( origId()->dataHandler()->isDataHere( i ) ) {
				numHere += i;
			}
		}
	}
	assert( syn->dataHandler()->localEntries() == numHere );
	syn->syncFieldDim();

	assert( syn->dataHandler()->totalEntries() == ( size - 1 ) * size );

	vector< double > delay( size * ( size - 1 ), 0 );
	// unsigned int k = 0;
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			// delay[ k++ ] = 3.14 * j + i * i;
			delay[ i * (size - 1 ) + j ] = 3.14 * j + i * i;
		}
	}
	ret = Field< double >::setVec( origSynId, "delay", delay );
	assert( ret );

	//////////////////////////////////////////////////////////////////
	Id copyId = shell->doCopy( origId, Id(), "dup", 1, false, false );
	//////////////////////////////////////////////////////////////////

	Eref origEr( origId(), 0 );
	Eref copyEr( copyId(), 0 );
	assert( origId()->dataHandler() != copyId()->dataHandler() );
	vector< Id > origChildren;
	vector< Id > copyChildren;
	Neutral::children( origEr, origChildren );
	Neutral::children( copyEr, copyChildren );
	assert( origChildren.size() == 1 );
	assert( origChildren[0] == origSynId );
	assert( origChildren.size() == copyChildren.size() );
	Id copySynId = copyChildren[0];

	Element* copySynElm = copySynId();
	
	// Element should exist even if data doesn't
	assert ( copySynElm != 0 );
	assert ( copySynElm->getName() == "synapse" );
	// assert( syn->dataHandler()->data( 0 ) != 0 ); // Should warn
	// assert( copySynElm->dataHandler()->data( 0 ) != 0 ); // Should warn
	assert( copySynElm->dataHandler()->localEntries() == numHere );
	assert( copySynElm->dataHandler()->totalEntries() ==
		( size * (size - 1) ) );
	
	vector< double > del;
	Field< double >::getVec( origSynId, "delay", del );
	assert( del.size() == size * (size - 1 ) );
	assert( del.size() == delay.size() );

	for ( unsigned int i = 0; i < del.size(); ++i ) {
		// cout << i << "	" << del[i] << "	" << delay[i] << endl;
		assert( doubleEq( del[i], delay[i] ) );
	}

	del.resize( 0 );
	Field< double >::getVec( copySynId, "delay", del );
	assert( del.size() == size * (size - 1 ) );
	assert( del.size() == delay.size() );

	for ( unsigned int i = 0; i < del.size(); ++i ) {
		// cout << i << "	" << del[i] << "	" << delay[i] << endl;
		assert( doubleEq( del[i], delay[i] ) );
	}
	
	shell->doDelete( origId );
	shell->doDelete( copyId );
	cout << "." << flush;
}

void testObjIdToAndFromPath()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	unsigned int s1 = 10;
	unsigned int s2 = 1;
	unsigned int s3 = 23;
	unsigned int s4 = 1;
	unsigned int s5 = 9; // slower varying
	unsigned int s5a = 11; // faster varying, rightmost bracket
	vector< int > dims( 1, s1 );
	Id level1 = shell->doCreate( "IntFire", Id(), "f1", dims );
	Id origSynId( level1.value() + 1 );
	dims[0] = s2;
	Id level2 = shell->doCreate( "Neutral", level1, "f2", dims );
	dims[0] = s3;
	Id level3 = shell->doCreate( "Neutral", level2, "f3", dims );
	dims[0] = s4;
	Id level4 = shell->doCreate( "Neutral", level3, "f4", dims );
	dims[0] = s5; // slower varying
	dims.push_back( s5a ); // faster varying, rightmost bracket
	Id level5 = shell->doCreate( "Neutral", level4, "f5", dims );

	unsigned int index1 = 1;
	unsigned int index2 = 0;
	unsigned int index3 = 3;
	unsigned int index4 = 0;
	unsigned int index5 = 5; // slower varying, closer to root.
	unsigned int index5a = 6; // faster varying

	unsigned int temp = ( ( ( ( index1 * s2 + index2 ) * 
		s3 + index3 ) * 
		s4 + index4 ) *
		s5 + index5 ) *
		s5a + index5a;

	ObjId oi( level5, DataId( temp ) );
	string path = oi.path();
	assert( path == "/f1[1]/f2/f3[3]/f4/f5[5][6]" );
	assert( oi.element()->dataHandler()->totalEntries() == s1 * s2 * s3 * s4 * s5 * s5a );
	assert( oi.element()->dataHandler()->numDimensions() == 4 );
	assert( oi.element()->dataHandler()->pathDepth() == 5 );
	vector< vector< unsigned int > > pathIndices = 
		oi.element()->dataHandler()->pathIndices( oi.dataId );
	assert( pathIndices.size() == 6 );
	assert( pathIndices[0].size() == 0 );
	assert( pathIndices[1].size() == 1 );
	assert( pathIndices[1][0] == index1 );
	assert( pathIndices[2].size() == 0 );
	assert( pathIndices[3].size() == 1 );
	assert( pathIndices[3][0] == index3 );
	assert( pathIndices[4].size() == 0 );
	assert( pathIndices[5].size() == 2 );
	assert( pathIndices[5][0] == index5 );
	assert( pathIndices[5][1] == index5a );

	ObjId readPath( path );
	assert( readPath.id == level5 );
	assert( readPath.dataId.value() == temp );

	ObjId f4 = Neutral::parent( oi.eref() );
	path = f4.path();
	assert( path == "/f1[1]/f2/f3[3]/f4" );

	ObjId f3 = Neutral::parent( f4.eref() );
	path = f3.path();
	assert( path == "/f1[1]/f2/f3[3]" );

	ObjId f2 = Neutral::parent( f3.eref() );
	path = f2.path();
	assert( path == "/f1[1]/f2" );

	ObjId f1 = Neutral::parent( f2.eref() );
	path = f1.path();
	assert( path == "/f1[1]" );

	ObjId f0 = Neutral::parent( f1.eref() );
	path = f0.path();
	assert( path == "/" );
	assert( f0 == Id() );

	// Here check what happens with a move.
	dims.resize( 1 );
	dims[0] = 1;
	Id level6 = shell->doCreate( "Neutral", Id(), "foo", dims );
	Id level7 = shell->doCreate( "Neutral", level6, "bar", dims );
	Id level8 = shell->doCreate( "Neutral", level7, "zod", dims );

	shell->doMove( level1, level8 );

	ObjId noi( "/foo/bar/zod/f1[1]/f2/f3[3]/f4/f5[5][6]" );
	assert( noi.dataId.value() == temp );
	assert( noi.id == level5 );
	assert( noi == oi );
	// cout << noi.path() << endl;

	shell->doDelete( level6 );
	cout << "." << flush;
}

void testMultiLevelCopyAndPath()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	unsigned int numChan = 10;
	unsigned int numMitDend = 25;
	unsigned int numPGdend = 4;
	unsigned int numMit = 25;
	unsigned int numPG = 40;
	unsigned int numGlom = 50;

	vector< int > dims;
	Id library = shell->doCreate( "Neutral", Id(), "library", dims );
	Id chan = shell->doCreate( "HHChannel", library, "chan", dims );
	Field< double >::set( chan, "Gbar", 111 );

	Id dend = shell->doCreate( "Compartment", library, "dend", dims );
	Field< double >::set( dend, "Em", -65 );
	Id level1 = shell->doCopy( chan, dend, "chan", numChan, 0, 0 );
	vector< double > gbar( numChan );
	for ( unsigned int i = 0; i < numChan; ++i ) {
		gbar[i] = i + 1;
	}
	Field< double >::setVec( level1, "Gbar", gbar );

	Id mit = shell->doCreate( "Neutral", library, "mit", dims );
	Id level2 = shell->doCopy( dend, mit, "dends", numMitDend, 0, 0 );

	Id pg = shell->doCreate( "Neutral", library, "pg", dims );
	Id level3 = shell->doCopy( dend, pg, "dends", numPGdend, 0, 0 );

	Id glom = shell->doCreate( "Neutral", library, "glom", dims );
	Id level4 = shell->doCopy( mit, glom, "mit", numMit, 0, 0 );
	Id level5 = shell->doCopy( pg, glom, "pg", numPG, 0, 0 );

	Id bulb = shell->doCreate( "Neutral", Id(), "bulb", dims );
	Id level6 = shell->doCopy( glom, bulb, "glom", numGlom, 0, 0 );

	unsigned int glomNum = 3;
	unsigned int mitNum = 4;
	unsigned int dendNum = 5;
	unsigned int chanNum = 6;
	unsigned long long index = ( ( glomNum * numMit + mitNum ) * 
								numMitDend + dendNum ) *
								numChan + chanNum;
	
	Id temp( "/bulb/glom/mit/dends/chan" );
	assert( temp != Id() );
	ObjId oid( temp, DataId( index ) );
	ObjId oid2( "/bulb/glom[3]/mit[4]/dends[5]/chan[6]" );
	assert( oid == oid2 );

	double g = Field< double >::get( oid2, "Gbar" );
	assert( doubleEq( g, chanNum + 1 ) );

	unsigned int totSize = numChan * numMitDend * numMit * numGlom;
	assert( oid.element()->dataHandler()->totalEntries() == totSize );
	
	gbar.resize( totSize );
	for ( unsigned int i = 0 ; i < totSize; ++i ) {
		gbar[i] = i;
	}
	Field< double >::setVec( temp, "Gbar", gbar );
	for ( unsigned int i = 0 ; i < totSize; i += 1234 ) {
		ObjId oi( temp, i );
		unsigned int index = i;
		chanNum = index % numChan;
		index = index / numChan;
		dendNum = index % numMitDend;
		index = index / numMitDend;
		mitNum = index % numMit;
		index = index / numMit;
		glomNum = index % numGlom;
		stringstream ss;
		ss << "/bulb/glom[" << glomNum << "]/mit[" << mitNum << 
			"]/dends[" << dendNum << "]/chan[" << chanNum << "]";
		assert( oi.path() == ss.str() );

		double x = Field< double >::get( oi, "Gbar" );
		assert( doubleEq( x, i ) );
	}

	shell->doDelete( library );
	shell->doDelete( bulb );
	cout << "." << flush;
}

// Here we create the element independently on each node, and connect
// it up independently. Using the doAddMsg we will be able to do this
// automatically on all nodes.
void testShellParserStart()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	shell->doSetClock( 0, 5.0 );
	shell->doSetClock( 1, 2.0 );
	shell->doSetClock( 2, 2.0 );
	shell->doSetClock( 3, 1.0 );
	shell->doSetClock( 4, 3.0 );
	shell->doSetClock( 5, 5.0 );


	const Cinfo* testSchedCinfo = TestSched::initCinfo();
	vector< DimInfo > dims;
	Id tsid = Id::nextId();
	Element* tse = new Element( tsid, testSchedCinfo, "tse", dims,
		1, true );

	// testThreadSchedElement tse;
	Eref ts( tse, 0 );
	Element* ticke = Id( 2 )();
	Eref er0( ticke, DataId( 0 ) );
	Eref er1( ticke, DataId( 1 ) );
	Eref er2( ticke, DataId( 2 ) );
	Eref er3( ticke, DataId( 3 ) );
	Eref er4( ticke, DataId( 4 ) );
	Eref er5( ticke, DataId( 5 ) );

	// No idea what FuncId to use here. Assume 0.
	FuncId f( 0 );
	SingleMsg m0( Msg::nextMsgId(), er0, ts ); 
	er0.element()->addMsgAndFunc( m0.mid(), f, 0 );
	SingleMsg m1( Msg::nextMsgId(), er1, ts ); 
	er1.element()->addMsgAndFunc( m1.mid(), f, 1 );
	SingleMsg m2( Msg::nextMsgId(), er2, ts ); 
	er2.element()->addMsgAndFunc( m2.mid(), f, 2 );
	SingleMsg m3( Msg::nextMsgId(), er3, ts ); 
	er3.element()->addMsgAndFunc( m3.mid(), f, 3 );
	SingleMsg m4( Msg::nextMsgId(), er4, ts ); 
	er4.element()->addMsgAndFunc( m4.mid(), f, 4 );
	SingleMsg m5( Msg::nextMsgId(), er5, ts ); 
	er5.element()->addMsgAndFunc( m5.mid(), f, 5 );

	if ( shell->myNode() != 0 )
		return;

	shell->doStart( 10 );

	tsid.destroy();
	cout << "." << flush;
}

/**
 * Tests Shell operations carried out on multiple nodes
 */
void testInterNodeOps() // redundant.
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	Id child;
	if ( shell->myNode() == 0 ) {
		vector< int > dimensions;
		dimensions.push_back( 6139 );
		child = shell->doCreate( "Neutral", Id(), "test", dimensions );
	} 
	// cout << shell->myNode() << ": testInterNodeOps: #entries = " << child()->dataHandler()->numData() << endl;

	shell->doDelete( child );
	// child.destroy();
	cout << "." << flush;
}

void testShellSetGet()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	const unsigned int size = 100;
	vector< int > dimensions;
	dimensions.push_back( size );
	vector< double > val;

	// Set up the objects.
	Id a1 = shell->doCreate( "Arith", Id(), "a1", dimensions );

	// cout << Shell::myNode() << ": testShellSetGet: data here = (" << a1()->dataHandler()->begin() << " to " << a1()->dataHandler()->end() << ")" << endl;
	for ( unsigned int i = 0; i < size; ++i ) {
		val.push_back( i * i * i ); // use i^3 as a simple test.
		bool ret = SetGet1< double >::set( ObjId( a1, i ), "set_outputValue", i * i );
		assert( ret );
	}
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = Field< double >::get( ObjId( a1, i ), "outputValue" );
		assert( doubleEq( x, i * i ) );
	}
	bool ret = SetGet1< double >::setVec( a1, "set_outputValue", val );
	assert( ret );
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = Field< double >::get( ObjId( a1, i ), "outputValue" );
		// cout << i << "	x=" << x << "	i^3=" << i * i * i << endl;
		assert( doubleEq( x, i * i * i ) );
	}

	shell->doDelete( a1 );
	cout << "." << flush;
}

void testShellSetGetVec()
{
	/*
	unsigned int numCells = 65535;
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	vector< unsigned int > dimensions;
	dimensions.push_back( numCells );
	Id child = shell->doCreate( "IntFire", Id(), "test", dimensions );
	for ( unsigned int i = 0; i < numCells; ++i )
		numSyn.push_back( i % 32 );
	shell->doSetUintVec( child, DataId( i ), "numSynapses", numSyn );
	numSyn.clear();
	shell->doGetUintVec( child, DataId( i ), "numSynapses", numSyn );
	for ( unsigned int i = 0; i < numCells; ++i )
		assert( numSyn[i] == i % 32 );
	// Also want to assign each of the synapses to a different value.
	
	shell->doDelete( child );
	cout << "." << flush;
	*/
}

bool setupSched( Shell* shell, ObjId& tick, Id dest )
{
	MsgId ret = shell->doAddMsg( "OneToAll", 
		tick, "proc0", ObjId( dest, 0 ), "proc" );
	return ( ret != Msg::bad );
}

bool checkArg1( Id e, 
	double v0, double v1, double v2, double v3, double v4 )
{
	bool ret = 1;
	bool report = 0;
	Eref e0( e(), 0 );
	double val = reinterpret_cast< Arith* >( e0.data() )->getArg1();
	ret = ret && ( fabs( val - v0 ) < 1e-6 );
	if (report) cout << "( " << v0 << ", " << val << " ) ";

	Eref e1( e(), 1 );
	val = reinterpret_cast< Arith* >( e1.data() )->getArg1();
	ret = ret && ( fabs( val - v1 ) < 1e-6 );
	if (report) cout << "( " << v1 << ", " << val << " ) ";

	Eref e2( e(), 2 );
	val = reinterpret_cast< Arith* >( e2.data() )->getArg1();
	ret = ret && ( fabs( val - v2 ) < 1e-6 );
	if (report) cout << "( " << v2 << ", " << val << " ) ";

	Eref e3( e(), 3 );
	val = reinterpret_cast< Arith* >( e3.data() )->getArg1();
	ret = ret && ( fabs( val - v3 ) < 1e-6 );
	if (report) cout << "( " << v3 << ", " << val << " ) ";

	Eref e4( e(), 4 );
	val = reinterpret_cast< Arith* >( e4.data() )->getArg1();
	ret = ret && ( fabs( val - v4 ) < 1e-6 );
	if (report) cout << "( " << v4 << ", " << val << " )\n";

	return ret;
}

bool checkOutput( Id e, 
	double v0, double v1, double v2, double v3, double v4 )
{
	bool ret = 1;
	bool report = 0;

	vector< double > correct;
	correct.push_back( v0 );
	correct.push_back( v1 );
	correct.push_back( v2 );
	correct.push_back( v3 );
	correct.push_back( v4 );
	vector< double > retVec;
	Field< double >::getVec( e, "outputValue", retVec );
	assert( retVec.size() == 5 );

	for ( unsigned int i = 0; i < 5; ++i ) {
		ret = ret & doubleEq( retVec[i], correct[i] );
		if (report) cout << "( " << correct[i] << ", " << retVec[i] << " ) ";
	}
	if ( !ret ) {
		for ( unsigned int i = 0; i < 5; ++i ) {
			cout << "( " << correct[i] << ", " << retVec[i] << " ) ";
		}
	}
	return ret;
}

void testShellAddMsg()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	vector< int > dimensions;
	dimensions.push_back( 5 );


	///////////////////////////////////////////////////////////
	// Set up the objects.
	///////////////////////////////////////////////////////////
	Id a1 = shell->doCreate( "Arith", Id(), "a1", dimensions );
	Id a2 = shell->doCreate( "Arith", Id(), "a2", dimensions );

	Id b1 = shell->doCreate( "Arith", Id(), "b1", dimensions );
	Id b2 = shell->doCreate( "Arith", Id(), "b2", dimensions );

	Id c1 = shell->doCreate( "Arith", Id(), "c1", dimensions );
	Id c2 = shell->doCreate( "Arith", Id(), "c2", dimensions );

	Id d1 = shell->doCreate( "Arith", Id(), "d1", dimensions );
	Id d2 = shell->doCreate( "Arith", Id(), "d2", dimensions );

	Id e1 = shell->doCreate( "Arith", Id(), "e1", dimensions );
	Id e2 = shell->doCreate( "Arith", Id(), "e2", dimensions );

	Id f1 = shell->doCreate( "Arith", Id(), "f1", dimensions );
	Id f2 = shell->doCreate( "Arith", Id(), "f2", dimensions );

	Id g1 = shell->doCreate( "Arith", Id(), "g1", dimensions );
	Id g2 = shell->doCreate( "Arith", Id(), "g2", dimensions );


	///////////////////////////////////////////////////////////
	// Set up messaging
	///////////////////////////////////////////////////////////
	// Should give 04000
	MsgId m1 = shell->doAddMsg( "Single", 
		ObjId( a1, 3 ), "output", ObjId( a2, 1 ), "arg3" );
	assert( m1 != Msg::bad );

	// Should give 33333
	MsgId m2 = shell->doAddMsg( "OneToAll", 
		ObjId( b1, 2 ), "output", ObjId( b2, 0 ), "arg3" );
	assert( m2 != Msg::bad );

	// Should give 12345
	MsgId m3 = shell->doAddMsg( "OneToOne", 
		ObjId( c1, 0 ), "output", ObjId( c2, 0 ), "arg3" );
	assert( m3 != Msg::bad );

	// Should give 01234
	MsgId m4 = shell->doAddMsg( "Diagonal", 
		ObjId( d1, 0 ), "output", ObjId( d2, 0 ), "arg3" );
	assert( m4 != Msg::bad );

	// Should give 54321
	MsgId m5 = shell->doAddMsg( "Sparse", 
		ObjId( e1, 0 ), "output", ObjId( e2, 0 ), "arg3" );
	assert( m5 != Msg::bad );

	const Msg* m5p = Msg::getMsg( m5 );
	Eref m5er = m5p->manager();
	ObjId m5oid = m5er.objId();

	bool ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 0, 4, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 1, 3, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 2, 2, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 3, 1, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 4, 0, 0 );
	assert( ret );

	// Should give 15 15 15 15 15
	for ( unsigned int i = 0; i < 5; ++i ) {
		MsgId m6 = shell->doAddMsg( "OneToAll", 
			ObjId( f1, i ), "output", ObjId( f2, i ), "arg3" );
		assert( m6 != Msg::bad );
	}

	// Should give 14 13 12 11 10
	MsgId m7 = shell->doAddMsg( "Sparse", 
		ObjId( g1, 0 ), "output", ObjId( g2, 0 ), "arg3" );
	assert( m7 != Msg::bad );
	const Msg* m7p = Msg::getMsg( m7 );
	Eref m7er = m7p->manager();
	for ( unsigned int i = 0; i < 5; ++i ) {
		for ( unsigned int j = 0; j < 5; ++j ) {
			if ( i != j ) {
				ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
				m7er.objId(), "setEntry", i, j, 0 );
				assert( ret );
			}
		}
	}

	///////////////////////////////////////////////////////////
	// Set up scheduling
	///////////////////////////////////////////////////////////
	shell->doSetClock( 0, 1.0 );

	ObjId tick( Id( 2 ), 0 );
	// I want to compare the # of process msgs before and after.
	vector< Id > tgts;
	const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >(
		Tick::initCinfo()->findFinfo( "process0" ) );
	assert( sf );
	unsigned int numTgts = tick.eref().element()->getNeighbours( tgts, 
		sf );
	assert( numTgts == 0 );
	ret = setupSched( shell, tick, a1 ); assert( ret );
	ret = setupSched( shell, tick, a2 ); assert( ret );
	ret = setupSched( shell, tick, b1 ); assert( ret );
	ret = setupSched( shell, tick, b2 ); assert( ret );
	ret = setupSched( shell, tick, c1 ); assert( ret );
	ret = setupSched( shell, tick, c2 ); assert( ret );
	ret = setupSched( shell, tick, d1 ); assert( ret );
	ret = setupSched( shell, tick, d2 ); assert( ret );
	ret = setupSched( shell, tick, e1 ); assert( ret );
	ret = setupSched( shell, tick, e2 ); assert( ret );
	ret = setupSched( shell, tick, f1 ); assert( ret );
	ret = setupSched( shell, tick, f2 ); assert( ret );
	ret = setupSched( shell, tick, g1 ); assert( ret );
	ret = setupSched( shell, tick, g2 ); assert( ret );

	numTgts = tick.eref().element()->getNeighbours( tgts, sf );
	assert( numTgts == 14 );

	///////////////////////////////////////////////////////////
	// Set up initial conditions
	///////////////////////////////////////////////////////////
	
	shell->doReinit();

	vector< double > init; // 12345
	for ( unsigned int i = 1; i < 6; ++i )
		init.push_back( i );
	ret = SetGet1< double >::setVec( a1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( b1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( c1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( d1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( e1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( f1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( g1, "arg1", init ); // 12345
	assert( ret );

	double val = 0;
	val = (reinterpret_cast< const Arith* >( ObjId( a1, 0 ).data() ) )->getArg1();
	assert( doubleEq( val, 1 ) );
	val = (reinterpret_cast< const Arith* >( ObjId( a1, 1 ).data() ) )->getArg1();
	assert( doubleEq( val, 2 ) );

	val = Field< double >::get( ObjId( a1, 0 ), "arg1Value" );
	assert( doubleEq( val, 1 ) );
	val = Field< double >::get( ObjId( a1, 1 ), "arg1Value" );
	assert( doubleEq( val, 2 ) );
	val = Field< double >::get( ObjId( a1, 2 ), "arg1Value" );
	assert( doubleEq( val, 3 ) );
	val = Field< double >::get( ObjId( a1, 3 ), "arg1Value" );
	assert( doubleEq( val, 4 ) );
	val = Field< double >::get( ObjId( a1, 4 ), "arg1Value" );
	assert( doubleEq( val, 5 ) );

	vector< double > retVec( 0 );
	Field< double >::getVec( a1, "arg1Value", retVec );
	for ( unsigned int i = 0; i < 5; ++i )
		assert( doubleEq( retVec[i], i + 1 ) );

	retVec.resize( 0 );
	Field< double >::getVec( b1, "arg1Value", retVec );
	for ( unsigned int i = 0; i < 5; ++i )
		assert( doubleEq( retVec[i], i + 1 ) );

	retVec.resize( 0 );
	Field< double >::getVec( c1, "arg1Value", retVec );
	for ( unsigned int i = 0; i < 5; ++i )
		assert( doubleEq( retVec[i], i + 1 ) );


	///////////////////////////////////////////////////////////
	// Run it
	///////////////////////////////////////////////////////////

	shell->doStart( 2 );

	// Clock* clock = reinterpret_cast< Clock* >( Id(1).eref().data() );
	// clock->printCounts();
	// Qinfo::reportQ();

	///////////////////////////////////////////////////////////
	// Check output.
	///////////////////////////////////////////////////////////
	
	ret = checkOutput( a1, 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( b1, 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( c1, 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( d1, 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( e1, 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( a2, 0, 4, 0, 0, 0 );
	assert( ret );
	ret = checkOutput( b2, 3, 3, 3, 3, 3 );
	assert( ret );
	ret = checkOutput( c2, 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( d2, 0, 1, 2, 3, 4 );
	assert( ret );
	ret = checkOutput( e2, 5, 4, 3, 2, 1 );
	assert( ret );
	ret = checkOutput( f2, 15, 15, 15, 15, 15 );
	assert( ret );
	ret = checkOutput( g2, 14, 13, 12, 11, 10 );
	assert( ret );

	///////////////////////////////////////////////////////////
	// Clean up.
	///////////////////////////////////////////////////////////
	shell->doDelete( a1 );
	shell->doDelete( a2 );
	shell->doDelete( b1 );
	shell->doDelete( b2 );
	shell->doDelete( c1 );
	shell->doDelete( c2 );
	shell->doDelete( d1 );
	shell->doDelete( d2 );
	shell->doDelete( e1 );
	shell->doDelete( e2 );
	shell->doDelete( f1 );
	shell->doDelete( f2 );
	shell->doDelete( g1 );
	shell->doDelete( g2 );

	cout << "." << flush;
}

// Very similar to above, except that the tests are done on a copy.
void testCopyMsgOps()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	vector< int > dimensions;
	Id pa = shell->doCreate( "Neutral", Id(), "pa", dimensions );
	dimensions.push_back( 5 );


	///////////////////////////////////////////////////////////
	// Set up the objects.
	///////////////////////////////////////////////////////////
	Id a1 = shell->doCreate( "Arith", pa, "a1", dimensions );
	Id a2 = shell->doCreate( "Arith", pa, "a2", dimensions );

	Id b1 = shell->doCreate( "Arith", pa, "b1", dimensions );
	Id b2 = shell->doCreate( "Arith", pa, "b2", dimensions );

	Id c1 = shell->doCreate( "Arith", pa, "c1", dimensions );
	Id c2 = shell->doCreate( "Arith", pa, "c2", dimensions );

	Id d1 = shell->doCreate( "Arith", pa, "d1", dimensions );
	Id d2 = shell->doCreate( "Arith", pa, "d2", dimensions );

	Id e1 = shell->doCreate( "Arith", pa, "e1", dimensions );
	Id e2 = shell->doCreate( "Arith", pa, "e2", dimensions );

	///////////////////////////////////////////////////////////
	// Set up initial conditions and some scheduling.
	///////////////////////////////////////////////////////////
	shell->doSetClock( 0, 1.0 );
	shell->doReinit();
	bool ret = 0;
	vector< double > init; // 12345
	for ( unsigned int i = 1; i < 6; ++i )
		init.push_back( i );
	ret = SetGet1< double >::setVec( a1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( b1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( c1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( d1, "arg1", init ); // 12345
	assert( ret );
	ret = SetGet1< double >::setVec( e1, "arg1", init ); // 12345
	assert( ret );


	///////////////////////////////////////////////////////////
	// Set up messaging
	///////////////////////////////////////////////////////////
	// Should give 04000
	MsgId m1 = shell->doAddMsg( "Single", 
		ObjId( a1, 3 ), "output", ObjId( a2, 1 ), "arg1" );
	assert( m1 != Msg::bad );

	// Should give 33333
	MsgId m2 = shell->doAddMsg( "OneToAll", 
		ObjId( b1, 2 ), "output", ObjId( b2, 0 ), "arg1" );
	assert( m2 != Msg::bad );

	// Should give 12345
	MsgId m3 = shell->doAddMsg( "OneToOne", 
		ObjId( c1, 0 ), "output", ObjId( c2, 0 ), "arg1" );
	assert( m3 != Msg::bad );

	// Should give 01234
	MsgId m4 = shell->doAddMsg( "Diagonal", 
		ObjId( d1, 0 ), "output", ObjId( d2, 0 ), "arg1" );
	assert( m4 != Msg::bad );

	// Should give 54321
	MsgId m5 = shell->doAddMsg( "Sparse", 
		ObjId( e1, 0 ), "output", ObjId( e2, 0 ), "arg1" );
	assert( m5 != Msg::bad );

	const Msg* m5p = Msg::getMsg( m5 );
	Eref m5er = m5p->manager();
	ObjId m5oid = m5er.objId();

	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 0, 4, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 1, 3, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 2, 2, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 3, 1, 0 );
	assert( ret );
	ret = SetGet3< unsigned int, unsigned int, unsigned int >::set(
		m5oid, "setEntry", 4, 0, 0 );
	assert( ret );

	assert( ret );

	ObjId tick( Id( 2 ), 0 );

	///////////////////////////////////////////////////////////
	// Copy it
	///////////////////////////////////////////////////////////

	Id pa2 = shell->doCopy( pa, Id(), "pa2", 1, false, false );

	///////////////////////////////////////////////////////////
	// Pull out the child Ids.
	///////////////////////////////////////////////////////////
	vector< Id > kids = Field< vector< Id > >::get( pa2, "children");
	assert ( kids.size() == 10 );
	for ( unsigned int i = 0; i < kids.size(); ++i ) {
		ret = setupSched( shell, tick, kids[i] ); 
		assert( ret );
	}

	unsigned int j = 0;
	assert( kids[j]()->getName() == "a1" ); ++j;
	assert( kids[j]()->getName() == "a2" ); ++j;
	assert( kids[j]()->getName() == "b1" ); ++j;
	assert( kids[j]()->getName() == "b2" ); ++j;
	assert( kids[j]()->getName() == "c1" ); ++j;
	assert( kids[j]()->getName() == "c2" ); ++j;
	assert( kids[j]()->getName() == "d1" ); ++j;
	assert( kids[j]()->getName() == "d2" ); ++j;
	assert( kids[j]()->getName() == "e1" ); ++j;
	assert( kids[j]()->getName() == "e2" ); ++j;

	vector< double > retVec;
	Field< double >::getVec( kids[0], "arg1Value", retVec );
	assert( retVec.size() == 5 );
	for (unsigned int i = 0; i < 5; ++i )
		assert( doubleEq( retVec[i], init[i] ) );

	///////////////////////////////////////////////////////////
	// Run it
	///////////////////////////////////////////////////////////

	shell->doStart( 2 );

	///////////////////////////////////////////////////////////
	// Check output.
	///////////////////////////////////////////////////////////
	
	ret = checkOutput( kids[1], 0, 4, 0, 0, 0 );
	assert( ret );
	ret = checkOutput( kids[2], 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( kids[3], 3, 3, 3, 3, 3 );
	assert( ret );
	ret = checkOutput( kids[5], 1, 2, 3, 4, 5 );
	assert( ret );
	ret = checkOutput( kids[7], 0, 1, 2, 3, 4 );
	assert( ret );
	ret = checkOutput( kids[9], 5, 4, 3, 2, 1 );
	assert( ret );

	///////////////////////////////////////////////////////////
	// Clean up.
	///////////////////////////////////////////////////////////
	shell->doDelete( pa );
	shell->doDelete( pa2 );

	cout << "." << flush;
}

void testShellParserQuit()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	shell->doQuit( );
	cout << "." << flush;
}

extern bool 
	extractIndices( const string& s, vector< unsigned int >& indices );

void testExtractIndices()
{
	vector< unsigned int > ret;

	bool ok = extractIndices( "foo", ret );
	assert( ok );
	assert( ret.size() == 0 );

	ok = extractIndices( "..", ret );
	assert( ok );
	assert( ret.size() == 0 );

	ok = extractIndices( "a1[2]", ret );
	assert( ok );
	assert( ret.size() == 1 );
	assert( ret[0] == 2 );

	ok = extractIndices( "be451[0]", ret );
	assert( ok );
	assert( ret.size() == 1 );
	assert( ret[0] == 0 );

	ok = extractIndices( "be[0", ret );
	assert( !ok );
	assert( ret.size() == 0 );

	ok = extractIndices( "[0]be", ret );
	assert( !ok );
	assert( ret.size() == 0 );

	ok = extractIndices( "oops[0]]", ret );
	assert( !ok );
	assert( ret.size() == 0 );

	ok = extractIndices( "fine[0] [ 123 ]", ret );
	assert( ok );
	assert( ret.size() == 2 );
	assert( ret[0] == 0 );
	assert( ret[1] == 123 );

	cout << "." << flush;
}

void testChopString()
{
	vector< string > args;

	assert( Shell::chopString( ".", args ) == 0 );
	assert( args.size() == 1 );
	assert( args[0] == "." );

	assert( Shell::chopString( "/", args ) == 1 );
	assert( args.size() == 0 );

	assert( Shell::chopString( "..", args ) == 0 );
	assert( args.size() == 1 );
	assert( args[0] == ".." );

	assert( Shell::chopString( "./", args ) == 0 );
	assert( args.size() == 1 );
	assert( args[0] == "." );

	assert( Shell::chopString( "./foo", args ) == 0 );
	assert( args.size() == 2 );
	assert( args[0] == "." );
	assert( args[1] == "foo" );

	assert( Shell::chopString( "/foo", args ) == 1 );
	assert( args.size() == 1 );
	assert( args[0] == "foo" );

	assert( Shell::chopString( "foo", args ) == 0 );
	assert( args.size() == 1 );
	assert( args[0] == "foo" );

	assert( Shell::chopString( "foo/", args ) == 0 );
	assert( args.size() == 1 );
	assert( args[0] == "foo" );

	assert( Shell::chopString( "/foo/", args ) == 1 );
	assert( args.size() == 1 );
	assert( args[0] == "foo" );

	assert( Shell::chopString( "foo/bar/zod", args ) == 0 );
	assert( args.size() == 3 );
	assert( args[0] == "foo" );
	assert( args[1] == "bar" );
	assert( args[2] == "zod" );

	cout << "." << flush;
}

void testChopPath()
{
	vector< string > args;
	vector< vector< unsigned int > > index;

	assert( Shell::chopPath( "foo[1]/bar[2]/zod[3]", args, index, Id() ) == 0 );
	assert( args.size() == 3 );
	assert( args[0] == "foo" );
	assert( args[1] == "bar" );
	assert( args[2] == "zod" );

	assert( index.size() == 4 );
	assert( index[0].size() == 0 ); // for the root element.
	assert( index[1].size() == 1 );
	assert( index[2].size() == 1 );
	assert( index[3].size() == 1 );
	
	assert( index[1][0] == 1 );
	assert( index[2][0] == 2 );
	assert( index[3][0] == 3 );

	assert( Shell::chopPath( "/foo/bar[1]/zod[2][3]/zung[4][5][6]", 
		args, index, Id() ) == 1 );
	assert( args.size() == 4 );
	assert( args[0] == "foo" );
	assert( args[1] == "bar" );
	assert( args[2] == "zod" );
	assert( args[3] == "zung" );

	assert( index.size() == 5 );
	assert( index[0].size() == 0 );
	assert( index[1].size() == 0 );
	assert( index[2].size() == 1 );
	assert( index[3].size() == 2 );
	assert( index[4].size() == 3 );
	
	assert( index[2][0] == 1 );
	assert( index[3][0] == 2 );
	assert( index[3][1] == 3 );
	assert( index[4][0] == 4 );
	assert( index[4][1] == 5 );
	assert( index[4][2] == 6 );

	cout << "." << flush;
}

void testFindModelParent()
{
	bool findModelParent( Id cwe, const string& path,
		Id& parentId, string& modelName );

	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	///////////////////////////////////////////////////////////
	// Set up the objects.
	///////////////////////////////////////////////////////////
	vector< int > dimensions( 1, 1 );
	Id foo = shell->doCreate( "Neutral", Id(), "foo", dimensions );
	Id zod = shell->doCreate( "Neutral", Id(), "zod", dimensions );
	Id foo2 = shell->doCreate( "Neutral", zod, "foo", dimensions );

	// shell->setCwe( zod );

	string modelName;
	Id parentId;

	//////////////// do not spec model name
	bool ok = findModelParent( zod, "", parentId, modelName );
	assert( ok );
	assert( parentId == zod );
	assert( modelName == "model" );
	modelName = "";

	ok = findModelParent( zod, "/", parentId, modelName );
	assert( ok );
	assert( parentId == Id() );
	assert( modelName == "model" );
	modelName = "";

	ok = findModelParent( zod, "/foo", parentId, modelName );
	assert( ok );
	assert( parentId == foo );
	assert( modelName == "model" );
	modelName = "";

	ok = findModelParent( zod, "foo", parentId, modelName );
	assert( ok );
	assert( parentId == foo2 );
	assert( modelName == "model" );
	modelName = "";

	//////////////// spec model name too
	ok = findModelParent( zod, "bar", parentId, modelName );
	assert( ok );
	assert( parentId == zod );
	assert( modelName == "bar" );
	modelName = "";

	ok = findModelParent( zod, "/bar", parentId, modelName );
	assert( ok );
	assert( parentId == Id() );
	assert( modelName == "bar" );
	modelName = "";

	ok = findModelParent( foo, "/foo/bar", parentId, modelName );
	assert( ok );
	assert( parentId == foo );
	assert( modelName == "bar" );
	modelName = "";

	ok = findModelParent( zod, "foo/bar", parentId, modelName );
	assert( ok );
	assert( parentId == foo2 );
	assert( modelName == "bar" );
	
	shell->doDelete( foo );
	shell->doDelete( foo2 );
	shell->doDelete( zod );
	cout << "." << flush;
}

void testSyncSynapseSize()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	const Finfo* f = Cinfo::find( "IntFire" )->findFinfo( "get_numSynapses" );
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( f );
	assert( df );
	unsigned int size = 1000;
	vector< int > dims( 1, size );
	Id neuronId = shell->doCreate( "IntFire", Id(), "neurons", dims );
	assert( neuronId != Id() );
	Id synId( neuronId.value() + 1 );
	Element* syn = synId();

	// Element should exist even if data doesn't
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" ); 

	// assert( syn->dataHandler()->data( 0 ) != 0 ); // Should warn

	assert( syn->dataHandler()->totalEntries() == 65536 * size );
	assert( syn->dataHandler()->localEntries() == 0 );
	vector< unsigned int > ns( size, 0 );
	for ( unsigned int i = 0; i < size; ++i )
		ns[i] = i;
	bool ret = Field< unsigned int >::setVec( neuronId, "numSynapses", ns);
	assert( ret );
	// Here we check local entries
	/*
	DataHandler::iterator begin = syn->dataHandler()->begin();
	DataHandler::iterator end = syn->dataHandler()->end();
	assert( syn->dataHandler()->localEntries() == ( size * (size - 1) ) / 2 );
	*/

	// shell->doSyncDataHandler( neuronId, "get_numSynapses", synId );
	synId.element()->syncFieldDim();
	// shell->doSyncDataHandler( synId );

	assert( syn->dataHandler()->totalEntries() == size * (size - 1 ) );

	assert( Field< unsigned int >::get( neuronId, "linearSize" ) == size );
	assert( Field< unsigned int >::get( synId, "linearSize" ) == size * (size - 1 ) );

	vector< unsigned int > udims =
		Field< vector< unsigned int > >::get( neuronId, "objectDimensions" );
	assert( udims.size() == 1 );
	assert( udims[0] == size );
	udims = Field< vector< unsigned int > >::get( synId, "objectDimensions" );
	assert( udims.size() == 2 );
	assert( udims[0] == size );
	assert( udims[1] == size - 1 );

	// cout << "NumSyn = " << syn.totalEntries() << endl;
	shell->doDelete( neuronId );
	cout << "." << flush;
}

/**
 * Tests message inspection fields on Neutral.
 * These include 
 *  msgOut: all outgoing Msgs, reported as ObjId of their managers
 *  msgIn: all incoming Msgs, reported as ObjId of their managers
 *  msgSrc: All source Ids of messages coming into specified field
 *  msgDest: All dest Ids of messages going out of specified field
 */
void testGetMsgs()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	vector< int > dimensions;
	dimensions.push_back( 5 );


	///////////////////////////////////////////////////////////
	// Set up the objects.
	///////////////////////////////////////////////////////////
	Id a1 = shell->doCreate( "Arith", Id(), "a1", dimensions );
	Id a2 = shell->doCreate( "Arith", Id(), "a2", dimensions );

	Id b1 = shell->doCreate( "Arith", Id(), "b1", dimensions );
	Id b2 = shell->doCreate( "Arith", Id(), "b2", dimensions );

	Id c1 = shell->doCreate( "Arith", Id(), "c1", dimensions );
	Id c2 = shell->doCreate( "Arith", Id(), "c2", dimensions );

	Id d1 = shell->doCreate( "Arith", Id(), "d1", dimensions );
	Id d2 = shell->doCreate( "Arith", Id(), "d2", dimensions );

	Id e1 = shell->doCreate( "Arith", Id(), "e1", dimensions );
	Id e2 = shell->doCreate( "Arith", Id(), "e2", dimensions );

	///////////////////////////////////////////////////////////
	// Set up messaging
	///////////////////////////////////////////////////////////
	// Should give 04000
	MsgId m1 = shell->doAddMsg( "Single", 
		ObjId( a1, 3 ), "output", ObjId( a2, 1 ), "arg3" );
	assert( m1 != Msg::bad );

	// Should give 33333
	MsgId m2 = shell->doAddMsg( "OneToAll", 
		ObjId( a1, 2 ), "output", ObjId( b2, 0 ), "arg3" );
	assert( m2 != Msg::bad );

	// Should give 12345
	MsgId m3 = shell->doAddMsg( "OneToOne", 
		ObjId( a1, 0 ), "output", ObjId( c2, 0 ), "arg3" );
	assert( m3 != Msg::bad );

	// Should give 01234
	MsgId m4 = shell->doAddMsg( "Diagonal", 
		ObjId( a1, 0 ), "output", ObjId( d2, 0 ), "arg3" );
	assert( m4 != Msg::bad );

	// Should give 54321
	MsgId m5 = shell->doAddMsg( "Sparse", 
		ObjId( a1, 0 ), "output", ObjId( e2, 0 ), "arg3" );
	assert( m5 != Msg::bad );

	////////////////////////////////////////////////////////////////
	// Check that the outgoing Msgs are OK.
	////////////////////////////////////////////////////////////////

	vector< ObjId > msgMgrs = 
		Field< vector< ObjId > >::get( a1, "msgOut" );
	assert( msgMgrs.size() == 5 ); // 5 above.
	for ( unsigned int i = 0; i < 5; ++i )
		assert( Field< Id >::get( msgMgrs[i], "e1" ) == a1 );

	assert( Field< string >::get( msgMgrs[0], "className" ) == "SingleMsg" );
	assert( Field< string >::get( msgMgrs[1], "className" ) == "OneToAllMsg" );
	assert( Field< string >::get( msgMgrs[2], "className" ) == "OneToOneMsg" );
	assert( Field< string >::get( msgMgrs[3], "className" ) == "DiagonalMsg" );
	assert( Field< string >::get( msgMgrs[4], "className" ) == "SparseMsg" );

	assert( Field< Id >::get( msgMgrs[0], "e2" ) == a2 );
	assert( Field< Id >::get( msgMgrs[1], "e2" ) == b2 );
	assert( Field< Id >::get( msgMgrs[2], "e2" ) == c2 );
	assert( Field< Id >::get( msgMgrs[3], "e2" ) == d2 );
	assert( Field< Id >::get( msgMgrs[4], "e2" ) == e2 );
	cout << "." << flush;

	////////////////////////////////////////////////////////////////
	// Check that the incoming Msgs are OK.
	////////////////////////////////////////////////////////////////
	msgMgrs = Field< vector< ObjId > >::get( a1, "msgIn" );
	assert( msgMgrs.size() == 1 ); // parent msg
	assert( msgMgrs[0].id == OneToAllMsg::managerId_ );

	assert( Field< Id >::get( msgMgrs[0], "e1" ) == Id() );
	assert( Field< Id >::get( msgMgrs[0], "e2" ) == a1 );

	msgMgrs = Field< vector< ObjId > >::get( a2, "msgIn" );
	assert( msgMgrs.size() == 2 ); // parent msg + input msg
	assert( Field< Id >::get( msgMgrs[0], "e1" ) == Id() );
	assert( Field< Id >::get( msgMgrs[0], "e2" ) == a2 );
	assert( Field< Id >::get( msgMgrs[1], "e1" ) == a1 );
	assert( Field< Id >::get( msgMgrs[1], "e2" ) == a2 );

	msgMgrs = Field< vector< ObjId > >::get( b2, "msgIn" );
	assert( msgMgrs.size() == 2 ); // parent msg + input msg
	assert( Field< Id >::get( msgMgrs[0], "e1" ) == Id() );
	assert( Field< Id >::get( msgMgrs[0], "e2" ) == b2 );
	assert( Field< Id >::get( msgMgrs[1], "e1" ) == a1 );
	assert( Field< Id >::get( msgMgrs[1], "e2" ) == b2 );

	msgMgrs = Field< vector< ObjId > >::get( c2, "msgIn" );
	assert( msgMgrs.size() == 2 ); // parent msg + input msg
	assert( Field< Id >::get( msgMgrs[0], "e1" ) == Id() );
	assert( Field< Id >::get( msgMgrs[0], "e2" ) == c2 );
	assert( Field< Id >::get( msgMgrs[1], "e1" ) == a1 );
	assert( Field< Id >::get( msgMgrs[1], "e2" ) == c2 );

	msgMgrs = Field< vector< ObjId > >::get( d2, "msgIn" );
	assert( msgMgrs.size() == 2 ); // parent msg + input msg
	assert( Field< Id >::get( msgMgrs[0], "e1" ) == Id() );
	assert( Field< Id >::get( msgMgrs[0], "e2" ) == d2 );
	assert( Field< Id >::get( msgMgrs[1], "e1" ) == a1 );
	assert( Field< Id >::get( msgMgrs[1], "e2" ) == d2 );

	msgMgrs = Field< vector< ObjId > >::get( e2, "msgIn" );
	assert( msgMgrs.size() == 2 ); // parent msg + input msg
	assert( Field< Id >::get( msgMgrs[0], "e1" ) == Id() );
	assert( Field< Id >::get( msgMgrs[0], "e2" ) == e2 );
	assert( Field< Id >::get( msgMgrs[1], "e1" ) == a1 );
	assert( Field< Id >::get( msgMgrs[1], "e2" ) == e2 );
	cout << "." << flush;

	////////////////////////////////////////////////////////////////
	// Check that the MsgSrcs are OK. 
	////////////////////////////////////////////////////////////////
	vector< Id > srcIds;
	srcIds = LookupField< string, vector< Id > >::get( a2, "neighbours", "arg3" );
	assert( srcIds.size() == 1 );
	assert( srcIds[0] == a1 );
	srcIds.resize( 0 );
	srcIds = LookupField< string, vector< Id > >::get( b2, "neighbours", "arg3" );
	assert( srcIds.size() == 1 );
	assert( srcIds[0] == a1 );
	srcIds.resize( 0 );
	srcIds = LookupField< string, vector< Id > >::get( c2, "neighbours", "arg3" );
	assert( srcIds.size() == 1 );
	assert( srcIds[0] == a1 );

	MsgId m6 = shell->doAddMsg( "Single", 
		ObjId( b1, 3 ), "output", ObjId( b2, 1 ), "arg3" );
	assert( m6 != Msg::bad );
	srcIds.resize( 0 );
	srcIds = LookupField< string, vector< Id > >::get( b2, "neighbours", "arg3" );
	assert( srcIds.size() == 2 );
	assert( srcIds[0] == a1 );
	assert( srcIds[1] == b1 );
	cout << "." << flush;

	////////////////////////////////////////////////////////////////
	// Check that the MsgDests are OK. 
	////////////////////////////////////////////////////////////////
	vector< Id > destIds;
	destIds = LookupField< string, vector< Id > >::get( a1, "neighbours", "output" );
	assert( destIds.size() == 5 );
	assert( destIds[0] == a2 );
	assert( destIds[1] == b2 );
	assert( destIds[2] == c2 );
	assert( destIds[3] == d2 );
	assert( destIds[4] == e2 );
	destIds.resize( 0 );
	destIds = LookupField< string, vector< Id > >::get( b1, "neighbours", "output" );
	assert( destIds.size() == 1 );
	assert( destIds[0] == b2 );
	cout << "." << flush;

	////////////////////////////////////////////////////////////////
	// Clean up.
	////////////////////////////////////////////////////////////////


	shell->doDelete( a1 );
	shell->doDelete( a2 );
	shell->doDelete( b1 );
	shell->doDelete( b2 );
	shell->doDelete( c1 );
	shell->doDelete( c2 );
	shell->doDelete( d1 );
	shell->doDelete( d2 );
	shell->doDelete( e1 );
	shell->doDelete( e2 );
}

void testGetMsgSrcAndTarget()
{
	// cout << "." << flush;
}

void testShellMesh()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	vector< int > dimensions(1,1);

	///////////////////////////////////////////////////////////
	// Set up the objects.
	///////////////////////////////////////////////////////////
	Id compt = shell->doCreate( "CubeMesh", Id(), "compt", dimensions );
	Id mesh( compt.value() + 1 );
	Id pool = shell->doCreate( "Pool", compt, "pool", dimensions );
	shell->doAddMsg( "OneToOne",
		ObjId( pool, 0 ), "requestVolume",
		ObjId( mesh, 0 ), "get_volume" );

	vector< double > meshCoords( 9, 0.0 );
	meshCoords[3] = meshCoords[4] = meshCoords[5] = 10.0;
	meshCoords[6] = meshCoords[7] = meshCoords[8] = 1.0;

	bool ret = Field< bool >::set( compt, "preserveNumEntries", 0 );
	assert( ret );
	
	ret = Field< vector< double > >::set( compt, "coords", meshCoords );
	assert( ret );

	double testN = 123.0;
	ret = Field< double >::set( pool, "n", testN );
	assert( ret );
	
	unsigned int size = Field< unsigned int >::get( mesh, "localNumField");
	assert( size == 1000 );

	size = Field< unsigned int >::get( pool, "linearSize" );
	assert( size == 1 );
	shell->handleReMesh( mesh );
	size = Field< unsigned int >::get( pool, "linearSize" );
	assert( size == 1000 );

	vector< double > numMols;
	Field< double >::getVec( pool, "n", numMols );
	assert( numMols.size() == size );
	for ( unsigned int i = 0; i < size; ++i )
		assert( doubleEq( numMols[i] , testN ) );
	
	///////////////////////////////////////////////////////////////////
	// Here we preserve the numEntries but change sizes to 1/1000 of orig.
	///////////////////////////////////////////////////////////////////
	meshCoords.resize( 6 );
	ret = Field< bool >::set( compt, "preserveNumEntries", 1 );
	meshCoords[3] = meshCoords[4] = meshCoords[5] = 1.0;
	ret = Field< vector< double > >::set( compt, "coords", meshCoords );
	assert( ret );
	meshCoords = Field< vector< double > >::get( compt, "coords" );
	assert( doubleEq( meshCoords[6], 0.1 ) );
	assert( doubleEq( meshCoords[7], 0.1 ) );
	assert( doubleEq( meshCoords[8], 0.1 ) );

	shell->handleReMesh( mesh );
	size = Field< unsigned int >::get( pool, "linearSize" );
	assert( size == 1000 );

	// vector< double > temp( size, testN );
	// Field< double >::setVec( pool, "conc", temp );
	ret = Field< double >::setRepeat( pool, "conc", testN );
	assert( ret );
	Field< double >::getVec( pool, "n", numMols );

	// This comes from scaling testN to the 0.1x0.1x0.1 metre mesh size.
	double numShouldBe = NA * testN * 1e-3;
	for ( unsigned int i = 0; i < size; ++i )
		assert( doubleEq( numMols[i] , numShouldBe ) );

	meshCoords[3] = meshCoords[4] = meshCoords[5] = 10.0;
	meshCoords.resize( 6 ); // Make it preserveNumEntries.
	ret = Field< vector< double > >::set( compt, "coords", meshCoords );
	assert( ret );
	vector< double > conc;
	Field< double >::getVec( pool, "conc", conc );
	assert( conc.size() == size );
	// Here we make the size of the mesh 10x bigger. We've set the field
	// 'preserveNumEntries', so making the whole volume bigger increases
	// voxel size accordingly. Voxel size is now 1x1x1.
	// Conc was testN.
	// Num mols in 1e-3 m^3 was NA * testN * 1e-3. 
	// Expanding the mesh preserves numMols. So conc is now 1e-3 smaller
	for ( unsigned int i = 0; i < size; ++i )
		assert( doubleEq( conc[i], testN * 1e-3 ) );

	shell->doDelete( compt );
	cout << "." << flush;
}

void testShell( )
{
	testExtractIndices();
	testChopPath();
	testTreeTraversal();
	testChildren();
	// testShellParserQuit();
	testGetMsgs();	// Tests getting Msg info from Neutral.
	testGetMsgSrcAndTarget();
	testShellMesh();
}

extern void testWildcard();

void testMpiShell( )
{
	testShellParserCreateDelete();
	testTreeTraversal();
	testChildren();
	testDescendant();
	testMove();
	testCopy();
	testCopyFieldElement();

	testObjIdToAndFromPath();
	testMultiLevelCopyAndPath();

	testShellSetGet();
	testInterNodeOps();
	testShellAddMsg();
	testCopyMsgOps();
	testWildcard();
	testSyncSynapseSize();

	// Stuff for doLoadModel
	testFindModelParent();
}
