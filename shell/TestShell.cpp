/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include "moose.h"
#include "IdManager.h"
#include "../element/Neutral.h"
#include "../element/Wildcard.h"
#include "Shell.h"
#include "ReadCell.h"
#include "SimDump.h"
#include "Ftype3.h"
#include "../randnum/Probability.h"
#include "../randnum/Uniform.h"
#include "../randnum/Exponential.h"
#include "../randnum/Normal.h"
#include "math.h"
#include "sstream"
#include <math.h>
#include "../element/Neutral.h"

extern void testOffNodeQueue();

void testShell()
{
	cout << "\nTesting Shell";

	Element* root = Element::root();
	ASSERT( root->id().zero() , "creating /root" );

	vector< string > vs;
	separateString( "a/b/c/d/e/f/ghij/k", vs, "/" );

	/////////////////////////////////////////
	// Test path parsing
	/////////////////////////////////////////

	ASSERT( vs.size() == 8, "separate string" );
	ASSERT( vs[0] == "a", "separate string" );
	ASSERT( vs[1] == "b", "separate string" );
	ASSERT( vs[2] == "c", "separate string" );
	ASSERT( vs[3] == "d", "separate string" );
	ASSERT( vs[4] == "e", "separate string" );
	ASSERT( vs[5] == "f", "separate string" );
	ASSERT( vs[6] == "ghij", "separate string" );
	ASSERT( vs[7] == "k", "separate string" );

	separateString( "a->b->ghij->k", vs, "->" );
	ASSERT( vs.size() == 4, "separate string" );
	ASSERT( vs[0] == "a", "separate string" );
	ASSERT( vs[1] == "b", "separate string" );
	ASSERT( vs[2] == "ghij", "separate string" );
	ASSERT( vs[3] == "k", "separate string" );
	
	Shell sh;

	/////////////////////////////////////////
	// Test element creation in trees
	// This used to be a set of unit tests for Shell, but now
	// the operations have been shifted over to Neutral.
	// I still set up the creation operations because they are
	// used later for path lookup
	/////////////////////////////////////////

	Id n = Id::lastId();
	Id a = Id::scratchId();
	bool ret = 0;

	ret = sh.create( "Neutral", "a", Id(), a );
	ASSERT( ret, "creating a" );
	ASSERT( a.id() == n.id() + 1 , "creating a" );
	ASSERT( ( sh.parent( a ) == 0 ), "finding parent" );

	Id b = Id::scratchId();
	ret = sh.create( "Neutral", "b", Id(), b );
	ASSERT( ret, "creating b" );
	ASSERT( b.id() == n.id() + 2 , "creating b" );

	Id c = Id::scratchId();
	ret = sh.create( "Neutral", "c", Id(), c );
	ASSERT( ret, "creating c" );
	ASSERT( c.id() == n.id() + 3 , "creating c" );

	Id a1 = Id::scratchId();
	ret = sh.create( "Neutral", "a1", a, a1 );
	ASSERT( ret, "creating a1" );
	ASSERT( a1.id() == n.id() + 4 , "creating a1" );
	ASSERT( ( sh.parent( a1 ) == a ), "finding parent" );

	Id a2 = Id::scratchId();
	ret = sh.create( "Neutral", "a2", a, a2 );
	ASSERT( ret, "creating a2" );
	ASSERT( a2.id() == n.id() + 5 , "creating a2" );

	/////////////////////////////////////////
	// Test path lookup operations
	/////////////////////////////////////////

	string path = sh.eid2path( a1 );
	ASSERT( path == "/a/a1", "a1 eid2path" );
	path = sh.eid2path( a2 );
	ASSERT( path == "/a/a2", "a2 eid2path" );

	Id eid = sh.innerPath2eid( "/a/a1", "/", 1 );
	ASSERT( eid == a1, "a1 path2eid" );
	eid = sh.innerPath2eid( "/a/a2", "/", 1 );
	ASSERT( eid == a2, "a2 path2eid" );

	/////////////////////////////////////////
	// Test digestPath
	/////////////////////////////////////////
	/*
	Id foo = Id::scratchId(); // first make another test element.
	ret = sh.create( "Neutral", "foo", a2, foo );
	ASSERT( ret, "creating /a/a2/foo" );
	Id f = Id::scratchId(); // first make another test element.
	ret = sh.create( "Neutral", "f", a2, f );
	ASSERT( ret, "creating /a/a2/f" );
	*/

	sh.cwe_ = a2;
	sh.recentElement_ = a1;
	path = "";
	sh.digestPath( path );
	ASSERT( path == "", "path == blank" );

	path = ".";
	sh.digestPath( path );
	ASSERT( path == "/a/a2", "path == /a/a2" );

	path = "^";
	sh.digestPath( path );
	ASSERT( path == "/a/a1", "path == /a/a1" );

	path = "f";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/f", "path == /a/a2/f" );

	path = "./";
	sh.digestPath( path );
	ASSERT( path == "/a/a2", "path == /a/a2" );

	path = "..";
	sh.digestPath( path );
	ASSERT( path == "/a", "path == /a" );

	path = "ax";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/ax", "path == /a/a2/ax" );

	path = "/a";
	sh.digestPath( path );
	ASSERT( path == "/a", "path == /a" );

	sh.cwe_ = Id();
	path = "..";
	sh.digestPath( path );
	ASSERT( path == "/", "path == /" );

	path = "^/b/c/d";
	sh.digestPath( path );
	ASSERT( path == "/a/a1/b/c/d", "path == /a/a1/b/c/d" );

	sh.cwe_ = a2;
	path = "./b/c/d";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/b/c/d", "path == /a/a2/b/c/d" );

	path = "bba/bba";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/bba/bba", "path == /a/a2/bba/bba" );

	path = "../below";
	sh.digestPath( path );
	ASSERT( path == "/a/below", "path == /a/below" );

	sh.cwe_ = Id();
	path = "../rumbelow";
	sh.digestPath( path );
	ASSERT( path == "/rumbelow", "path == /rumbelow" );
	
	sh.cwe_ = a2;
	path = "x/y/z";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/x/y/z", "path == /a/a2/x/y/z" );

	path = "/absolute/x/y/z";
	sh.digestPath( path );
	ASSERT( path == "/absolute/x/y/z", "path == /absolute/x/y/z" );
	
	/////////////////////////////////////////
	// Test destroy operation
	/////////////////////////////////////////
	sh.destroy( a );
	sh.destroy( b );
	sh.destroy( c );
	ASSERT( a() == 0, "destroy a" );
	ASSERT( a1() == 0, "destroy a1" );
	ASSERT( a2() == 0, "destroy a2" );

	/////////////////////////////////////////
	// Test the loadTab operation
	/////////////////////////////////////////
	Element* tab = Neutral::create( "Table", "t1", Element::root()->id(),
		Id::scratchId() );
	static const double EPSILON = 1.0e-9;
	static double values[] = 
		{ 1, 1.0628, 1.1253, 1.1874, 1.2487, 1.309,
			1.3681, 1.4258, 1.4817, 1.5358, 1.5878 };
	sh.innerLoadTab( "/t1 table 1 10 0 10		1 1.0628 1.1253 1.1874 1.2487 1.309 1.3681 1.4258 1.4817 1.5358 1.5878" );
	int ival;
	ret = get< int >( tab, "xdivs", ival );
	ASSERT( ret, "LoadTab" );
	ASSERT( ival == 10 , "LoadTab" );
	for ( unsigned int i = 0; i < 11; i++ ) {
		double y = 0.0;
		ret = lookupGet< double, unsigned int >( tab, "table", y, i );
		ASSERT( ret, "LoadTab" );
		ASSERT( fabs( y - values[i] ) < EPSILON , "LoadTab" );
	}
	set( tab, "destroy" );

	testOffNodeQueue();
}

void testOffNodeQueue()
{
#ifdef USE_MPI
	cout << "\nTesting Shell OffNodeQueue";
	Shell testShell;
	Shell* sh = &testShell;
	// ASSERT( Id::shellId().good(), "testOffNodeQueue" );
	// Shell *sh = static_cast< Shell* >( Id::shellId().eref().data() );

	double x = 1.234;
	int y = 1234;
	string z = "1234";
	unsigned int rid0 = openOffNodeValueRequest< double >( sh, &x, 3 );
	ASSERT( rid0 == 0 , "testOffNodeQueue" );
	ASSERT( sh->numPendingOffNode( rid0 ) == 3, "testOffNodeQueue" );
	ASSERT( sh->freeRidStack_.size() == sh->maxNumOffNodeRequests - 1, "testOffNodeQueue" );
	ASSERT( sh->offNodeData_.size() == sh->maxNumOffNodeRequests, "testOffNodeQueue" );
	ASSERT( sh->offNodeData_[0].data == &x, "testOffNodeQueue" );
	ASSERT( sh->offNodeData_[0].numPending == 3, "testOffNodeQueue" );
	sh->decrementOffNodePending( rid0 );
	ASSERT( sh->numPendingOffNode( rid0 ) == 2, "testOffNodeQueue" );
	
	unsigned int rid1 = openOffNodeValueRequest< int >( sh, &y, 1 );
	ASSERT( rid1 == 1 , "testOffNodeQueue" );
	sh->decrementOffNodePending( rid1 );
	ASSERT( sh->numPendingOffNode( rid1 ) == 0, "testOffNodeQueue" );

	unsigned int rid2 = openOffNodeValueRequest< string >( sh, &z, 2 );
	ASSERT( rid2 == 2 , "testOffNodeQueue" );
	ASSERT( sh->freeRidStack_.size() == sh->maxNumOffNodeRequests - 3, "testOffNodeQueue" );

	int* testy = closeOffNodeValueRequest< int >( sh, rid1 );
	ASSERT( testy == &y, "testOffNodeQueue" );

	double p;
	unsigned int rid3 = openOffNodeValueRequest< double >( sh, &p, 1 );
	ASSERT( rid3 == 1, "testOffNodeQueue" );
	sh->decrementOffNodePending( rid3 );
	ASSERT( sh->numPendingOffNode( rid3 ) == 0, "testOffNodeQueue" );
	double* testp = closeOffNodeValueRequest< double >( sh, rid3 );
	ASSERT( testp == &p, "testOffNodeQueue" );

	sh->decrementOffNodePending( rid0 );
	sh->decrementOffNodePending( rid0 );
	ASSERT( sh->numPendingOffNode( rid0 ) == 0, "testOffNodeQueue" );
	double* testx = closeOffNodeValueRequest< double >( sh, rid0 );
	ASSERT( testx == &x, "testOffNodeQueue" );

	sh->decrementOffNodePending( rid2 );
	sh->decrementOffNodePending( rid2 );
	ASSERT( sh->numPendingOffNode( rid2 ) == 0, "testOffNodeQueue" );
	string* testz = closeOffNodeValueRequest< string >( sh, rid2 );
	ASSERT( testz == &z, "testOffNodeQueue" );

	ASSERT( sh->freeRidStack_.size() == sh->maxNumOffNodeRequests, "testOffNodeQueue" );
#endif // USE_MPI
}

#endif // DO_UNIT_TESTS
