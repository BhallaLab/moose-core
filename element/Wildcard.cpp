/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <sstream>
#include "moose.h"
#include "Neutral.h"
#include "Wildcard.h"
// #include "../shell/Shell.h"
#define NOINDEX (UINT_MAX - 2)

// Defined in GenesisParserWrapper.cpp
extern map< string, string >& sliClassNameConvert();

static int wildcardRelativeFind( Id start, const vector< string >& path, 
		unsigned int depth, vector< Id >& ret );

static void findBraceContent( const string& path, string& beforeBrace, 
	string& insideBrace, unsigned int& index );

static bool matchName( Id parent, Id id, 
	const string& beforeBrace, const string& insideBrace, 
	unsigned int index );

static bool matchBeforeBrace( Id id, const string& name,
	bool bracesInName, unsigned int index );

static bool matchInsideBrace( Id id, const string& inside );
/**
 * wildcardFieldComparison returns true if the value of the
 * specified field matches the value in the comparsion string mid.
 * Format is FIELD(name)=val
 * If the format or the value does not match, return 0
 */
static bool wildcardFieldComparison( Id id, const string& mid )
{
	// where = could be the usual comparison operators and val
	// could be a number. No strings yet

	string::size_type pos = mid.find(')');
	if ( pos == string::npos )
		return 0;
	string fieldName = mid.substr( 0, pos );
	string::size_type pos2 = mid.find_last_of( "=<>" );
	if ( pos2 == string::npos )
		return 0;
	string op = mid.substr( pos + 1, pos2 - pos );

	string testValue = mid.substr( pos2 + 1 );

	if ( testValue.length() == 0 )
		return 0;

	const Finfo* f = id()->findFinfo( fieldName );
	if ( !f )
		return 0;

	// At this point we don't want to compare multivalue fields.
	if ( f->ftype()->nValues() != 1 )
		return 0;
	
	string actualValue;
	bool ret = f->strGet( id.eref(), actualValue );
	if ( ret == 0 )
		return 0;
	if ( op == "==" || op == "=" )
		return ( testValue == actualValue );
	if ( op == "!=" )
		return ( testValue != actualValue );
	
	double v1 = atof( actualValue.c_str() );
	double v2 = atof( testValue.c_str() );
	if ( op == ">" )
		return ( v1 > v2 );
	if ( op == ">=" )
		return ( v1 >= v2 );
	if ( op == "<" )
		return ( v1 < v2 );
	if ( op == "<=" )
		return ( v1 <= v2 );

	return 0;
}

/**
 * Does the wildcard find on a single path
 */
static int innerFind( const string& path, vector< Id >& ret)
{
	if ( path == "/" || path == "/root") {
		ret.push_back( Id() );
		return 1;
	}

	vector< string > names;
	Id start; // set to root id.
	if ( path[0] == '/' ) {
		// separateString puts in a blank first entry if the first char
		// is a separator.
		separateString( path.substr( 1 ) , names, "/" );
	} else {
		separateString( path, names, "/" );
		bool ret = get( Id::shellId().eref(), "cwe", start );
		assert( ret );
	}
	return wildcardRelativeFind( start, names, 0, ret );
}

/*
static int wildcardRelativeFind( Id start, const vector< string >& path, 
		unsigned int depth, vector< Id >& ret )
		*/

/**
 * This is the basic wildcardFind function, working on a single
 * tree. It adds entries into the vector 'ret' with Ids found according
 * to the path string. It preserves the order of the returned Ids
 * as the order of elements traversed in the search. It does NOT
 * eliminate duplicates. This is a depth-first search.
 * Note that it does the dumb but backward compatible thing with
 * Ids of arrays: it lists every entry.
 *
 * It returns the number of Ids found here.
 */
int simpleWildcardFind( const string& path, vector<Id>& ret)
{
	if ( path.length() == 0 )
		return 0;
	unsigned int n = ret.size();
	vector< string > wildcards;
	separateString( path, wildcards, "," );
	vector< string >::iterator i;
	for ( i = wildcards.begin(); i != wildcards.end(); ++i )
		innerFind( *i, ret );

	return ret.size() - n;
}

static void myUnique(vector<Id>& ret)
{
	sort(ret.begin(), ret.end());
	unsigned int i, j;
	j = 0;
	for (i = 1; i < ret.size(); i++) {
		if (ret[j] != ret[i]) {
			ret[++j] = ret[i];
		}
	}
	j++;
	if (j < ret.size())
		ret.resize(j);
}

int wildcardFind(const string& path, vector<Id>& ret) 
{
	ret.resize( 0 );
	simpleWildcardFind( path, ret );
	myUnique( ret );
	return ret.size();
}

/**
 * 	singleLevelWildcard parses a single level of the path and returns all
 * 	ids that match it. If there is a suitable doublehash, it will recurse
 * 	into child elements.
 * 	Returns # of ids found.
 */
int singleLevelWildcard( Id start, const string& path, vector< Id >& ret )
{
	if ( path.length() == 0 )
		return 0;
	unsigned int nret = ret.size();

/*
#ifdef USE_MPI
	// Won't work for global nodes
	if ( start.node() != Shell::myNode() ) {
		Eref shellE = Id::shellId().eref();
		Shell* sh = static_cast< Shell* >( shellE.data() );
		assert( shellE.e != 0 );
		vector< Nid > kids;
		unsigned int requestId = openOffNodeValueRequest< vector< Nid > >( 
			sh, &kids, 1 );
		unsigned int tgtNode = start.node();
		if ( tgtNode > Shell::myNode() )
			--tgtNode;
		sendTo3< Nid, string, unsigned int >( 
			shellE, singleLevelWildcardSlot, tgtNode,
			start, path, requestId );
		vector< Nid >* temp = 
			closeOffNodeValueRequest< vector< Nid > >( sh, requestId );
		assert( temp == &kids );
		for ( size_t i = 0; i < kids.size(); ++i ) {
			ret.push_back( kids[i] );
		}
		return ret.size() - nret;
	}
#endif
*/

	string beforeBrace;
	string insideBrace;
	unsigned int index;
	// This has to handle ghastly cases like foo[][FIELD(x)=12.3]
	findBraceContent( path, beforeBrace, insideBrace, index );
	if ( beforeBrace == "##" )
		return allChildren( start, insideBrace, index, ret ); // recursive.

	/*
	 * Future off-node stuff. But this really needs to be delegated
	 * to a separate routine, which does nothing but get the wildcard
	 * list off node. Also should bring back the names and indices.
	 *
	vector< Nid > kids;
	unsigned int requestId = openOffNodeValueRequest< vector< Nid > >( 
		sh, &kids, sh->numNodes() - 1 );
	send2< Nid, unsigned int >( shellE, requestLeSlot, start, requestId );
	vector< Nid >* temp = 
		closeOffNodeValueRequest< vector< Nid > >( sh, requestId );
	assert( temp == &kids );
		
	vector< Nid >::iterator i;
	for ( i = kids.begin(); i != kids.end(); i++ ) {
		if ( matchName( start, *i, beforeBrace, insideBrace, index ) )
			ret.push_back( *i );
	}
	return ret.size() - nret;
	*/

	vector< Id > kids; 
	Neutral::getChildren( start.eref(), kids );
// 	cout << start.eref().name() << endl;
// 	for (size_t i = 0; i < kids.size(); i++)
// 		cout << "* " << kids[i].eref().name() << endl;
	vector< Id >::iterator i;
	for ( i = kids.begin(); i != kids.end(); i++ ) {
		if ( matchName( start, *i, beforeBrace, insideBrace, index ) )
			ret.push_back( *i );
	}

	return ret.size() - nret;
}

/**
 * Parses the name and separates out the stuff before the brace, 
 * the stuff inside it, and if present, the index which is also in a 
 * brace.
 * Assume order is foo[index][insideBrace]
 */
void findBraceContent( const string& path, string& beforeBrace, 
	string& insideBrace, unsigned int& index )
{
	beforeBrace = "";
	insideBrace = "";
	index = NOINDEX;

	if ( path.length() == 0 )
		return;
	vector< string > names;
	separateString( path, names, "[" );
	if ( names.size() == 0 )
		return;
	if ( names.size() >= 1 )
		beforeBrace = names[0];
	if ( names.size() >= 2 ) {
		if ( names[1].find_first_not_of( " 	]" ) == string::npos ) {
			index = Id::AnyIndex;
		} else {
			string n1 = names[1].substr( 0, names[1].length() - 1 );
			if ( n1[0] >= '0' && n1[0] <= '9' ) { // An index
				index = atoi( n1.c_str() );
			} else { // Some wildcard conditionals
				insideBrace = n1;
			}
		}
	}
	if ( names.size() >= 3 ) {
		insideBrace = names[2].substr( 0, names[2].length() - 1 );
	}
}

/**
 * Compares the various parts of the wildcard name with the id
 * Indexing is messy here because we may refer to any of 3 things:
 * - Regular array indexing
 * - Wildcards within the braces
 * - Simple elements with index as part of their names.
 */
bool matchName( Id parent, Id id, 
	const string& beforeBrace, const string& insideBrace, 
	unsigned int index )
{
	string temp = id()->name();
	assert( temp.length() > 0 );
	bool bracesInName = 
		( temp.length() > 3 && 
		temp[temp.length() - 1] == ']' && 
		id()->elementType() == "Simple" );

	if ( !( index == Id::AnyIndex || id.index() == index || bracesInName ||
		index == NOINDEX ) )
		return 0;
	
	if (index == NOINDEX){
		if ( parent()->elementType() == "Simple" ){
			index = 0;
			if (id.index() != 0) return 0;
		}
		else if ( parent()->elementType() == "Array" )
			index = parent.index();
	}
	
	if ( matchBeforeBrace( id, beforeBrace, bracesInName, index ) ) {
		if ( insideBrace.length() == 0 ) {
			return 1;
		} else {
			return matchInsideBrace( id, insideBrace );
		}
	}
	return 0;
}

/**
 * matchInsideBrace checks for element property matches
 */
bool matchInsideBrace( Id id, const string& inside )
{
	/* Map from Genesis class names to Moose class names */
	const map< string, string >& classNameMap = sliClassNameConvert();
	
	if ( inside.substr(0, 4 ) == "TYPE" ||
		inside.substr(0, 5 ) == "CLASS" ||
		inside.substr(0, 3 ) == "ISA" )
	{
		string::size_type pos = inside.rfind( "=" );
		if ( pos == string::npos ) 
			return 0;
		bool isEquality = ( inside[ pos - 1 ] != '!' );
		string typeName = inside.substr( pos + 1 );
		if ( typeName == "membrane" )
			typeName = "Compartment";
		
		if ( inside.substr( 0, 5 ) == "CLASS" && typeName == "channel" )
			typeName = "HHChannel";
		
		bool isEqual;
		map< string, string >::const_iterator iter = classNameMap.find( typeName );
		if ( iter != classNameMap.end() )
			isEqual = ( iter->second == id()->className() );
		else
			isEqual = ( typeName == id()->className() );
		
		return ( isEqual == isEquality );
	} else if ( inside.substr( 0, 6 ) == "FIELD(" ) {
		return wildcardFieldComparison( id, inside.substr( 6 ) );
	}
	return 0;
}

/**
 * matchBeforeBrace checks to see if the wildcard string 'name' matches
 * up with the name of the id.
 * Rules:
 *      # may only be used once in the wildcard, but substitutes for any
 *      number of characters.
 *
 * 		? may be used any number of times in the wildcard, and
 * 		must substitute exactly for characters.
 *
 * 		If bracesInName, then the Id name itself includes braces.
 */
bool matchBeforeBrace( Id id, const string& name,
	bool bracesInName, unsigned int index )
{
	if ( name == "#" )
		return 1;

	string ename = id()->name();
	if ( bracesInName ) {
		string::size_type pos = ename.rfind( '[' );
		if ( pos == string::npos )
			return 0;
		if ( pos == 0 )
			return 0;
		if ( index != Id::AnyIndex ) {
			ostringstream ost( "ost" );
			ost << "[" << index << "]";
			if ( ost.str() != ename.substr( pos ) )
				return 0;
		}
		ename = ename.substr( 0, pos );
	}
	

	if ( name == ename )
		return 1;

	string::size_type pre = name.find( "#" );
	string::size_type post = name.rfind( "#" );

	// # may only be used once in the wildcard, but substitutes for any
	// number of characters.
	if ( pre != string::npos && post == pre ) {
		unsigned int epos = ename.length() - ( name.length() - post - 1 );
		return ( name.substr( 0, pre ) == ename.substr( 0, pre ) &&
			name.substr( post + 1 ) == ename.substr( epos ) );
	}

	// ? may be used any number of times in the wildcard, and
	// must substitute exactly for characters.
	if ( name.length() != ename.length() )
		return 0;
	for ( unsigned int i = 0; i < name.length(); i++ )
		if ( name[i] != '?' && name[i] != ename[i] )
			return 0;
	return 1;
}

/**
 * Recursive function to compare all descendants and cram matches into ret.
 * Returns number of matches.
 */
int allChildren( Id start, const string& insideBrace, unsigned int index,
	vector< Id >& ret )
{
	unsigned int nret = ret.size();
	vector< Id > kids;
	Neutral::getChildren( start.eref(), kids );
	vector< Id >::iterator i;
	for ( i = kids.begin(); i != kids.end(); i++ ) {
		if ( matchName( start, *i, "#", insideBrace, index ) )
			ret.push_back( *i );
		allChildren( *i, insideBrace, index, ret );
	}
	return ret.size() - nret;
}

/**
 * This is the main recursive function of the wildcarding scheme.
 * It builds a wildcard list based on path. Puts found Ids into ret,
 * and returns # found.
 * The start id is one that already matches.
 * depth is the position on the path.
 * Note that this is a single-node function: does not work for 
 * multi-node wildcard searches.
 */
int wildcardRelativeFind( Id start, const vector< string >& path, 
		unsigned int depth, vector< Id >& ret )
{
	int nret = 0;
	vector< Id > currentLevelIds;
	if ( depth == path.size() ) {
		ret.push_back( start );
		return 1;
	}

	if ( singleLevelWildcard( start, path[depth], currentLevelIds ) > 0 ) {
		vector< Id >::iterator i;
		for ( i = currentLevelIds.begin(); i != currentLevelIds.end(); ++i )
			nret += wildcardRelativeFind( *i, path, depth + 1, ret );
	}
	return nret;
}

#ifdef DO_UNIT_TESTS

// Checks if the elements in elist are identical to those that path
// should generate.
void wildcardTestFunc( 
	Element** elist, unsigned int ne, const string& path )
{
	vector< Id > ret;
	simpleWildcardFind( path, ret );
	if ( ne != ret.size() ) {
		cout << "!\nAssert	'" << path << "' : expected " <<
			ne << ", found " << ret.size() << "\n";
		assert( 0 );
	}
	for ( unsigned int i = 0; i < ne ; i++ ) {
		if ( elist[ i ] != ret[ i ]() ) {
			cout << "!\nAssert	" << path << ": item " << i << 
				": " << elist[ i ]->name() << " != " <<
					ret[ i ]()->name() << "\n";
			assert( 0 );
		}
	}
	cout << ".";
}

void testWildcard()
{
	unsigned long i;
	cout << "\nChecking wildcarding";

	
	string bb;
	string ib;
	unsigned int ii;
	findBraceContent( "foo[23][TYPE=Compartment]", bb, ib, ii );
	ASSERT( bb == "foo", "findBraceContent" );
	ASSERT( ib == "TYPE=Compartment", "findBraceContent" );
	ASSERT( ii == 23, "findBraceContent" );
	findBraceContent( "foo[][TYPE=Channel]", bb, ib, ii );
	ASSERT( bb == "foo", "findBraceContent" );
	ASSERT( ib == "TYPE=Channel", "findBraceContent" );
	ASSERT( ii == Id::AnyIndex, "findBraceContent" );
	findBraceContent( "foo[TYPE=membrane]", bb, ib, ii );
	ASSERT( bb == "foo", "findBraceContent" );
	ASSERT( ib == "TYPE=membrane", "findBraceContent" );
	ASSERT( ii == NOINDEX, "findBraceContent" );
	findBraceContent( "bar[]", bb, ib, ii );
	ASSERT( bb == "bar", "findBraceContent" );
	ASSERT( ib == "", "findBraceContent" );
	ASSERT( ii == Id::AnyIndex, "findBraceContent" );
	findBraceContent( "zod[24]", bb, ib, ii );
	ASSERT( bb == "zod", "findBraceContent" );
	ASSERT( ib == "", "findBraceContent" );
	ASSERT( ii == 24, "findBraceContent" );


	Element* a1 = Neutral::create( "Neutral", "a1", Element::root()->id(), Id::scratchId() );
	Element* c1 = Neutral::create( "Compartment", "c1", a1->id(), Id::scratchId() );
	Element* c2 = Neutral::create( "Compartment", "c2", a1->id(), Id::scratchId() );
	Element* c3 = Neutral::create( "Compartment", "c3", a1->id(), Id::scratchId() );
	Element* cIndex = Neutral::create( "Compartment", "c4[1]", a1->id(), Id::scratchId() );

	ASSERT( cIndex->elementType() == "Simple", "Wildcard test" );


	bool ret = matchBeforeBrace( a1->id(), "a1", 0, 0 );
	ASSERT( ret, "matchBeforeBrace" );
	ret = matchBeforeBrace( a1->id(), "a2", 0, 0 );
	ASSERT( ret == 0, "matchBeforeBrace" );
	ret = matchBeforeBrace( a1->id(), "a?", 0, 0 );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( a1->id(), "?1", 0, 0 );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( a1->id(), "??", 0, 0 );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( a1->id(), "#", 0, 0 );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( a1->id(), "a#", 0, 0 );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( a1->id(), "#1", 0, 0 );
	ASSERT( ret == 1, "matchBeforeBrace" );

	ret = matchBeforeBrace( cIndex->id(), "c4", 1, 1 );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( cIndex->id(), "c4", 1, Id::AnyIndex );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( cIndex->id(), "#4", 1, 1 );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( cIndex->id(), "#", 1, 1 );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( cIndex->id(), "?4", 1, 1 );
	ASSERT( ret == 1, "matchBeforeBrace" );
	ret = matchBeforeBrace( cIndex->id(), "c4", 1, 2 );
	ASSERT( ret == 0, "matchBeforeBrace" );
	ret = matchBeforeBrace( cIndex->id(), "c1", 1, 2 );
	ASSERT( ret == 0, "matchBeforeBrace" );
	ret = matchBeforeBrace( cIndex->id(), "c4", 0, 0 );
	ASSERT( ret == 0, "matchBeforeBrace" );


	ret = matchInsideBrace( a1->id(), "TYPE=Neutral" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( a1->id(), "TYPE==Neutral" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( a1->id(), "CLASS=Neutral" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( a1->id(), "ISA=Neutral" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( a1->id(), "CLASS=Neutral" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( a1->id(), "TYPE!=Channel" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( a1->id(), "CLASS!=Channel" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( a1->id(), "ISA!=Channel" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "ISA!=Neutral" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "ISA=Compartment" );
	ASSERT( ret, "matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "TYPE=membrane" );
	ASSERT( ret, "matchInsideBrace" );

	set< double >( c3, "Em", double( 123.5 ) );
	ret = matchInsideBrace( c3->id(), "FIELD(Em)=123.5" );
	ASSERT( ret, "Field matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "FIELD(Em)==123.5" );
	ASSERT( ret, "Field matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "FIELD(Em)!=123.4" );
	ASSERT( ret, "Field matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "FIELD(Em)>123.4" );
	ASSERT( ret, "Field matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "FIELD(Em)<123.6" );
	ASSERT( ret, "Field matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "FIELD(Em)>=123.4" );
	ASSERT( ret, "Field matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "FIELD(Em)<=123.6" );
	ASSERT( ret, "Field matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "FIELD(Em)>=123.5" );
	ASSERT( ret, "Field matchInsideBrace" );
	ret = matchInsideBrace( c3->id(), "FIELD(Em)<=123.5" );
	ASSERT( ret, "Field matchInsideBrace" );


	Element* el1[] = { Element::root(), a1, c1 };
	wildcardTestFunc( el1, 3, "/,/a1,/a1/c1" );
	Element* el3[] = { c1, c2, c3 };
	wildcardTestFunc( el3, 3, "a1/c#" );
	wildcardTestFunc( el3, 3, "a1/c#[TYPE=Compartment]" );

	int initialNumInstances = SimpleElement::numInstances;

	Element* el2[ 100 ];
	for ( i = 0 ; i < 100; i++ ) {
		char name[10];
		sprintf( name, "ch%ld", i );
		el2[i] = Neutral::create( "HHChannel", name, c1->id(), Id::scratchId() );
		set< double >( el2[i], "Ek", static_cast< double >( i ) );
		set< double >( el2[i], "Gbar", static_cast< double >( i * 10 ));
	}
	ASSERT( SimpleElement::numInstances - initialNumInstances == 100,
			"Check that array is made" );

	wildcardTestFunc( el2, 100, "/a1/c1/##" );
	wildcardTestFunc( el2, 100, "/a1/c1/#" );

	wildcardTestFunc( el2, 100, "/a1/##[TYPE=HHChannel]" );
	wildcardTestFunc( el2, 0, "/a1/##[TYPE=HHGate]" );

	// Here we set up some thoroughly ugly nesting.
	// Note the sequence: The wildcarding goes depth first,
	// and then in order of creation.
	el2[0] = Neutral::create( "HHGate", "g0", el2[0]->id(), Id::scratchId() );
	el2[1] = Neutral::create( "HHGate", "g1", el2[1]->id(), Id::scratchId() );
	el2[2] = Neutral::create( "HHGate", "g2", el2[1]->id(), Id::scratchId() );
	el2[3] = Neutral::create( "HHGate", "g3", el2[2]->id(), Id::scratchId() );
	el2[4] = Neutral::create( "HHGate", "g4", el2[2]->id(), Id::scratchId() );
	el2[5] = Neutral::create( "HHGate", "g5", el2[4]->id(), Id::scratchId() );
	el2[6] = Neutral::create( "HHGate", "g6", el2[5]->id(), Id::scratchId() );
	el2[7] = Neutral::create( "HHGate", "g7", el2[6]->id(), Id::scratchId() );
	el2[8] = Neutral::create( "HHGate", "g8", el2[1]->id(), Id::scratchId() );
	el2[9] = Neutral::create( "HHGate", "g9", el2[1]->id(), Id::scratchId() );
	el2[10] = Neutral::create( "HHGate", "g10", c2->id(), Id::scratchId() );
	el2[11] = Neutral::create( "HHGate", "g11", c3->id(), Id::scratchId() );
	wildcardTestFunc( el2, 12, "/a1/##[TYPE=HHGate]" );
	wildcardTestFunc( el2, 12, "/##[TYPE=HHGate]" );

	ASSERT( set( a1, "destroy" ), "Cleaning up" );
	ASSERT( SimpleElement::numInstances - initialNumInstances == -5,
			"Check that array is gone" );

	/*
	Field( "/classes/child_out" ).dest( flist );
	Element** el2 = new Element*[ flist.size() ];
	for ( i = 0; i < flist.size(); i++ )
		el2[ i ] = flist[ i ].getElement();

	wildcardTestFunc( el2, flist.size(), "/classes/##" );
	wildcardTestFunc( el2, flist.size(), "/classes/#" );
	wildcardTestFunc( el2, flist.size(), "/##[TYPE=Cinfo]" );
	wildcardTestFunc( el2, flist.size(), "/##[ISA=Cinfo]" );
	wildcardTestFunc( el2 + 2, 1, "/##[FIELD(name)=Cinfo]" );
	wildcardTestFunc( el2 + 3, 1, "/##[FIELD(name)=ClockJob]" );
	wildcardTestFunc( el2 + 4, 1, "/##[FIELD(name)==ClockTick]" );
	wildcardTestFunc( el2, flist.size() - 1, 
		"classes/##[FIELD(name)!=Tock]" );
	cout << " done\n";

	cout << "Checking wildcarding: Numerical Field tests";
	flist.resize( 0 );
	flist.push_back( Field( "/new_i1/value" ) );
	flist.push_back( Field( "/i2/value" ) );
	flist.push_back( Field( "/i3/value" ) );
	flist.push_back( Field( "/i4/value" ) );
	flist.push_back( Field( "/i7/value" ) );

	Element** el3 = new Element*[ flist.size() ];
	for ( i = 0; i < flist.size(); i++ )
		el3[ i ] = flist[ i ].getElement();

	wildcardTestFunc( el3, flist.size(), "/#[TYPE=Int]" );
	wildcardTestFunc( el3, flist.size(), "/#[TYPE==Int]" );
	wildcardTestFunc( el3, flist.size(), "/#[FIELD(value)!=3]" );
	wildcardTestFunc( el3, 1, "/#[FIELD(value)>3]" );
	wildcardTestFunc( el3, flist.size(), "/#[FIELD(value)>=0]" );
	wildcardTestFunc( el3, flist.size(), "/#[FIELD(value)<10]" );
	wildcardTestFunc( el3, flist.size(), "/#[FIELD(value)<=5]" );
	wildcardTestFunc( el3, 1, "/#[FIELD(value)!=0]" );
	wildcardTestFunc( el3 + 1, flist.size() - 1, "/#[FIELD(value)==0]");
	wildcardTestFunc( el3 + 1, flist.size() - 1, "/#[FIELD(value)<5]");

	cout << " done\n";
	cout << "Wildcarding tests complete\n\n";
	*/
}

#endif
