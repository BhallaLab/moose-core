/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <stdio.h>
#include "Neutral.h"
#include "Shell.h"
#include "Wildcard.h"
// #define NOINDEX (UINT_MAX - 2)

static int wildcardRelativeFind( Id start, const vector< string >& path, 
		unsigned int depth, vector< Id >& ret );

static void findBraceContent( const string& path, string& beforeBrace, 
	string& insideBrace );

static bool matchName( Id parent, Id id, 
	const string& beforeBrace, const string& insideBrace ); 

static bool matchBeforeBrace( Id id, const string& name,
	bool bracesInName );

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

	/*
	const Finfo* f = id()->findFinfo( fieldName );
	if ( !f )
		return 0;

	string actualValue;
	bool ret = f->strGet( id.eref(), actualValue );
	*/
	string actualValue;

	bool ret = SetGet::strGet( ObjId( id, 0 ), fieldName, actualValue );
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
	bool isAbsolute = Shell::chopPath( path, names );
	Id start; // set to root id.
	if ( !isAbsolute ) {
		Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
		start = s->getCwe();
	}
		
		/*
	if ( path[0] == '/' ) {
		// separateString puts in a blank first entry if the first char
		// is a separator.
		separateString( path.substr( 1 ) , names, "/" );
	} else {
		Shell* s = reinterpret_cast< Shell* >( Id.eref().data() );
		separateString( path, names, "/" );
		start = s->getCwe();
	}
	*/
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
	Shell::chopPath( path, wildcards, ',' );
	// separateString( path, wildcards, "," );
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

	string beforeBrace;
	string insideBrace;
	// This has to handle ghastly cases like foo[][FIELD(x)=12.3]
	findBraceContent( path, beforeBrace, insideBrace );
	if ( beforeBrace == "##" )
		return allChildren( start, insideBrace, ret ); // recursive.

	vector< Id > kids;
	Neutral::children( start.eref(), kids );
	vector< Id >::iterator i;
	for ( i = kids.begin(); i != kids.end(); i++ ) {
		if ( matchName( start, *i, beforeBrace, insideBrace ) )
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
	string& insideBrace )
{
	beforeBrace = "";
	insideBrace = "";

	if ( path.length() == 0 )
		return;
	vector< string > names;
	Shell::chopPath( path, names, '[' );
	if ( names.size() == 0 )
		return;
	if ( names.size() >= 1 )
		beforeBrace = names[0];
	if ( names.size() >= 2 ) {
		if ( names[1].find_first_not_of( " 	]" ) == string::npos ) {
			// look up numerical index which lives in first brace.
		} else {
			string n1 = names[1].substr( 0, names[1].length() - 1 );
			insideBrace = n1;
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
	const string& beforeBrace, const string& insideBrace )
{
	string temp = id()->getName();
	assert( temp.length() > 0 );
	bool bracesInName = 
		( temp.length() > 3 && 
		temp[temp.length() - 1] == ']' );

	if ( matchBeforeBrace( id, beforeBrace, bracesInName ) ) {
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
 * Still has some legacy hacks for reading GENESIS code.
 */
bool matchInsideBrace( Id id, const string& inside )
{
	/* Map from Genesis class names to Moose class names */
	// const map< string, string >& classNameMap = sliClassNameConvert();
	
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
		if ( inside.substr( 0, 3 ) == "ISA" ) {
			isEqual = id()->cinfo()->isA( typeName );
		} else {
			isEqual = ( typeName == id()->cinfo()->name() );
		}
		/*
		map< string, string >::const_iterator iter = classNameMap.find( typeName );
		if ( iter != classNameMap.end() )
			isEqual = ( iter->second == id()->className() );
		else
			isEqual = ( typeName == id()->className() );
			*/
		
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
bool matchBeforeBrace( Id id, const string& name, bool bracesInName )
{
	if ( name == "#" )
		return 1;

	string ename = id()->getName();
	if ( bracesInName ) {
		string::size_type pos = ename.rfind( '[' );
		if ( pos == string::npos )
			return 0;
		if ( pos == 0 )
			return 0;
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
int allChildren( Id start, const string& insideBrace, vector< Id >& ret )
{
	unsigned int nret = ret.size();
	vector< Id > kids;
	Neutral::children( start.eref(), kids );
	vector< Id >::iterator i;
	for ( i = kids.begin(); i != kids.end(); i++ ) {
		if ( matchName( start, *i, "#", insideBrace ) )
			ret.push_back( *i );
		allChildren( *i, insideBrace, ret );
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

void wildcardTestFunc( Id* elist, unsigned int ne, const string& path )
{
	vector< Id > ret;
	simpleWildcardFind( path, ret );
	if ( ne != ret.size() ) {
		cout << "!\nAssert	'" << path << "' : expected " <<
			ne << ", found " << ret.size() << "\n";
		assert( 0 );
	}
	sort( ret.begin(), ret.end() );
	for ( unsigned int i = 0; i < ne ; i++ ) {
		if ( elist[ i ] != ret[ i ] ) {
			cout << "!\nAssert	" << path << ": item " << i << 
				": " << elist[ i ]()->getName() << " != " <<
					ret[ i ]()->getName() << "\n";
			assert( 0 );
		}
	}
	cout << ".";
}

void testWildcard()
{
	unsigned long i;
	string bb;
	string ib;
	findBraceContent( "foo[23][TYPE=Compartment]", bb, ib );
	assert( bb == "foo" );
	assert( ib == "TYPE=Compartment" );
	findBraceContent( "foo[][TYPE=Channel]", bb, ib );
	assert( bb == "foo" );
	assert( ib == "TYPE=Channel" );
	findBraceContent( "foo[TYPE=membrane]", bb, ib );
	assert( bb == "foo" );
	assert( ib == "TYPE=membrane" );
	findBraceContent( "bar[]", bb, ib );
	assert( bb == "bar" );
	assert( ib == "" );
	findBraceContent( "zod[24]", bb, ib );
	assert( bb == "zod" );
	assert( ib == "24" );


	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Id a1 = shell->doCreate( "Neutral", Id(), "a1", dims );
	Id c1 = shell->doCreate( "Arith", a1, "c1", dims );
	Id c2 = shell->doCreate( "Arith", a1, "c2", dims );
	Id c3 = shell->doCreate( "Arith", a1, "c3", dims );
	Id cIndex = shell->doCreate( "Neutral", a1, "c4[1]", dims );

	bool ret = matchBeforeBrace( a1, "a1", 0 );
	assert( ret );
	ret = matchBeforeBrace( a1, "a2", 0 );
	assert( ret == 0 );
	ret = matchBeforeBrace( a1, "a?", 0 );
	assert( ret == 1 );
	ret = matchBeforeBrace( a1, "?1", 0 );
	assert( ret == 1 );
	ret = matchBeforeBrace( a1, "??", 0 );
	assert( ret == 1 );
	ret = matchBeforeBrace( a1, "#", 0 );
	assert( ret == 1 );
	ret = matchBeforeBrace( a1, "a#", 0 );
	assert( ret == 1 );
	ret = matchBeforeBrace( a1, "#1", 0 );
	assert( ret == 1 );

	ret = matchBeforeBrace( cIndex, "c4", 1 );
	assert( ret == 1 );
	ret = matchBeforeBrace( cIndex, "c4", 1 );
	assert( ret == 1 );
	ret = matchBeforeBrace( cIndex, "#4", 1 );
	assert( ret == 1 );
	ret = matchBeforeBrace( cIndex, "#", 1 );
	assert( ret == 1 );
	ret = matchBeforeBrace( cIndex, "?4", 1 );
	assert( ret == 1 );
	ret = matchBeforeBrace( cIndex, "c1", 1 );
	assert( ret == 0 );
	ret = matchBeforeBrace( cIndex, "c4", 0 );
	assert( ret == 0 );


	ret = matchInsideBrace( a1, "TYPE=Neutral" );
	assert( ret );
	ret = matchInsideBrace( a1, "TYPE==Neutral" );
	assert( ret );
	ret = matchInsideBrace( a1, "CLASS=Neutral" );
	assert( ret );
	ret = matchInsideBrace( a1, "ISA=Neutral" );
	assert( ret );
	ret = matchInsideBrace( a1, "CLASS=Neutral" );
	assert( ret );
	ret = matchInsideBrace( a1, "TYPE!=Channel" );
	assert( ret );
	ret = matchInsideBrace( a1, "CLASS!=Channel" );
	assert( ret );
	ret = matchInsideBrace( a1, "ISA!=Channel" );
	assert( ret );
	ret = matchInsideBrace( c3, "ISA==Neutral" ); // Everything is a Neutral
	assert( ret );
	ret = matchInsideBrace( c3, "ISA=Arith" );
	assert( ret );
	ret = matchInsideBrace( c3, "TYPE=membrane" );
	assert( !ret );

	Field<double>::set( ObjId( c3, 0 ), "outputValue", 123.5 );
	ret = matchInsideBrace( c3, "FIELD(outputValue)=123.5" );
	assert( ret );
	ret = matchInsideBrace( c3, "FIELD(outputValue)==123.5" );
	assert( ret );
	ret = matchInsideBrace( c3, "FIELD(outputValue)!=123.4" );
	assert( ret );
	ret = matchInsideBrace( c3, "FIELD(outputValue)>123.4" );
	assert( ret );
	ret = matchInsideBrace( c3, "FIELD(outputValue)<123.6" );
	assert( ret );
	ret = matchInsideBrace( c3, "FIELD(outputValue)>=123.4" );
	assert( ret );
	ret = matchInsideBrace( c3, "FIELD(outputValue)<=123.6" );
	assert( ret );
	ret = matchInsideBrace( c3, "FIELD(outputValue)>=123.5" );
	assert( ret );
	ret = matchInsideBrace( c3, "FIELD(outputValue)<=123.5" );
	assert( ret );


	Id el1[] = { Id(), a1, c1 };
	wildcardTestFunc( el1, 3, "/,/a1,/a1/c1" );
	Id el3[] = { c1, c2, c3, cIndex };
	wildcardTestFunc( el3, 4, "/a1/c#" );
	wildcardTestFunc( el3, 3, "/a1/c#[TYPE=Arith]" );

	Id el2[ 100 ];
	for ( i = 0 ; i < 100; i++ ) {
		char name[10];
		sprintf( name, "ch%ld", i );
		el2[i] = shell->doCreate( "Mdouble", c1, name, dims );
		//el2[i] = Neutral::create( "HHChannel", name, c1->id(), Id::scratchId() );
		Field< double >::set( ObjId( el2[i], i ), "value", i );
	}

	wildcardTestFunc( el2, 100, "/a1/c1/##" );
	wildcardTestFunc( el2, 100, "/a1/c1/#" );

	wildcardTestFunc( el2, 0, "/a1/##[TYPE=IntFire]" );
	wildcardTestFunc( el2, 100, "/a1/##[TYPE=Mdouble]" );
	wildcardTestFunc( el2, 50, "/a1/c1/##[TYPE=Mdouble][FIELD(value)<50]" );

	// Here we set up some thoroughly ugly nesting.
	// Note the sequence: The wildcarding goes depth first,
	// and then in order of creation.
	Id el4[12];
	i = 0;
	el4[i] = shell->doCreate( "IntFire", el2[0], "g0", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", el2[1], "g1", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", el2[1], "g2", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", el2[2], "g3", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", el2[2], "g4", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", el2[4], "g5", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", el2[5], "g6", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", el2[6], "g7", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", el2[1], "g8", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", el2[1], "g9", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", c2, "g10", dims ); ++i;
	el4[i] = shell->doCreate( "IntFire", c3, "g11", dims ); ++i;

	wildcardTestFunc( el4, 12, "/a1/##[TYPE=IntFire]" );
	wildcardTestFunc( el4, 12, "/##[TYPE=IntFire]" );

	//a1.destroy();
	shell->doDelete( a1 );
}

