/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Neutral.h"
#include "Wildcard.h"

static bool wildcardFieldComparison( Element* e, const string& mid );

/**
 * wildcardName checks match for the name of an element.
 * The name must not contain any following slashes.
 * The name must be nonzero.
 * It returns a pointer to the found element or 0 on failure
 */
static bool wildcardName( Element* e, const string& n)
{
	size_t pos;

	if (n == "") {
		cerr << "Error in wildcardName: zero length name\n";
			return 0;
	}

	if (n == "#" || n == e->name() || n == "##") {
		return 1;
	}

	// Need to put in code to handle partial string matches

	string mid;
	if (n.substr(0, 2) == "##")
		mid = n.substr(2);
	else if (n.substr(0, 1) == "#")
		mid = n.substr(1);
	else 
		return 0;

	string head;

	// Look for type checks
	unsigned int end = 0;
	if (mid.substr(0, 7) == "[TYPE==") end = 7;
	else if (mid.substr(0, 6) == "[TYPE=") end = 6;
	else if (mid.substr(0, 5) == "[ISA=") end = 5;
	else if (mid.substr(0, 6) == "[ISA==") end = 5;
	else if (mid.substr(0, 7) == "[TYPE!=") end = 7;
	else if (mid.substr(0, 6) == "[ISA!=") end = 6;

	if ( end > 0 ) {
		pos = mid.find(']');
		if ( pos == string::npos ) {
			cerr << "wildcardName(" << n << "): Missing ']'\n";
			return 0;
		}
		head = mid.substr(end, pos - end);
		if ( mid[5] == '!' || mid[6] == '!' ) {
			if ( head == e->className() )
				return 0;
		} else {
			if ( head != e->className() )
				return 0;
		}
		mid = mid.substr( pos + 1 ); // In case there are further tests
	}
	return wildcardFieldComparison( e, mid );
}

/**
 * wildcardFieldComparison returns the Element if the value of the
 * specified field matches the value in the comparsion string mid.
 * Format is FIELD(name)=val
 * If the format does not match, return 0
 * Return the current element e if the comparison matches.
 * The actual comparison is done by Ftype.
 * Return 0 otherwise.
 */
static bool wildcardFieldComparison( Element* e, const string& mid )
{
	// where = could be the usual comparison operators and val
	// could be a number. No strings yet

	if (mid.substr(0,7) == "[FIELD(") {
		unsigned int pos = mid.find(']');
		string head = mid.substr(7, pos - 7);
		// mid = mid.substr(pos);
		pos = head.find(')');
		string fieldName = head.substr(0,pos);
		const Finfo* f = e->findFinfo( fieldName );
		if ( !f )
				return 0;

		// At this point we don't want to compare multivalue fields.
		if ( f->ftype()->nValues() != 1 )
				return 0;

		string temp = head.substr(pos + 1);
		int opLength;
		if ( temp[1] == '=' )
			opLength = 2;
		else
			opLength = 1;
		
		string op = temp.substr( 0, opLength );
		string value = temp.substr( opLength );

		/*
		 * \Todo: Figure out how to set this up cleanly.
		if ( f->ftype()->compareValue( e, op, value ) )
			return 1;
			*/

		return 0;
	}
	return 1;
}

/*
 * This was the function for the Element in the old MOOSE, which
 * did not allow children.
// Builds a wildcard list based on path, starting from an element
// in the path. Returns number found.
// We are in an Element that matches. n is nonzero.
static int wildcardRelativeFind( Element* start,
	const string& n, vector< Element* >& ret, int doublehash)
{
	if (wildcardName( start, n )) {
		ret.push_back( start );
		return 1;
	}
	return 0;
}
*/

static int innerFind( const string& n_arg, vector< Element* >& ret)
{
	string n = n_arg;
	if (n == "/") {
		ret.push_back( Element::root() );
		return 1;
	}
	if (n.rfind('/') == n.length() - 1)
		n = n.substr(0,n.length() - 1);
	
	if (n == "/root") {
		ret.push_back( Element::root() );
		return 1;
	}
	if (n.find('/') == 0)
		return wildcardRelativeFind(
			Element::root(), n.substr(1), ret, 0 );
	else
		return wildcardRelativeFind(
			Element::root(), n, ret, 0 );
}

/**
 * This is the basic wildcardFind function, working on a single
 * tree. It fills up the vector 'ret' with Element* found according
 * to the path string. It preserves the order of the returned Elements
 * as the order of elements traversed in the search. This is a 
 * depth-first search.
 * It returns the number of Elements found.
 */
int simpleWildcardFind( const string& path, vector<Element *>& ret)
{
	if ( path.length() == 0 )
		return 0;

	int n = 0;
	string temp = path;
	unsigned long pos = 0;
	while ( pos != string::npos ) {
		if ( pos > 0 )
			temp = temp.substr( pos + 1 );
		pos = temp.find(",");
		n += innerFind( temp.substr( 0, pos ), ret );
	}
	return n;
}

static void my_unique(vector<Element *>& ret)
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

int wildcardFind(const string& n, vector<Element *>& ret) 
{
	if (n == "")
		return 0;
	
	int nret = 0;
	string s = n;
	unsigned long i = 0;
	string mid;
	do {
		i = s.find(',');
		mid = s.substr(0, i);
		s = s.substr(i + 1);
		nret += simpleWildcardFind( mid, ret );
	} while (i < string::npos);
	my_unique( ret );
	return ret.size();
}


/**
 * This is the main recursive function of the wildcarding scheme.
 * It builds a wildcard list based on path. Returns number found.
 * The start element is one that already matches.
 * The string n is nonzero: there is more to come.
 * If doublehash is set then the branches as well as leaves are used.
 * Note that this is a single-node function: does not work for 
 * multi-node wildcard searches.
 * \todo: Check if still have problem that it currently returns duplicates.
 */
int wildcardRelativeFind( Element* e,
	const string& n, vector< Element* >& ret, int doublehash)
{
	unsigned long pos = n.find('/');
	string head = n.substr(0, pos);
	int nret = 0;

	if (doublehash == 0 && head.find("##") != string::npos)
		doublehash = 1;

	vector< Conn > kids;
	vector< Conn >::iterator i;
	e->findFinfo( "childSrc" )->outgoingConns( e, kids );
	for ( i = kids.begin(); i != kids.end(); i++ ) {
		Element* temp = i->targetElement();
		if ( wildcardName( temp, head ) ) {
			if (pos == string::npos) {
				ret.push_back(temp);
				nret++;
			} else {
				nret += wildcardRelativeFind( temp,
					n.substr(pos + 1, string::npos), ret, doublehash);
			}
		}
		if (doublehash) { // apply the search string recursively.
			nret += wildcardRelativeFind( temp, n, ret, doublehash );
		}
	}
	return nret;
}

#ifdef DO_UNIT_TESTS

// Checks if the elements in elist are identical to those that path
// should generate.
void wildcardTestFunc( 
	Element** elist, unsigned int ne, const string& path )
{
	vector< Element* > ret;
	simpleWildcardFind( path, ret );
	if ( ne != ret.size() ) {
		cout << "!\nAssert	'" << path << "' : expected " <<
			ne << ", found " << ret.size() << "\n";
		return;
	}
	for ( unsigned int i = 0; i < ne ; i++ ) {
		if ( elist[ i ] != ret[ i ] ) {
			cout << "!\nAssert	" << path << ": item " << i << 
				": " << elist[ i ]->name() << " != " <<
					ret[ i ]->name() << "\n";
			return;
		}
	}
	cout << ".";
}

void testWildcard()
{
	unsigned long i;
	cout << "\nChecking wildcarding";
	Element* a1 = Neutral::create( "Neutral", "a1", Element::root() );
	Element* c1 = Neutral::create( "Compartment", "c1", a1 );
	Element* c2 = Neutral::create( "Compartment", "c2", a1 );
	Element* c3 = Neutral::create( "Compartment", "c3", a1 );

	Element* el1[] = { Element::root(), a1, c1 };
	wildcardTestFunc( el1, 3, "/,/a1,/a1/c1" );

	int initialNumInstances = SimpleElement::numInstances;

	Element* el2[ 100 ];
	for ( i = 0 ; i < 100; i++ ) {
		char name[10];
		sprintf( name, "ch%ld", i );
		el2[i] = Neutral::create( "HHChannel", name, c1 );
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
	el2[0] = Neutral::create( "HHGate", "g0", el2[0] );
	el2[1] = Neutral::create( "HHGate", "g1", el2[1] );
	el2[2] = Neutral::create( "HHGate", "g2", el2[1] );
	el2[3] = Neutral::create( "HHGate", "g3", el2[2] );
	el2[4] = Neutral::create( "HHGate", "g4", el2[2] );
	el2[5] = Neutral::create( "HHGate", "g5", el2[4] );
	el2[6] = Neutral::create( "HHGate", "g6", el2[5] );
	el2[7] = Neutral::create( "HHGate", "g7", el2[6] );
	el2[8] = Neutral::create( "HHGate", "g8", el2[1] );
	el2[9] = Neutral::create( "HHGate", "g9", el2[1] );
	el2[10] = Neutral::create( "HHGate", "g10", c2 );
	el2[11] = Neutral::create( "HHGate", "g11", c3 );
	wildcardTestFunc( el2, 12, "/a1/##[TYPE=HHGate]" );
	wildcardTestFunc( el2, 12, "/##[TYPE=HHGate]" );

	ASSERT( set( a1, "destroy" ), "Cleaning up" );
	ASSERT( SimpleElement::numInstances - initialNumInstances == -4,
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
