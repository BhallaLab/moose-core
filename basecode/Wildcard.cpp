/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"

// Checks match for the name of an element.
// The name does not contain any following slashes.
// The name is nonzero.
Element* Element::wildcardName(const string& n) {
	size_t pos;

	if (n == "") {
		cerr << "Error in wildcardName: zero length name\n";
			return 0;
	}

	if (n == "#" || n == name() || n == "##") {
		return this;
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
			cerr << "Element::wildcardName(" << n << "): Missing ']'\n";
			return 0;
		}
		head = mid.substr(end, pos - end);
		if ( mid[5] == '!' || mid[6] == '!' ) {
			if (head == cinfo()->name())
				return 0;
		} else {
			if (head != cinfo()->name())
				return 0;
		}
		mid = mid.substr( pos + 1 ); // In case there are further tests
	}

	/*
	if (mid.substr(0, 6) == "[TYPE=") {
		pos = mid.find(']');
		head = mid.substr(6, pos - 6);
		mid = mid.substr(pos + 1);
		if (head != cinfo()->name())
			return 0;
	}
	if (mid.substr(0,5) == "[ISA=") {
		pos = mid.find(']');
		head = mid.substr(5, pos - 5);
		mid = mid.substr(pos + 1);
		if (head != cinfo()->name())
			return 0;
	}
	if (mid.substr(0, 7) == "[TYPE!=") {
		pos = mid.find(']');
		head = mid.substr(7, pos - 7);
		mid = mid.substr(pos + 1);
		if (head == cinfo()->name())
			return 0;
	}
	if (mid.substr(0,6) == "[ISA!=") {
		pos = mid.find(']');
		head = mid.substr(6, pos - 6);
		mid = mid.substr(pos + 1);
		if (head == cinfo()->name())
			return 0;
	}
	*/
	return wildcardFieldComparison(mid);
}

// Look for specific fields, in format FIELD(name)=val
// Ithe format does not match, return the current element.
// Return the current element if the comparison matches.
// The actual comparison is done by Ftype.
// Return 0 otherwise.
Element* Element::wildcardFieldComparison(const string& mid) {
	// where = could be the usual comparison operators and val
	// could be a number. No strings yet

	if (mid.substr(0,7) == "[FIELD(") {
		unsigned int pos = mid.find(']');
		string head = mid.substr(7, pos - 7);
		// mid = mid.substr(pos);
		pos = head.find(')');
		string fieldName = head.substr(0,pos);

		Field f = field( fieldName );
		if ( !f.good() )
			return 0;

		string temp = head.substr(pos + 1);
		int opLength;
		if ( temp[1] == '=' )
			opLength = 2;
		else
			opLength = 1;
		
		string op = temp.substr( 0, opLength );
		string value = temp.substr( opLength );
		if ( f.valueComparison( op, value ) )
			return this;
		return 0;
	}
	return this;
}

// Builds a wildcard list based on path. Returns number found.
// We are in an Element that matches. n is nonzero.
int Element::wildcardRelativeFind(
	const string& n, vector<Element *>& ret, int doublehash)
{
	if (wildcardName(n)) {
		ret.push_back(this);
		return 1;
	}
	return 0;
}

int innerFind( const string& n_arg, vector<Element *>& ret) {
	string n = n_arg;
	if (n == "/") {
		ret.push_back(Element::root());
		return 1;
	}
	if (n.rfind('/') == n.length() - 1)
		n = n.substr(0,n.length() - 1);
	
	if (n == "/root") {
		ret.push_back(Element::root());
		return 1;
	}
	if (n.find('/') == 0)
		return Element::root()->wildcardRelativeFind(n.substr(1), ret, 0);
	else
		return Element::root()->wildcardRelativeFind(n, ret, 0);
}

int Element::startFind(const string& path, vector<Element *>& ret) {
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

void my_unique(vector<Element *>& ret)
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

int Element::wildcardFind(const string& n, vector<Element *>& ret) 
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
		nret += startFind(mid, ret);
	} while (i < string::npos);
	my_unique(ret);
	// sort(ret.begin(), ret.end());
	// upi_unique
	// unique(ret.begin(), ret.end());
	// ret.erase(unique(ret.begin(), ret.end()));
	return ret.size();
}

// Builds a wildcard list based on path. Returns number found.
// We are in an element that matches.
// n is nonzero: there is more to come.
// If doublehash is set then the branches as well as leaves are used.
// Check if still have problem that it currently returns duplicates.
int Neutral::wildcardRelativeFind(
	const string& n, vector< Element* >& ret, int doublehash)
{
	unsigned long pos = n.find('/');
	string head = n.substr(0, pos);
	Element* e;
	int nret = 0;

	if (doublehash == 0 && head.find("##") != string::npos)
		doublehash = 1;

	vector< Conn* > kids;
	childOut_.listTargets( kids );
	for (unsigned int i = 0; i < kids.size(); i++) {
		e = kids[i]->parent();
		if (e->wildcardName(head)) {
			if (pos == string::npos) {
				ret.push_back(e);
				nret++;
			} else {
				nret += e->wildcardRelativeFind(
					n.substr(pos + 1, string::npos), ret, doublehash);
			}
		}
		if (doublehash) {
			nret += e->wildcardRelativeFind(n, ret, doublehash);
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
	Element::startFind( path, ret );
	if ( ne != ret.size() ) {
		cout << "!\nAssert	'" << path << "' : expected " <<
			ne << ", found " << ret.size() << "\n";
		return;
	}
	for ( unsigned int i = 0; i < ne ; i++ ) {
		if ( elist[ i ] != ret[ i ] ) {
			cout << "!\nAssert	" << path << ": item " << i << 
				": " << elist[ i ]->path() << " != " <<
					ret[ i ]->path() << "\n";
			return;
		}
	}
	cout << ".";
}

void testWildcard()
{
	unsigned long i;
	cout << "Checking wildcarding: Enumerated list";
	Element* el1[] = { Element::root(), Element::classes() };
	wildcardTestFunc( el1, 2, "/,/classes" );
	vector< Field > flist;
	cout << " done\n";

	cout << "Checking wildcarding: Trees, Types and Field equalities";
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
}

#endif
