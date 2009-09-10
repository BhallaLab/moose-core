/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <sstream>
#include "header.h"
#include "SimpleElement.h"
#include "Send.h"
#include "SetConn.h"
#include "ProcInfo.h"
#include "DerivedFtype.h"


void separateString( const string& s, vector< string>& v, 
	const string& separator );

// Some template specializations to handle common conversions.
template<> bool val2str< string >( string v, string& ret)
{
	if ( v.empty() )
		ret = "<blank-string>";
	else
		ret = v;
	return 1;
}

template<> bool str2val< string >( const string& v, string& ret)
{
	ret = v;
	return 1;
}

template<> bool val2str< int >( int v, string& ret)
{
	char temp[40];
	sprintf(temp, "%d", v);
	ret = temp;
	return 1;
}

template<> bool str2val< int >( const string& s, int& ret )
{
	ret = strtol( s.c_str(), NULL, 10 );
	return 1;
}

template<> bool val2str< unsigned int >( unsigned int v, string& ret)
{
	char temp[40];
	sprintf(temp, "%u", v);
	ret = temp;
	return 1;
}

template<> bool str2val< unsigned int >( const string& s, unsigned int& ret )
{
	ret = strtol( s.c_str(), NULL, 10 );
	return 1;
}

template<> bool val2str< double >( double v, string& ret)
{
	char temp[40];
	// cerr << "in val2str< double >\n";
	sprintf(temp, "%g", v);
	ret = temp;
	return 1;
}

template<> bool str2val< double >( const string& s, double& ret )
{
	// cerr << "in str2val< double >\n";
	ret = strtod( s.c_str(), 0 );
	return 1;
}

// This one was initiated for 2D lookup tables.
// The finfo is named table[xindex][yindex] and when passed to
// str2val, index string looks like xindex][yindex
// so - split on the ][ - hopefully this should be able to handle
// more than 2D indexing
template<> bool str2val< vector<int> >( const string& s, vector<int>& ret )
{
    // cerr << "in str2val< vector<int> >\n";
    vector<string> dims;
    separateString(s, dims, "][");
    for (vector<string>::iterator ii = dims.begin(); ii != dims.end(); ++ii){
        int index = strtol((*ii).c_str(), NULL, 0);
        ret.push_back(index);
    }
    return 1;
}
template<> bool str2val< vector<unsigned int> >( const string& s, vector<unsigned int>& ret )
{
    // cerr << "in str2val< vector<int> >\n";
    vector<string> dims;
    separateString(s, dims, "][");
    for (vector<string>::iterator ii = dims.begin(); ii != dims.end(); ++ii){
        unsigned int index = strtol((*ii).c_str(), NULL, 0);
        ret.push_back(index);
    }
    return 1;
}

/////////////////////////////////////////////////////////////////////////
// Id conversions
/////////////////////////////////////////////////////////////////////////

template<> bool val2str< Id >( Id v, string& ret)
{
	ret = Id::id2str( v );
	return 1;
}

template<> bool str2val< Id >( const string& s, Id& ret )
{
	ret = Id::str2Id( s );
	return 1;
}

/////////////////////////////////////////////////////////////////////////
// String conversions
/////////////////////////////////////////////////////////////////////////

template<> bool val2str< bool >( bool v, string& ret)
{
	char temp[40];
	sprintf(temp, "%d", v);
	ret = temp;
	return 1;
}

template<> bool str2val< bool >( const string& s, bool& ret )
{
	ret = strtol( s.c_str(), NULL, 10 );
	if ( s.length() == 0 ) {
		ret = 0;
	} else if ( s == "0" || s == "false" || s == "False" ) {
		ret = 0;
	} else
		ret = 1;

	return 1;
}

template<> bool val2str< vector< string > >(
				vector< string > v, string& ret)
{
	ret = "";
	unsigned int i;
	for ( i = 0 ; i < v.size(); i++ ) {
		if ( i + 1 < v.size() )
			ret = ret + v[i] + ", ";
		else
			ret = ret + v[i];
	}
	return 1;
}

template<> bool str2val< vector< string > >( 
				const string& s, vector< string >& ret )
{
	// cerr << "in str2val< double >\n";
	
	separateString( s, ret, "," );
	return 1;
}

template<> bool val2str< vector< double > >(
				vector< double > v, string& ret)
{
	stringstream s;
	s << "Size: " << v.size() << " [";
	if ( v.size() < 20 ) {
		if ( v.size() > 0 )
			s << v[0];
		for ( unsigned int i = 1 ; i < v.size(); i++ )
			s << ", " << v[i];
	} else {
		for ( unsigned int i = 0 ; i < 10; i++ )
			s << v[i] << ", ";
		s << "... ";
		for ( unsigned int i = v.size() - 10 ; i < v.size(); i++ )
			s << ", " << v[i];
	}
	s << "]";

	ret = s.str();
	return 1;
}

/////////////////////////////////////////////////////////////////////////
// Ftype conversions
/////////////////////////////////////////////////////////////////////////

template<> bool str2val< const Ftype* >( const string& s, const Ftype* &ret)
{
	ret = 0;
	return 0;
}

template<> bool val2str< const Ftype* >(
				const Ftype* f, string& ret )
{
	if ( f ) {
		ret = f->typeStr();
		return 1;
	}
	ret = "";
	return 0;
}

/////////////////////////////////////////////////////////////////////////
// ProcInfo conversions
/////////////////////////////////////////////////////////////////////////
template<> bool str2val< ProcInfo >( const string& s, ProcInfo &ret)
{
	static ProcInfoBase pb; 
	// thread-unsafe hack, but it should never be used anyway.
	vector< string > svec;
	separateString( s, svec, " " );
	if ( svec.size() == 2 ) {
		pb.dt_ = atof( svec[0].c_str() );
		pb.currTime_ = atof( svec[1].c_str() );
	} else {
		pb.dt_ = 1.0;
		pb.currTime_ = 0.0;
	}

	ret = &pb;
	return 1;
}

template<> bool val2str< ProcInfo >( ProcInfo v, string& ret)
{
	if ( v != 0 ) {
		char line[40];
		sprintf( line, "%g %g", v->dt_, v->currTime_ );
		ret = line;
	} else {
		ret = "0 0";
	}
	return 1;
}

/////////////////////////////////////////////////////////////////////////
// Utility functions
/////////////////////////////////////////////////////////////////////////

/**
 * Locates the next separator on the string, analogous to the string::find
 * function. The difference is that here it also checks if the string
 * is protected with quotes anywhere. If so, it skips the quoted portion
 * and looks for the separator after that.
 */
string::size_type nextSeparator( const string& s, const string& separator )
{
	string::size_type spos = s.find( separator );
	if ( spos == string::npos )
		return spos;

	string::size_type qpos = s.find( '"' );
	if ( qpos == string::npos )
		return spos;

	if ( qpos > spos )
		return spos;
	
	string::size_type nextqpos = s.substr( qpos + 1 ).find( '"' );
	if ( nextqpos == string::npos ) {
		// Ugh. Unterminated quote
		cout << "Error: separateString: Unterminated quote in '" <<
			s << "'\n";
		return string::npos;
	}

	nextqpos += qpos;
	spos = s.substr( nextqpos ).find( separator );
	if ( spos == string::npos )
		return spos;
	return spos + nextqpos;
}

/**
 * Chops up a string s into pieces at separator, stuffs the pieces
 * into the vector v. Here the separator is precisely the provided
 * string.
 * Consecutive separators are assumed to be around a blank string.
 * If a quote " is found, the routine ignores separators till the
 * matching end of the quote.
 */

void separateString( const string& s, vector< string>& v, 
				const string& separator )
{
	string temp = s;
	unsigned int separatorLength = separator.length();
	string::size_type pos = nextSeparator( s, separator );
	v.resize( 0 );

	while ( pos != string::npos ) {
		string t = temp.substr( 0, pos );
		v.push_back( t );
		temp = temp.substr( pos + separatorLength );
		pos = nextSeparator( temp, separator );
	}
	if ( temp.length() > 0 )
		v.push_back( temp );
}


/**
 * Chops up a string s into pieces at separator, stuffs the pieces
 * into the vector v. Here the separator can be any one or more
 * of the values in the char*.
 */
void parseString( const string& s, vector< string>& v, 
				const char* separators )
{
	string temp = s;
	string::size_type pos = s.find_first_of( separators );
	v.resize( 0 );

	while ( pos != string::npos ) {
		string t = temp.substr( 0, pos );
		if ( t.length() > 0 )
			v.push_back( t );
		temp = temp.substr( pos );
		pos = temp.find_first_not_of( separators );
		if ( pos == string::npos) {
			temp = "";
			break;
		}
		temp = temp.substr( pos );
		pos = temp.find_first_of( separators );
	}
	if ( temp.length() > 0 )
		v.push_back( temp );
}
