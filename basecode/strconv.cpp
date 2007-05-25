/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MsgSrc.h"
#include "MsgDest.h"
#include "SimpleElement.h"
#include "send.h"
#include "DerivedFtype.h"

// Some template specializations to handle common conversions.
template<> bool val2str< string >( string v, string& ret)
{
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
	sprintf(temp, "%d", v);
	ret = temp;
	return 1;
}

template<> bool str2val< unsigned int >( const string& s, unsigned int& ret )
{
	ret = strtol( s.c_str(), NULL, 10 );
	return 1;
}

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
/*
void separateString( const string& s, vector< string>& v, 
				const string& separator )
{
	string temp = s;
	unsigned int separatorLength = separator.length();
	string::size_type pos;
	v.resize( 0 );
	if ( s[0] == '"' ) {
		pos = s.substr( 1 ).find( '"' );
		if ( pos == string::npos ) {
			v.push_back( s );
			return;
		}
		pos = s.substr( pos + 1 ).find( separator );
		if ( pos == string::npos ) {
			v.push_back( s );
			return;
		}
	}
	pos = s.find( separator );

	while ( pos != string::npos ) {
		string t = temp.substr( 0, pos );
		v.push_back( t );
		temp = temp.substr( pos + separatorLength );
		pos = temp.find( separator );
	}
	if ( temp.length() > 0 )
		v.push_back( temp );
}
*/


/*
void separateString( const string& s, vector< string>& v, 
				const string& separator )
{
	string temp = s;
	unsigned int separatorLength = separator.length();
	string::size_type pos = s.find( separator );
	v.resize( 0 );

	while ( pos != string::npos ) {
		string t = temp.substr( 0, pos );
		v.push_back( t );
		temp = temp.substr( pos + separatorLength );
		pos = temp.find( separator );
	}
	if ( temp.length() > 0 )
		v.push_back( temp );
}
*/

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
		temp = temp.substr( pos );
		pos = temp.find_first_of( separators );
	}
	if ( temp.length() > 0 )
		v.push_back( temp );
}

template<> bool str2val< vector< string > >( 
				const string& s, vector< string >& ret )
{
	// cerr << "in str2val< double >\n";
	
	separateString( s, ret, "," );
	return 1;
}
