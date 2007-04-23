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
 * Chops up a string s into pieces at separator, stuffs the pieces
 * into the vector v. Here the separator is precisely the provided
 * string.
 */
void separateString( const string& s, vector< string>& v, 
				const string& separator )
{
	string temp = s;
	unsigned int separatorLength = separator.length();
	string::size_type pos = s.find( separator );
	v.resize( 0 );

	while ( pos != string::npos ) {
		string t = temp.substr( 0, pos );
		if ( t.length() > 0 )
			v.push_back( t );
		temp = temp.substr( pos + separatorLength );
		pos = temp.find( separator );
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
