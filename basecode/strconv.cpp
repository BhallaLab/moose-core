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
