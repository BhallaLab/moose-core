/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _STRCONV_H
#define _STRCONV_H

/**
 * This function has to be specialized for each Ftype
 * that we wish to be able to convert. Otherwise it
 * reports failure.
 * These functions are defined in strconv.cpp
 */
template< class T >bool val2str( T v, string& s );
template<> bool val2str< string >( string v, string& ret);
template<> bool val2str< int >( int v, string& ret);
template<> bool val2str< unsigned int >( unsigned int v, string& ret);
template<> bool val2str< double >( double v, string& ret);
template<> bool val2str< Id >( Id v, string& ret);
template<> bool val2str< bool >( bool v, string& ret);
template<> bool val2str< vector< string > >(
	vector< string > v, string& ret);
template<> bool val2str< const Ftype* >(
	const Ftype* f, string& ret );

template<> bool val2str< ProcInfo >( ProcInfo v, string& ret);

template<> bool val2str< vector< double > >(
				vector< double > v, string& ret);

template< class T >bool val2str( T v, string& s ) {
	s = "";
	return 0;
}

/**
 * This function has to be specialized for each Ftype
 * that we wish to be able to convert. Otherwise it
 * reports failure.
 */
template< class T > bool str2val( const string& s, T& v );
template<> bool str2val< string >( const string& v, string& ret);
template<> bool str2val< int >( const string& s, int& ret );
template<> bool str2val< unsigned int >( 
	const string& s, unsigned int& ret );
template<> bool str2val< double >( const string& s, double& ret );
template<> bool str2val< Id >( const string& s, Id& ret );
template<> bool str2val< bool >( const string& s, bool& ret );
template<> bool str2val< vector< string > >(
	const string& s, vector< string >& ret );
template<> bool str2val< const Ftype* >( 
	const string& s, const Ftype* &ret );
template<> bool str2val< ProcInfo >( const string& s, ProcInfo& ret );
template<> bool str2val< vector<int> >(const string& s, vector<int> & ret);
template<> bool str2val< vector<unsigned int> >(const string& s, vector<unsigned int> & ret);
template< class T > bool str2val( const string& s, T& v ) {
	cerr << "This is the default str2val.\n";
	return 0;
}


#endif // _STRCONV_H
