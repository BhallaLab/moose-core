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

// Some template specializations to handle common conversions.
template<> string val2str<string>(string val) {
	return val;
}

template<> string str2val<string>(const string& s) {
	return s;
}

template<> string val2str<int>(int val) {
	char ret[40];
	sprintf(ret, "%d", val);
	return ret;
}

template<> int str2val<int>(const string& s) {
	return strtol(s.c_str(), NULL, 10);
}

template<> string val2str<double>(double val) {
	char ret[40];
	sprintf(ret, "%g", val);
	return ret;
}

template<> double str2val<double>(const string& s) {
	return strtod(s.c_str(), 0);
}

//////////////////////////////////////////////////////////////////

bool Ftype0::set( Element* e, Finfo* f )
{
	if ( dynamic_cast< const Ftype0* >( f->ftype() ) ) {
		RecvFunc func = f->recvFunc();
		if ( func ) {
			SynConn< int > c( e );
			func( &c );
			return 1;
		}
	}
	return 0;
}

//////////////////////////////////////////////////////////////////
bool MultiFtype::isSameType( const Ftype* other ) const 
{
	const MultiFtype* f = dynamic_cast< const MultiFtype* >( other );
	if ( f ) {
		if ( finfos_.size() == f->finfos_.size() ) {
			unsigned int i ;
			for ( i = 0; i < finfos_.size(); i++ ) {
				if ( !finfos_[ i ]->isSameType( f->finfos_[ i ] ) )
					return 0;
			}
			return 1;
		}
	}
	return 0;
}
