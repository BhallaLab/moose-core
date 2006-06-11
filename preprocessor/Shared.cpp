/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;
#include "header.h"
#include "Conn.h"
#include "Shared.h"

void Shared::parse( vector< Shared* >& sharedVec, 
	vector< Conn* >& connVec,
	vector< string >& sharedString )
{
	unsigned int j = 0;
	vector< string >::iterator i;
	vector< string > msgNames;
	string name;
	string temp;
	string type;
	Conn* c;

	for(i = sharedString.begin(); i != sharedString.end(); i++) {
		j = next_token( type, *i, 0);
		j = next_token( name, *i, j);
		j = next_token( temp, *i, j); // should be open bracket

		if ( temp != "(" ) {
			cout << "syntax error: Shared::parse: " << *i << "\n";
			continue;
		}

		if ( type == "single" ) {
			c = new Conn( "UniConn", name + "Conn" );
		} else if ( type == "multi" ) {
			c = new Conn( "MultiConn", name + "Conn" );
		} else {
			cout << "syntax error: Shared::Conn::parse: " << *i << "\n";
			continue;
		}
		connVec.push_back( c );

		j = next_token( temp, *i, j );
		msgNames.resize( 0 );
		while ( temp != ")" ) {
			if ( j == i->length() ) {
				i++;
				j = 0;
				if (i == sharedString.end())
					continue;
			}
			msgNames.push_back( temp );
			j = next_token( temp, *i, j );
			while ( temp != "," && temp != ")" )
				// It is the name of the argument. We are not interested
				j = next_token( temp, *i, j );
			if (temp == ",")
				j = next_token( temp, *i, j );
		}
		j = next_token( temp, *i, j); // should be semicolon
		if ( temp == ";" ) {
			sharedVec.push_back( new Shared( type, name, msgNames ) );
			c->setShared( msgNames, type );
		} else {
			cout << "syntax error: Shared::semicolon? " << *i << "\n";
			continue;
		}
	}
}

/*
void Shared::printConstructorWrapperH( ofstream& fout ) 
{
	fout << ",\n			" << 
		name_ << "Shared_( &" << connName_ << "_ )";
}
*/

/*
void Shared::printWrapperH( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );
	fout << "		static ";
	if ( type_ == "single" )
		fout << "SingleMsgShared" << argtypes_.size();
	else 
		fout << "NMsgShared" << argtypes_.size();
	if (argtypes_.size() == 0) {
		fout << "* get" << capsName << "Shared( Element* e ) {\n";
	} else {
		fout << "< " << argtypes_[ 0 ];
		for ( int i = 1; i < argtypes_.size(); i++ )
			fout << ", " << argtypes_[ i ];
		fout << " >* get" << capsName << "Shared( Element* e ) {\n";
	}
	fout << "			return &( static_cast< " << className <<
		"* >( e )->" << name_ << "Shared_ );\n";
	fout << "		}\n\n";
}
*/

/*
void Shared::printPrivateWrapperH(
	const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );
	fout << "		";
	if ( type_ == "single" )
		fout << "SingleMsgShared" << argtypes_.size();
	else 
		fout << "NMsgShared" << argtypes_.size();
	if (argtypes_.size() == 0) {
		fout << " " << name_ << "Shared_;\n";
	} else {
		fout << "< " << argtypes_[ 0 ];
		for ( int i = 1; i < argtypes_.size(); i++ )
			fout << ", " << argtypes_[ i ];
		fout << " > " << name_ << "Shared_;\n";
	}
}
*/

void Shared::printWrapperCpp( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );
	
	fout << "	new SharedFinfo(\n";

	fout << "		\"" << name_ << "\", &" << 
		className << "Wrapper::get" << capsName << "Conn,";
	if ( args_.size() == 0 ) {
		fout << " \"\" ),\n";
	} else {
		fout << "\n		\"";
		fout << args_[ 0 ];
		for ( unsigned int i = 1; i < args_.size(); i++ )
			fout << ", " << args_[ i ];
		fout << "\" ),\n";
	}
}

/*
void Shared::printWrapperCppFuncs(
	const string& className, ofstream& fout ) 
{
}
*/
