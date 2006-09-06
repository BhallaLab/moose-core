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
#include "Src.h"

void Src::parse( vector< Src* >& srcVec, 
	vector< Conn* >& connVec,
	vector< string >& srcString )
{
	unsigned int j = 0;
	vector< string >::iterator i;
	vector< string > argtypes;
	string name;
	string temp;
	string type;
	string connName;
	bool sharesConn = 0;

	for(i = srcString.begin(); i != srcString.end(); i++) {
		j = next_token( type, *i, 0);
		j = next_token( name, *i, j);
		j = next_token( temp, *i, j); // should be open bracket

		connName = Conn::parse( connVec, name, type, "Out", sharesConn);
		if ( connName == "") {
			cout << "syntax error: Src::Conn::parse: " << *i << "\n";
			continue;
		}

		if ( temp != "(" ) {
			cout << "syntax error: Src::parse: " << *i << "\n";
			continue;
		}
		j = next_token( temp, *i, j );
		argtypes.resize( 0 );
		while ( temp != ")" ) {
			if ( j == i->length() ) {
				i++;
				j = 0;
				if (i == srcString.end())
					continue;
			}
			temp = checkForVector( temp, *i, j );
			argtypes.push_back( temp );
			j = next_token( temp, *i, j );
			while ( temp != "," && temp != ")" )
				// It is the name of the argument. We are not interested
				j = next_token( temp, *i, j );
			if (temp == ",")
				j = next_token( temp, *i, j );
		}
		j = next_token( temp, *i, j); // should be semicolon
		if ( temp == ";" )
			srcVec.push_back( 
				new Src( type, name, connName, argtypes, sharesConn ) );
	}
}

void Src::printConstructorWrapperH( ofstream& fout ) 
{
	fout << ",\n			" << 
		name_ << "Src_( &" << connName_ << "_ )";
}

void Src::printWrapperH( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );
	fout << "		static ";
	if ( type_ == "single" )
		fout << "SingleMsgSrc";
	else // This includes both multi and solve.
		fout << "NMsgSrc";
	fout << "* get" << capsName << "Src( Element* e ) {\n";

	/*
	if ( type_ == "single" )
		fout << "SingleMsgSrc" << argtypes_.size();
	else 
		fout << "NMsgSrc" << argtypes_.size();
	if (argtypes_.size() == 0) {
		fout << "* get" << capsName << "Src( Element* e ) {\n";
	} else {
		fout << "< " << argtypes_[ 0 ];
		for ( int i = 1; i < argtypes_.size(); i++ )
			fout << ", " << argtypes_[ i ];
		fout << " >* get" << capsName << "Src( Element* e ) {\n";
	}
	*/
	fout << "			return &( static_cast< " << className <<
		"Wrapper* >( e )->" << name_ << "Src_ );\n";
	fout << "		}\n\n";
}

void Src::printPrivateWrapperH(
	const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );
	fout << "		";
	if ( type_ == "single" )
		fout << "SingleMsgSrc" << argtypes_.size();
	else 
		fout << "NMsgSrc" << argtypes_.size();
	if (argtypes_.size() == 0) {
		fout << " " << name_ << "Src_;\n";
	} else {
		fout << "< " << argtypes_[ 0 ];
		for ( unsigned int i = 1; i < argtypes_.size(); i++ )
			fout << ", " << argtypes_[ i ];
		fout << " > " << name_ << "Src_;\n";
	}
}

void Src::printWrapperCpp( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );
	
	if ( type_ == "single" )
		fout << "	new SingleSrc" << argtypes_.size() << "Finfo";
	else 
		fout << "	new NSrc" << argtypes_.size() << "Finfo";

	if (argtypes_.size() == 0) {
		fout << "(\n";
	} else {
		fout << "< " << argtypes_[ 0 ];
		for ( unsigned int i = 1; i < argtypes_.size(); i++ )
			fout << ", " << argtypes_[ i ];
		fout << " >(\n";
	}
	fout << "		\"" << name_ << "Out\", &" << 
		className << "Wrapper::get" << capsName << "Src, \n		\"";
	if ( internalMsgs_.size() > 0 ) {
		fout << internalMsgs_[ 0 ] << "In";
		for ( unsigned int i = 1; i < internalMsgs_.size(); i++ )
			fout << ", " << internalMsgs_[ i ] << "In";
	}
	if ( sharesConn_ )
		fout << "\", 1 ),\n";
	else 
		fout << "\" ),\n";
}

void Src::printWrapperCppFuncs(
	const string& className, ofstream& fout ) 
{
}
