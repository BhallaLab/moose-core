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

// Works on pre-tokenized data. If the type is 'single' or 'multi'
// then makes a new conn with the appropriate direction. Otherwise
// it looks up the type in the names of the existing conns. 
// Returns the connName if it succeeds, and "" if it fails.
const string& Conn::parse( 
	vector< Conn* >& connVec, 
	const string& name, const string& type, const string& direction, bool& sharedConn )
{
	const static string failed = "";
	Conn* c;

	// Start out by looking to see if Finfo is already handled by a
	// shared conn
	string temp = name + direction;
	vector< Conn* >::iterator i;
	vector< string >::iterator j;
	sharedConn = 0;
	for ( i = connVec.begin(); i != connVec.end(); i++ ) {
		for ( j = ( *i )->shared_.begin(); 
			j != ( *i )->shared_.end(); j++ ) {
			if ( *j == temp ) {
				// Do some type checking
				if ( type == ( *i )->sharedType_ ) {
					sharedConn = 1;
					return ( *i )->name_;
				} else {
					cerr << "Error: Conn::parse: " << temp << 
						": sharedType != type: " << 
						( *i )->sharedType_ << ", " << type << "\n";
					return failed;
				}
			}
		}
	}

	if ( type == "single" ) {
		c = new Conn( "UniConn", name + direction + "Conn" );
		connVec.push_back( c );
		return c->name_;
	} else if ( type == "multi" ) {
		if ( direction == "Out" ) {
			c = new Conn( "MultiConn", name + direction + "Conn" );
		} else if ( direction == "In" ) {
			c = new Conn( "PlainMultiConn", name + direction + "Conn" );
		} else {
			return failed;
		}
		connVec.push_back( c );
		return c->name_;
	} else if ( type == "solve" ) {
		c = new Conn( "SolveMultiConn", name + direction + "Conn" );
		connVec.push_back( c );
		return c->name_;
	}
	/*
	else {
		string temp = type + "Conn";
		vector< Conn* >::iterator i;
		for ( i = connVec.begin(); i != connVec.end(); i++ )
			if ( (*i)->name() == temp )
				return (*i)->polytype();
		return failed;
	}
	*/
	return failed;
}


void Conn::printConstructorWrapperH( ofstream& fout ) 
{
	if ( type_ != "UniConn" )
		fout << ",\n			" << name_ << "_( this )";
	else 
		fout << ",\n			// " << name_ <<
			" uses a templated lookup function";
}

void Conn::printFriendWrapperH( const string& className, ofstream& fout)
{
	if ( type_ == "UniConn" )
		fout << "	friend Element* " << name_ << className <<
			"Lookup( const Conn* );\n";
}

void Conn::printWrapperH( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	fout << "		static Conn* get" <<
		capsName << "( Element* e ) {\n";
	fout << "			return &( static_cast< " << className <<
		"Wrapper* >( e )->" << name_ << "_ );\n";
	fout << "		}\n";
}

void Conn::printPrivateWrapperH(
	const string& className, ofstream& fout ) 
{
	if ( type_ == "UniConn" )
		fout << "		" << type_ << "< " << name_ <<
			className << "Lookup > " << name_ << "_;\n";
	else 
		fout << "		" << type_ << " " << name_ << "_;\n";
}

void Conn::printWrapperCpp( const string& className, ofstream& fout ) 
{
}

void Conn::printWrapperCppFuncs(
	const string& className, ofstream& fout ) 
{
	if ( type_ == "UniConn" ) {
		fout << "Element* " << name_ << className <<
			"Lookup( const Conn* c )\n";
		fout << "{\n";
		fout << "	static const unsigned long OFFSET =\n";

		//This has been changed for compiler compatibility with Windows
		fout << "		FIELD_OFFSET ( " << className <<
			"Wrapper, " << name_ << "_ );\n";
		/*
		fout << "		(unsigned long) ( &" << className <<
			"Wrapper::" << name_ << "_ );\n";
		*/
		fout << "	return reinterpret_cast< " << className <<
			"Wrapper* >( ( unsigned long )c - OFFSET );\n";
		fout << "}\n\n";
	}
}
