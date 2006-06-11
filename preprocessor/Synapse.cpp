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
#include "Synapse.h"

void Synapse::parse( vector< Synapse* >& synapseVec,
	vector< string >& synapseString )
{
	unsigned int j = 0;
	vector<string>::iterator i;
	string name;
	string temp;
	string type;
	string infoType;
	vector< string > argtypes;
	vector< string > argnames;

	for(i = synapseString.begin(); i != synapseString.end(); i++) {
		j = next_token(type, *i, 0);
		j = next_token(name, *i, j);
		j = next_token(temp, *i, j);
		if ( temp != "(" ) { // oops.
			cerr << "Error: Failed on synapse specification line:\n";
			cerr << *i << "\n";
			continue;
		}
		// Now we go through and scan arguments.
		j = next_token(temp, *i, j);
		while ( j < i->length() && temp != ")" ) {
			argtypes.push_back( temp );
			j = next_token( temp, *i, j); // Name of arg
			argnames.push_back( temp );
			j = next_token( temp, *i, j); // comma/closing bracket
			if ( temp == "," )
				j = next_token( temp, *i, j); // next argtype
		}
		j = next_token( infoType, *i, j);
		j = next_token( temp, *i, j);
		if ( temp != ";" ) { // oops.
			cerr << "Error: Failed on synapse specification line:\n";
			cerr << *i << "\n";
			continue;
		}
		synapseVec.push_back(
			new Synapse( type, name, argtypes, argnames, infoType ) );
	}
}

void Synapse::printWrapperH( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	fout << "		static void set" << capsName << "Value(\n";
	fout << "			Element* e , unsigned long index, " <<
		infoType_ << " value );\n";
	fout << "		static " << infoType_ << " get" << capsName << 
		"Value(\n";
	fout << "			const Element* e , unsigned long index );\n";

	fout << "		static vector< Conn* >& get" << capsName <<
		"Conn( Element* e ) {\n";
	fout << "			return reinterpret_cast< vector< Conn* >& >(\n";
	fout << "				static_cast< " << className <<
		"Wrapper* >( e )->" << name_ << "Conn_\n";
	fout << "			);\n		}\n";
	fout << "		static unsigned long new" << capsName <<
		"Conn( Element* e );\n";
	fout << "		static void " << name_ << "Func( Conn* c";
	for ( unsigned int i = 0; i < argtypes_.size(); i++ ) {
		fout << ", " << argtypes_[ i ] << " " << argnames_[ i ];
	}
	fout << " );\n\n";
}

void Synapse::printPrivateWrapperH( ofstream& fout ) 
{
	fout << "		vector< SynConn< " << infoType_ << " >* > " <<
		name_ << "Conn_;\n";
}

void Synapse::printWrapperCpp( const string& className, ofstream& fout ) 
{
	unsigned int i;
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	fout << "	new ArrayFinfo< " << infoType_ << " >(\n";
	fout << "		\"" << name_ << "Value\", &" <<
		className << "Wrapper::get" << capsName << "Value,\n";
	fout << "		&" << className << "Wrapper::set" << capsName <<
		"Value, \"" << type_ << "\" ),\n";

	fout << "	new Synapse" << argtypes_.size() << "Finfo";

	if (argtypes_.size() == 0) {
		fout << "(\n";
	} else {
		fout << "< " << argtypes_[ 0 ];
		for ( i = 1; i < argtypes_.size(); i++ )
			fout << ", " << argtypes_[ i ];
		fout << " >(\n";
	}
	fout << "		\"" << name_ << "In\", &" << 
		className << "Wrapper::" << name_ << "Func,\n";
	fout << "		&" <<
		className << "Wrapper::get" << capsName << "Conn,";
	fout << " &" <<
		className << "Wrapper::new" << capsName << "Conn, \"";

	if ( internalMsgs_.size() == 0 ) {
		fout << "\" ),\n\n";
	} else {
		fout << internalMsgs_[ 0 ];
		for ( i = 1; i < internalMsgs_.size(); i++ )
			fout << ", " << internalMsgs_[ i ];
		fout << "\" ),\n\n";
	}
}

// User will need to repeat the set/get functions for each of the
// fields in complex classes
void Synapse::printWrapperCppFuncs(
	const string& className, ofstream& fout ) 
{
	unsigned int i;
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	// Here is the set function
	fout << "void " << className << "Wrapper::set" << capsName << 
		"Value(\n";
	fout << "	Element* e , unsigned long index, " <<
		infoType_ << " value )\n";
	fout << "{\n";
	fout << "	" << className << 
		"Wrapper* f = static_cast< " << className << "Wrapper* >( e );\n";
	fout << "	if ( f->" << name_ << 
		"Conn_.size() > index )\n";
	fout << "		f->" << name_ <<
		"Conn_[ index ]->value_ = value;\n";
	fout << "}\n\n";

	// Here is the get function
	fout << infoType_ << " " << className << "Wrapper::get" << capsName << 
			"Value(\n";
	fout << "	const Element* e , unsigned long index )\n";
	fout << "{\n";
	fout << "	const " << className << 
		"Wrapper* f = static_cast< const " << className << "Wrapper* >( e );\n";
	fout << "	if ( f->" << name_ << "Conn_.size() > index )\n";
	fout << "		return f->" << name_ << "Conn_[ index ]->value_;\n";
	fout << "	return f->" << name_ << "Conn_[ 0 ]->value_;\n";
	fout << "}\n\n";

	// Here is the Func to be executed when synaptic input arrives.
	fout << "void " << className << "Wrapper::" << name_ << 
		"Func( Conn* c";
	for ( i = 0; i < argtypes_.size(); i++ ) {
		fout << ", " << argtypes_[ i ] << " " << argnames_[ i ];
	}
	fout << " )\n{\n";
	fout << "	SynConn< " << infoType_ << 
		" >* s = static_cast< SynConn< " << infoType_ <<
		" >* >( c );\n";
	fout << "	" << className << "Wrapper* temp = static_cast< " << 
		className << "Wrapper* >( c->parent() );\n";
	fout << "	// Here we do the synaptic function\n";
	fout << "}\n\n";

	// Here is the function to make a new synaptic connection.
	fout << "unsigned long " << className << "Wrapper::new" <<
		capsName << "Conn( Element* e ) {\n";
	fout << "	" << className << "Wrapper* temp = static_cast < " <<
		className << "Wrapper* >( e );\n";
	fout << "	SynConn< " << infoType_ << " >* s = new SynConn< " <<
		infoType_ << " >( e );\n";
	fout << "	temp->" << name_ << "Conn_.push_back( s );\n";
	fout << "	return temp->" << name_ << "Conn_.size( ) - 1;\n";
	fout << " }\n";
}
