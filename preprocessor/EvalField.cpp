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
#include "EvalField.h"

// MAX allowed size of vectors.
const unsigned long EvalField::MAX_VECTOR = 1000000; 

static EvalField* parseGetFunc( 
			vector< string >::iterator& i,
			vector< string >::iterator end,
			const string& type, FieldMode mode, int j );
static 	void parseSetFunc(
			vector< string >::iterator& i,
			vector< string >::iterator end,
			EvalField* ef );

void EvalField::parse( vector< EvalField* >& fieldVec,
	vector< string >& fieldString )
{
	int j = 0;
	vector<string>::iterator i;
	EvalField* f;
	FieldMode mode = RW;
	// bool hasIndex = 0;
	string name;
	string temp;
	string type;

	// Iterator needs a blank at the end.
	fieldString.push_back( "" );

	for(i = fieldString.begin(); i != fieldString.end(); i++) {
		f = 0;
		j = next_token(temp, *i, 0);
		if ( j == 0 )
			continue;

		if (temp == "readonly") {
			mode = RO;
			j = next_token(temp, *i, j);
		} else {
			mode = RW;
		}
		type = temp;
		f = parseGetFunc( i, fieldString.end(), type, mode, j );
		if ( f )
			fieldVec.push_back( f );
		else {
			cerr << "Error: EvalField::parse: '" << *i << "\n";
			return;
		}
		if ( mode == RW ) {
			i++;
			parseSetFunc( i, fieldString.end(), f );
		}
	}
}

// Here we have strings of the form
// double getTau() {...}		for the get function
// void setTau( value ) {...}		for the set function
// The type will specify the field type
// The first token should then be the get/set convolved with 
// the fieldname
// The second token should be an open bracket.

EvalField* parseGetFunc( 
	vector< string >::iterator& i,
	vector< string >::iterator end,
	const string& type, FieldMode mode, int j )
{
	string temp;
	string name;
	EvalField* ef;
	const string& s = *i;

	j = next_token( name, s, j );
	j = next_token( temp, s, j );
	if ( temp != "(" ) {
		cerr << "Syntax Error: EvalField::parseGetFunc 1: " <<
			s << "\n";
		return 0;
	}
	if ( name.length() < 4 ) {
		cerr << "Syntax Error: EvalField::parseGetFunc 2: " <<
			s << "\n";
		return 0;
	}
	if ( name.substr( 0, 3 ) != "get" ) {
		cerr << "Syntax Error: EvalField::parseGetFunc 3: " <<
			s << "\n";
		return 0;
	}
	j = next_token( temp, s, j );
	if ( temp != ")" )  {
		cerr << "No closing brace: EvalField::parseGetFunc in: " <<
			s << "\n";
		return 0;
	}

	vector< string >::iterator k = i;
	if ( !balanceBraces( k, end, j, 1 ) ) {
		cerr << "unbalanced braces: EvalField::parseGetFunc in: " <<
			s << "\n";
		return 0;
	}

	name = name.substr( 3 );
	name[0] = tolower( name[ 0 ] );
	ef = new EvalField( type, name, mode );

	for ( ++i; i != k ; i++ )
		ef->addToGetFunc( *i );
	
	return ef;
}

void parseSetFunc( 
	vector< string >::iterator& i,
	vector< string >::iterator end,
	EvalField* ef )
{
	string type;
	string temp;
	string name;
	int j;

	for ( ; i != end && i->length() == 0 ; i++)
		;
	const string& s = *i;
	j = next_token( type, s, 0 );
	if ( type != "void" ) {
		cerr << "Syntax Error: EvalField::parseSetFunc: expecting 'void' type: " <<
			s << "\n";
		return;
	}
	j = next_token( name, s, j );
	string setName = ef->name();
	setName[0] = toupper( setName[0] );
	setName = "set" + setName;
	if ( name != setName ) {
		cerr << "Syntax Error: EvalField::parseSetFunc: expecting '" <<
				setName << "', got: " << s << "\n";
		return;
	}
	/*
	if ( s.find( "{" ) == string::npos ) {
		cerr << "Syntax Error: EvalField::parseSetFunc: expecting '{' in: " << s << "\n";
		return;
	}
	*/
	// If it got here we bludgeon ahead till the closing brace
	
	vector< string >::iterator k = i;
	if ( !balanceBraces( k, end, j, 1 ) ) {
		cerr << "unbalanced braces: EvalField::parseSetFunc in: " <<
			s << "\n";
		return;
	}
	
	ef->addToSetFunc( s.substr( j ) );
	for ( ++i ; i != k; i++ )
		ef->addToSetFunc( *i );
}

void EvalField::printWrapperH( const string& className, ofstream& fout )
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	fout << "		" << type_ << " localGet" << capsName << 
		"() const;\n";
	fout << "		static " << type_ << " get" << capsName <<
		"( const Element* e ) {\n";
	fout << "			return static_cast< const " << className << 
		"Wrapper* >( e )->\n" << 
		"			localGet" << capsName << "();\n";
	fout << "		}\n";

	if ( mode_ == RW ) {
		fout << "		void localSet" << capsName << 
			"( " << type_ << " value );\n";
		fout << "		static void set" << capsName << 
			"( Conn* c, " << type_ << " value ) {\n";
		fout << "			static_cast< " << className << 
			"Wrapper* >( c->parent() )->\n" <<
			"			localSet" << capsName << 
			"( value );\n";
		fout << "		}\n";
	}
}

void EvalField::printWrapperCpp( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	string finfoName = "ValueFinfo";
	fout << "	";
	if ( mode_ == RW )
		finfoName = "ValueFinfo";
	else
		finfoName = "ReadOnlyValueFinfo";

	fout << "new " << finfoName << "< " << type_ << " >(\n";
	fout << "		\"" << name_ << "\", &" <<
		className << "Wrapper::get" << capsName << ", ";
	if ( mode_ == RW ) {
		fout << "\n		&" << className << "Wrapper::set" << capsName << ", ";
	}
	fout << "\"" << type_ << "\" ),\n";
}

void EvalField::printWrapperCppFuncs(
	const string& className, ofstream& fout ) 
{
	vector< string >::iterator i;
	string capsName = name_;

	capsName[0] = toupper( capsName[ 0 ] );
	fout << type_ << " " << className << 
		"Wrapper::localGet" << capsName << "() const\n";
	fout << "{\n";
	for ( i = getFunc_.begin(); i != getFunc_.end(); i++ )
		fout << *i << "\n";
	fout << "}\n";

	if ( mode_ == RW ) {
		fout << "void " << className << "Wrapper::localSet" << capsName;
		for ( i = setFunc_.begin(); i != setFunc_.end(); i++ )
			fout << *i << "\n";
		fout << "}\n";
	}
}
