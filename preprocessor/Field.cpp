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
#include "Field.h"

// MAX allowed size of vectors.
const unsigned long Field::MAX_VECTOR = 1000000; 

void Field::parse( vector< Field* >& fieldVec,
	vector< string >& fieldString )
{
	int j = 0;
	vector<string>::iterator i;
	Field* f;
	FieldMode mode = RW;
	// bool hasIndex = 0;
	string name;
	string temp;
	string type;

	for(i = fieldString.begin(); i != fieldString.end(); i++) {
		f = 0;
		j = next_token(temp, *i, 0);

		if (temp == "const") {
			mode = CONST;
			j = next_token(temp, *i, j);
		} else if (temp == "readonly") {
			mode = RO;
			j = next_token(temp, *i, j);
		} else {
			mode = RW;
		}
		type = temp;
		if (type == "class" ) {
			f = parseClassField( i, fieldString.end(), mode, j );
		} else if ( type == "mooseclass" ) {
			f = parseMooseClassField( *i, mode, j );
		} else {
			f = parseSimpleField( *i, type, mode, j );
		}
		if ( f )
			fieldVec.push_back( f );
	}
}

Field* Field::parseSimpleField(
	const string& s, const string& type, FieldMode mode, int j )
{
	bool isVector = 0;
	bool hasInitVal = 0;
	string initVal;
	string temp;
	string name;
	unsigned long size;

	j = next_token( name, s, j );
	j = next_token( temp, s, j );
	if ( temp == "[" ) {
		isVector = 1;
		j = next_token( temp, s, j );
		if (temp == "]") {
			size = 0;
			j = next_token( temp, s, j );
		} else {
			size = atol( temp.c_str() );
			if ( size < 0 )
				size = 0;
			if ( size > MAX_VECTOR )
				size = MAX_VECTOR;
			j = next_token( temp, s, j );
			if (temp != "]") {
				cout << "syntax error: parseSimpleField: " << s << "\n";
				return 0;
			}
			j = next_token( temp, s, j );
		}
	}

	if ( temp == "=" ) { // here we set up initialization 
		j = next_token( initVal, s, j );
		hasInitVal = 1;
		j = next_token( temp, s, j );
	}

	if ( temp == ";" ) {
		if ( hasInitVal )
			return new Field( type, name, mode, isVector, size,
				initVal);
		else
			return new Field( type, name, mode, isVector, size );
	}
	return 0;
}

// Not yet implemented, but we need to get past any definitions.
Field* Field::parseClassField(
	vector< string >::iterator& i, 
	vector< string >::iterator end,
	FieldMode mode,
	unsigned int j
	)
{
	string className;
	string name;
	string tok;

	j = next_token( className, *i, j );
	if ( j == i->length() ) {
		i++;
		j = 0;
	}
	j = next_token( tok, *i, j );
	if ( tok != "{" ) {
		cout << "syntax error: parseSimpleField: " << *i << "\n";
		return 0;
	}
	j = next_token( tok, *i, j );
	while ( tok != "}" ) {
		if ( j == i->length() ) {
			i++;
			j = 0;
			if (i == end)
				return 0;
		}
		j = next_token( tok, *i, j );
	}
	j = next_token( name, *i, j );
	j = next_token( tok, *i, j );
	if ( tok != ";" )
		return 0;
	else {
		cout << "Class definition seems OK, but not yet implemented\n";
	}
	return 0;
}

Field* Field::parseMooseClassField(
	const string& s, FieldMode mode, int j
	)
{
	string type;
	string name;
	string tok;
	j = next_token( type, s, j );
	j = next_token( name, s, j );
	j = next_token( tok, s, j );
	if ( tok != ";" ) {
		cout << "syntax error: parseMooseClassField: " << s << "\n";
		return 0;
	}
	cout << "MooseClass definition seems OK, but not yet implemented\n";
	return 0;
}

void Field::printConstructor( ofstream& fout ) 
{
	if ( isVector_ && size_ > 0 )
		fout << "			" << name_ << "_.reserve( " << size_ << " ) ;\n";
	else if ( hasInitVal_ )
		fout << "			" << name_ << "_ = " << initVal_ << ";\n";
}

void Field::printPrivateHeader( ofstream& fout ) 
{
	fout << "		";
	if ( mode_ == CONST )
		fout << "const ";
	if ( isVector_ )
		fout << "vector < " << type_ << " > " << name_ << "_;\n";
	else 
		fout << type_ << " " << name_ << "_;\n";
}

void Field::printWrapperH( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	if ( isVector_ ) {
		if ( mode_ == RW ) {
			fout << "		static void set" << capsName << 
				"(\n";
			fout << "			Element* e, unsigned long index, " <<
				type_ << " value );\n";
		}
		fout << "		static " << type_ << " get" << capsName << 
			"(\n";
		fout << "			const Element* e, unsigned long index );\n";
	} else {
		if ( mode_ == RW ) {
			fout << "		static void set" << capsName << 
				"( Conn* c, " << type_ << " value ) {\n";
			fout << "			static_cast< " << className << 
				"Wrapper* >( c->parent() )->" << name_ << "_ = value;\n";
			fout << "		}\n";
		}
		fout << "		static " << type_ << " get" << capsName <<
			"( const Element* e ) {\n";
		fout << "			return static_cast< const " << className << 
			"Wrapper* >( e )->" << name_ << "_;\n";
		fout << "		}\n";
	}
}

void Field::printWrapperCpp( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	string finfoName = "ValueFinfo";
	fout << "	";
	if ( isVector_ ) {
		if ( mode_ == RW )
			finfoName = "ArrayFinfo";
		else
			finfoName = "ReadOnlyArrayFinfo";
	} else {
		if ( mode_ == RW )
			finfoName = "ValueFinfo";
		else
			finfoName = "ReadOnlyValueFinfo";
	}

	fout << "new " << finfoName << "< " << type_ << " >(\n";
	fout << "		\"" << name_ << "\", &" <<
		className << "Wrapper::get" << capsName << ", ";
	if ( mode_ == RW )
		fout << "\n		&" << className << "Wrapper::set" << capsName << ", ";
	fout << "\"" << type_ << "\" ),\n";
}

void Field::printWrapperCppFuncs(
	const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	if ( isVector_ ) {
		if ( mode_ == RW ) {
			fout << "void " << className << 
				"Wrapper::set" << capsName << "(\n";
			fout << "	Element* e , unsigned long index, " <<
				type_ << " value )\n";
			fout << "{\n";
			fout << "	" << className << 
				"Wrapper* f = static_cast< " << className << "Wrapper* >( e );\n";
			fout << "	if ( f->" << name_ << 
				"_.size() > index )\n";
			fout << "		f->" << name_ << "_[ index ] = value;\n";
			fout << "}\n\n";
		}
		fout << type_ << " " << className << "Wrapper::get" << capsName << "(\n";
		fout << "	const Element* e , unsigned long index )\n";
		fout << "{\n";
		fout << "	const " << className << 
			"Wrapper* f = static_cast< const " << className << "Wrapper* >( e );\n";
		fout << "	if ( f->" << name_ << "_.size() > index )\n";
		fout << "		return f->" << name_ << "_[ index ];\n";
		fout << "	return f->" << name_ << "_[ 0 ];\n";
		fout << "}\n\n";
	}
}
