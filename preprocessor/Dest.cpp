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
#include "Dest.h"

void Dest::parse( vector< Dest* >& destVec, 
	vector< Conn* >& connVec, 
	vector< Src* >& srcVec, 
	vector< string >& destString )
{
	int j = 0;
	vector< string >::iterator i;
	vector< string > argtypes;
	string name;
	string temp;
	string type;
	string connName;
	bool sharesConn = 0;

	for(i = destString.begin(); i != destString.end();) {
		j = next_token( type, *i, 0);
		j = next_token( name, *i, j);
		j = next_token( temp, *i, j); // should be open bracket

		connName = Conn::parse( connVec, name, type, "In", sharesConn );
		if ( connName == "" ) {
			cout << "syntax error: Dest::Conn::parse: " << *i << "\n";
			i++;
			continue;
		}
		if ( temp != "(" ) {
			cout << "syntax error: Dest::parse: " << *i << "\n";
			i++;
			continue;
		}
		string argstr = i->substr(j);
		int nesting = 0;
		Dest* d = new Dest( type, name, connName, argstr, sharesConn );
		if ( argstr.find("{") != string::npos ) {
			nesting = 1;
		} else if ( argstr.find(";") != string::npos ) {
			nesting = -1;
		}
		i++;
		while ( nesting == 0 && i != destString.end() ) {
			if ( i->find("{") != string::npos )
				nesting = 1;
			d->addLine( *i );
			i++;
		}
		while ( nesting > 0 && i != destString.end() ) {
			if ( i->find("{") != string::npos )
				nesting++;
			if ( i->find("}") != string::npos )
				nesting--;
			if ( nesting > 0 )
				d->addLine( *i );
			i++;
		}
			d->internalParse( srcVec );
			destVec.push_back( d );
		/*
		if ( i != destString.end() ) {
		}
		*/
	}
}

class SrcNameMatch: public unary_function< Src*, bool > {
	public:
		explicit SrcNameMatch( const string& name )
			: name_( name )
		{ }
		bool operator()( const Src* const& other ) const {
			return other->name() == name_;
		}

	private:
		string name_;
};

	//	single solve( vector< double >* y, double t, double dt );
// Parse the argument list to work out the templating
// make a list of candidate outgoing messages
void Dest::internalParse( vector< Src* >& srcVec ) 
{
	vector< string >::iterator i;
	string name;
	string temp;
	unsigned int j = 0;

	j = next_token( temp, argstr_, 0);
	while ( j < argstr_.length() && temp != ")" ) {
		temp = checkForVector( temp, argstr_, j );
		argtypes_.push_back( temp );
		j = next_token( temp, argstr_, j); // Name of arg
		argnames_.push_back( temp );
		j = next_token( temp, argstr_, j); // comma, or closing bracket
		if ( temp == "," )
			j = next_token( temp, argstr_, j); // next argtype
	}

	for (i = lines_.begin(); i != lines_.end() ; i++)
	{
		j = next_token( name, *i, 0);
		if ( j < i->length() )
			j = next_token( temp, *i, j);
		else 
			temp = "";

		// Check if the name is defined as a MsgSrc. That means
		// it should be expanded into the send command.
		if ( find_if( srcVec.begin(), srcVec.end(), SrcNameMatch( name )) != srcVec.end() && temp == "(" ) {
		// if ( name != "if" && temp == "(" ) // Deem it a possible outgoing msg
			targetnames_.push_back( name );
			destFuncLines_.push_back(
				"			" + name + "Src_.send(" + i->substr( j ) );
		} else {
			destFuncLines_.push_back( *i );
		}
	}
}

void Dest::printConstructor( ofstream& fout ) 
{
	;
}

void Dest::printWrapperH( const string& className, ofstream& fout ) 
{
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );
	unsigned int i;

	fout << "		void " << name_ << "FuncLocal( ";
	for ( i = 0; i < argtypes_.size(); i++ ) {
		fout << argtypes_[ i ] << " " << argnames_[ i ];
		if ( i < argtypes_.size() - 1 )
			fout << ", ";
	}
	if ( destFuncLines_.size() < 3 ) {
		fout << " ) {\n";
		for ( i = 0; i < destFuncLines_.size(); i++ )
			fout << destFuncLines_[ i ] << "\n";
		fout << "		}\n";
	} else {
		fout << " );\n";
	}

	fout << "		static void " << name_ << "Func( Conn* c";
	for ( i = 0; i < argtypes_.size(); i++ ) {
		fout << ", " << argtypes_[ i ] << " " << argnames_[ i ];
	}
	fout << " ) {\n";
	fout << "			static_cast< " << className <<
		"Wrapper* >( c->parent() )->\n				" << name_
		<< "FuncLocal( ";
	for ( i = 0; i < argnames_.size(); i++ ) {
		fout << argnames_[ i ];
		if ( i < argnames_.size() - 1 )
			fout << ", ";
	}
	fout << " );\n		}\n\n";
}

void Dest::printPrivateWrapperH(
	const string& className, ofstream& fout ) 
{
	/*
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );
	fout << "		";
	if ( type_ == "single" )
		fout << "SingleMsgDest" << argtypes_.size();
	else 
		fout << "NMsgDest" << argtypes_.size();
	if (argtypes_.size() == 0) {
		fout << " " << name_ << "Dest_;\n";
	} else {
		fout << "< " << argtypes_[ 0 ];
		for ( int i = 1; i < argtypes_.size(); i++ )
			fout << ", " << argtypes_[ i ];
		fout << " > " << name_ << "Dest_;\n";
	}
	*/
}

void Dest::printWrapperCpp( const string& className, ofstream& fout ) 
{
	unsigned int i;
	string capsName = name_;
	capsName[0] = toupper( capsName[ 0 ] );

	fout << "	new Dest" << argtypes_.size() << "Finfo";
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
	string capsConnName = connName_;
	capsConnName[0] = toupper( capsConnName[0] );
	fout << "		&" <<
		className << "Wrapper::get" << capsConnName << ", \"";

	if ( internalMsgs_.size() > 0 ) {
		fout << internalMsgs_[ 0 ] << "Out";
		for ( i = 1; i < internalMsgs_.size(); i++ )
			fout << ", " << internalMsgs_[ i ] << "Out";
	}
	/*
	if ( internalMsgs_.size() == 0 ) {
		fout << "\" ),\n";
	} else {
		fout << internalMsgs_[ 0 ] << "Out";
		for ( int i = 1; i < internalMsgs_.size(); i++ )
			fout << ", " << internalMsgs_[ i ] << "Out";
		fout << "\" ),\n";
	}
	*/
	if ( sharesConn_ ) 
		fout << "\", 1 ),\n";
	else
		fout << "\" ),\n";
}

void Dest::printWrapperCppFuncs(
	const string& className, ofstream& fout ) 
{
	unsigned int i;

	if ( destFuncLines_.size() >= 3 ) {
		fout << "void " << className << "Wrapper::" <<
			name_ << "FuncLocal( ";
		for ( i = 0; i < argtypes_.size(); i++ ) {
			fout << argtypes_[ i ] << " " << argnames_[ i ];
			if ( i < argtypes_.size() - 1 )
				fout << ", ";
		}
		fout << " )\n{\n";
		for ( i = 0; i < destFuncLines_.size(); i++ )
			fout << destFuncLines_[ i ] << "\n";
		fout << "}\n";
	}
}
