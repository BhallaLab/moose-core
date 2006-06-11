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
#include "Conn.h"
#include "Shared.h"
#include "Src.h"
#include "Dest.h"
#include "Synapse.h"
#include "Mpp.h"

// this is the moose preprocessor. It takes a .mh file and generates
// a .h, a wrapper.h, and a wrapper.cpp file.

// This function tests if a file exists, and if force is present,
// renames it.
void testFile( const string& name, bool force )
{
	fstream fin( name.c_str(), ios::in);
	if ( !fin.good() )
		return;
	if ( force ) { // Files exist. Move them.
		string temp = name + ".old";
		cerr << "Warning: existing file '" << name << 
			"' renamed to '" << temp << "'.\n";
		rename( name.c_str(), temp.c_str() );
	} else {
		cerr << "Error: target file '" << name << "' exists. Exiting\n";
		exit( 0 );
	}
}

int main(int argc, const char** argv) {
	Mpp mh;
	string fname;
	string basename;
	bool force = 0;

	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " classname -force\n";
		cerr << "The program looks for a file named <classname>.mh\n";
		cerr << "and outputs .h, Wrapper.h, and Wrapper.cpp files\n";
		cerr << "If <classname> is -test then it runs an internal\n";
		cerr << "test without any file input\n";
		cerr << "The -force flag removes existing output files.\n";
		return 0;
	}

	fname = argv[1];
	if (fname == "-test") {
		runTest();
		exit( 0 );
	}
	if (fname.rfind(".mh") == fname.length() - 3)  {
		basename = fname.substr( 0, fname.length() - 3 );
	} else {
		basename = fname;
		fname = fname + ".mh";
	}
	if ( argc == 3 && argv[2][0] == '-' && argv[2][1] == 'f' )
		force = 1;

	fstream fin(fname.c_str(), ios::in);
	if (!fin.good()) {
		cout << "Error: file " << fname << "not found\n";
		exit(0);
	}

	testFile( basename + ".h", force );
	testFile( basename + "Wrapper.h", force );
	testFile( basename + "Wrapper.cpp", force );

	string s = "";
	while ( s.find( "class" ) == string::npos ) {
		mh.addHeader( s );
		std::getline( fin, s );
		if ( !fin.good() ) {
			cerr << "Error: class definition not found. Exiting\n";
			exit(0);
		}
	}
	mh.addLine( s );

	char c;
	parse_state state = OK;

	while(fin.get(c))
	{
		switch (state) {
			case OK:
				if (c == '/') {
					fin.get(c);
					if (c == '/') {
						state = LINECOMMENT;
					} else if (c == '*') {
						state = LONGCOMMENT;
					} else {
						mh.addChar('/');
						fin.putback(c);
					}
				} else {
					mh.addChar(c);
				}
				break;
			case LINECOMMENT:
				if (c == '\n') {
					state = OK;
					mh.addChar(c);
				}
				break;
			case LONGCOMMENT:
				if (c == '*') {
					fin.get(c);
					if (c == '/')
						state = OK;
					else
						fin.putback(c);
				}
				break;
			default:
				break;
		}
	}
	mh.parse();
	mh.printHeader( basename );
	mh.printWrapperH( basename );
	mh.printWrapperCpp( basename );
}

// Checks if entire line is whitespace
bool is_whitespace(const string& line)
{
	const char* l = line.c_str();
	for (; *l; l++)
		if (!isspace(*l))
			return 0;
	return 1;
}

bool istok(const char c)
{
	static char* toks = "{}():=;,[]";
	for (char* t = toks; *t; t++)
		if (*t == c)
			return 1;
	return 0;
}

	// string s("void func(double a1, double a2) {");

//
// returns starting point on string s for next round.
// the next token is stored in ret
// s is the input string.
// i is the starting point on string s for this round.
//
int next_token(string& ret, const string& s, int i)
{
	string s1 = s.substr(i);
	unsigned int j = s1.find_first_not_of(" 	");
	if (j == string::npos)
		return s.length();
	string s2 = s1.substr(j);
	unsigned int k = s2.find_first_of(" 	{}():=;,[]");
	if (k == 0)
		k = 1;
	ret = s2.substr(0, k);
	if (k == string::npos) {
		return s.length();
	}
	return i + j + k;
}

// Looks for arguments of the form:
// single solve( vector< double >* y, vector< double >* yPrime );
// This function returns the string 
// vector< double >*
// as the type information. Other template classes are allowed too.
// The temp string should be:
// 	vector
// j should be the position of the angle bracket <
// When it returns j should point to the comma after the first y.
// Should handle nested braces and the like.

const string checkForVector( const string& temp, 
	const string& line, unsigned int& j )
{
	if (temp.length() == 0) {
		cerr << "Error: failed to parse line '" << line << "'\n";
		return "";
	}
	if ( temp[ temp.length() - 1 ] == '<' || line[j] == '<' ) {
		int nBraces = 0;
		if ( temp[ temp.length() - 1 ] == '<' )
			nBraces = 1;
		for (unsigned int i = j ; i < line.length(); i++) {
			if ( line[i] == '<' )
				nBraces++;
			if ( line[i] == '>' )
				nBraces--;
			if ( nBraces == 0 ) {
				// Now clear up any dangling pointers or references
				i++;
				while( i < line.length() && 
					(line[i] == ' ' || line[i] == '&' || line[i] == '*')
				)
					i++;
				string ret = temp + line.substr( j, i - j );
				j = i;
				return ret;
			}	
		}
		// If it gets here, something is wrong.
		cerr << "Error: failed to parse line '" << line << "'\n";
		cerr << "Possibly a problem with handling template braces\n";
	}
	// If it is not a template, just return the original temp string
	// and nothing changes.
	return temp;
}

// Prints a bunch of strings which may have leading whitespace, into
// a single multiline string
void printQuotedText( ofstream& fout, vector< string >& s )
{
	vector< string >::iterator i;
	for ( i = s.begin(); i != s.end(); i++ ) {
		unsigned long p = i->find_first_not_of( " 	" );
		if ( i == s.begin() )
			fout << i->substr( p );
		else 
			fout << "\\n" << i->substr( p );
	}
	fout << "\",\n";
}

///////////////////////////////////////////////////////////////////
// Mpp Class definition stuff
///////////////////////////////////////////////////////////////////

void Mpp::parse() 
{
	parseHeader();
	vector<string>::iterator i;
	string ret;
	int j = 0;
	// Chop it up into tokens
	for(i = original_.begin(); i != original_.end(); i++) {
		j = next_token(ret, *i, 0);
		if (ret == "class")
			break;
	}
	next_token(className_, *i, j);

	vector<string>* vec = &pub_;

	for(; i != original_.end(); i++) {
		next_token(ret, *i, 0);
		if (ret == "#endif")
			break;
		if (ret == "field") {
			vec = &fieldString_;
		} else if (ret == "src") {
			vec = &msgsrcString_;
		} else if (ret == "dest") {
			vec = &msgdestString_;
		} else if (ret == "synapse") {
			vec = &synapseString_;
		} else if (ret == "shared") {
			vec = &sharedString_;
		} else if (ret == "public") {
			vec = &pub_;
		} else if (ret == "private") {
			vec = &priv_;
		} else if (ret == "private_wrapper") {
			vec = &privWrapper_;
		} else if (ret == "author") {
			vec = &author_;
		} else if (ret == "description") {
			vec = &description_;
		} else if (ret == "wrapper_cpp") {
			// This should be unnecessary. There is a hack in the 
			// parsing for priv_ and priv_wrapper_ that checks
			// for the terminal brace and shifts over to wrapperCpp_
			vec = &wrapperCpp_;
		} else {
			vec->push_back(*i);
		}
	}

	Field::parse( fieldVec_, fieldString_ );
	Shared::parse( sharedVec_, connVec_, sharedString_ );
	Src::parse( srcVec_, connVec_, msgsrcString_ );
	Dest::parse( destVec_, connVec_, srcVec_, msgdestString_ );
	Synapse::parse( synapseVec_, synapseString_ );
	parseInternalMsgs();
}

void Mpp::parseInternalMsgs()
{
	vector< Dest* >::iterator i;
	vector< Src* >::iterator j;
	for ( i = destVec_.begin(); i != destVec_.end(); i++ ) {
		for ( j = srcVec_.begin(); j != srcVec_.end(); j++ ) {
			if ( ( *i )->matchSrc( ( *j )->name() ) ) {
				( *i )->addInternalMsg( ( *j )->name() );
				( *j )->addInternalMsg( ( *i )->name() );
			}
		}
	}
}

void Mpp::parseHeader( )
{
	vector< string >::iterator i;
	for (i = startstuff_.begin() + 1; i < startstuff_.end(); i++ ) {
		if ( i->find( "#include" ) == 0 ) {
			includes_.push_back( *i );
		} else {
			headerText_.push_back( *i );
		}
	}
}

void Mpp::printHeader( const string& name )
{
	ofstream fout( (name + ".h").c_str(), ios::out );

	unsigned int j = 0;
	string ret;
	vector<string>::iterator i;

	// fout << copyleft;

	for ( j = 0; j < headerText_.size(); j++ )
		fout << headerText_[ j ] << "\n";

	fout << "#ifndef _" << className_ << "_h\n";
	fout << "#define _" << className_ << "_h\n";
	
	fout << pub_[0] << '\n';
	fout << pub_[1] << '\n';
	fout << "	friend class " << className_ << "Wrapper;\n";

	fout << "	public:\n";
	fout << "		" << className_ << "()\n";
	fout << "		{\n";
	for ( j = 0; j < fieldVec_.size(); j++ )
		fieldVec_[ j ]->printConstructor( fout );
	fout << "		}\n";

	for(i = pub_.begin() + 2; i != pub_.end(); i++)
		fout << (*i) << '\n';

	fout << "\n	private:\n";

	for ( j = 0; j < fieldVec_.size(); j++ )
		fieldVec_[ j ]->printPrivateHeader( fout );

	for(i = priv_.begin(); i != priv_.end(); i++) {
		if ( *i != "};" ) {
			fout << (*i) << '\n';
		} else {
			for ( i = i + 1; i != priv_.end(); i++)
				wrapperCpp_.push_back( *i );
			break;
		}
	}

	fout << "};\n";
	fout << "#endif // _" << className_ << "_h\n";
}

void Mpp::printWrapperH(const string& name)
{
	string fname = name + "Wrapper.h";
	ofstream fout( fname.c_str(), ios::out );
	unsigned long i;

	for ( i = 0; i < headerText_.size(); i++ )
		fout << headerText_[ i ] << "\n";

	// fout << copyleft;
	fout << "#ifndef _" << className_ << "Wrapper_h\n";
	fout << "#define _" << className_ << "Wrapper_h\n";
	fout << "class " << className_ << "Wrapper: \n";
	fout << "	public " << className_ << ", public Neutral\n{\n";

	for ( i = 0; i < connVec_.size(); i++ )
		connVec_[ i ]->printFriendWrapperH( className_, fout );

	fout << "    public:\n";
	fout << "		" << className_ << "Wrapper(const string& n)\n";
	fout << "		:\n";
	fout << "			Neutral( n )";
	for (i = 0; i < srcVec_.size(); i++)
		srcVec_[i]->printConstructorWrapperH( fout );
	for (i = 0; i < connVec_.size(); i++)
		connVec_[i]->printConstructorWrapperH( fout );
	fout << "\n		{\n"; 
	fout << "			;\n"; 
	fout << "		}\n"; 

	fout << "///////////////////////////////////////////////////////\n";
	fout << "//    Field header definitions.                      //\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( unsigned long k = 0; k < fieldVec_.size(); k++ )
		fieldVec_[ k ]->printWrapperH( className_, fout );

	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Msgsrc header definitions .                       //\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( unsigned long k = 0; k < srcVec_.size(); k++ )
		srcVec_[ k ]->printWrapperH( className_, fout );

	fout << "///////////////////////////////////////////////////////\n";
	fout << "// dest header definitions .                         //\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( unsigned long k = 0; k < destVec_.size(); k++ )
		destVec_[ k ]->printWrapperH( className_, fout );

	fout << "\n";
	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Synapse creation and info access functions.       //\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( unsigned long k = 0; k < synapseVec_.size(); k++ )
		synapseVec_[ k ]->printWrapperH( className_, fout );

	fout << "\n";
	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Conn access functions.                            //\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( unsigned long k = 0; k < connVec_.size(); k++ )
		connVec_[ k ]->printWrapperH( className_, fout );

	fout << "\n";
	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Class creation and info access functions.         //\n";
	fout << "///////////////////////////////////////////////////////\n";

	fout << "		static Element* create(\n";
	fout << "			const string& name, Element* pa, const Element* proto ) {\n";
	fout << "			// Put tests for parent class here\n";
	fout << "			// Put proto initialization stuff here\n";
	fout << "			// const " << className_ <<
		"* p = dynamic_cast<const " << className_ << " *>(proto);\n";
	fout << "			// if (p)... and so on. \n";

	fout << "			return new " << className_ << "Wrapper(name);\n";
	fout << "		}\n\n";
	fout << "		const Cinfo* cinfo() const {\n";
	fout << "			return &cinfo_;\n";
	fout << "		}\n\n";

	fout << "\n    private:\n";
	fout << "///////////////////////////////////////////////////////\n";
	fout << "// MsgSrc template definitions.                      //\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( unsigned long k = 0; k < srcVec_.size(); k++ )
		srcVec_[ k ]->printPrivateWrapperH( className_, fout );
	for ( unsigned long k = 0; k < connVec_.size(); k++ )
		connVec_[ k ]->printPrivateWrapperH( className_, fout );

	fout << "\n";
	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Synapse definition.                               //\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( unsigned long k = 0; k < synapseVec_.size(); k++ )
		synapseVec_[ k ]->printPrivateWrapperH( fout );

	fout << "\n";
	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Private functions and fields for the Wrapper class//\n";
	fout << "///////////////////////////////////////////////////////\n";
	vector<string>::iterator j;
	for(j = privWrapper_.begin(); j != privWrapper_.end(); j++) {
		if ( *j != "};" ) {
			fout << (*j) << '\n';
		} else {
			for (j = j + 1; j != privWrapper_.end(); j++)
				wrapperCpp_.push_back( *j );
			break;
		}
	}
	fout << "\n";
	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Static initializers for class and field info      //\n";
	fout << "///////////////////////////////////////////////////////\n";

	fout << "		static Finfo* fieldArray_[];\n";
	fout << "		static const Cinfo cinfo_;\n";
	fout << "};\n";
	fout << "#endif // _" << className_ << "Wrapper_h\n";
}

void Mpp::printWrapperCpp(const string& name) 
{
	string funcname = name + "Wrapper.cpp";
	ofstream fout( funcname.c_str(), ios::out );

	unsigned long i;

	for ( i = 0; i < headerText_.size(); i++ )
		fout << headerText_[ i ] << "\n";

	fout << "#include \"header.h\"\n";
//	fout << "typedef double ProcArg;\n";
//	fout << "typedef int  SynInfo;\n";

	for ( i = 0; i < includes_.size(); i++ )
		fout << includes_[ i ] << "\n";

	fout << "#include \"" << className_ << ".h\"\n";
	fout << "#include \"" << className_ << "Wrapper.h\"\n\n";

	fout << "\nFinfo* " << className_ << "Wrapper::fieldArray_[] =\n";
	fout << "{\n";

	unsigned int n;
	string fname;
	string ftype;

	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Field definitions\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( n = 0; n < fieldVec_.size(); n++ )
		fieldVec_[ n ]->printWrapperCpp( className_, fout );

	fout << "///////////////////////////////////////////////////////\n";
	fout << "// MsgSrc definitions\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( n = 0; n < srcVec_.size(); n++ )
		srcVec_[ n ]->printWrapperCpp( className_, fout );

	fout << "///////////////////////////////////////////////////////\n";
	fout << "// MsgDest definitions\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( n = 0; n < destVec_.size(); n++ )
		destVec_[ n ]->printWrapperCpp( className_, fout );

	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Synapse definitions\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( n = 0; n < synapseVec_.size(); n++ )
		synapseVec_[ n ]->printWrapperCpp( className_, fout );

	fout << "///////////////////////////////////////////////////////\n";
	fout << "// Shared definitions\n";
	fout << "///////////////////////////////////////////////////////\n";
	for ( n = 0; n < sharedVec_.size(); n++ )
		sharedVec_[ n ]->printWrapperCpp( className_, fout );

	fout << "};\n\n";

	fout << "const Cinfo " << className_ << "Wrapper::cinfo_(\n";
	fout << "	\"" << className_ << "\",\n";
	fout << "	\"";
	printQuotedText( fout, author_ );
	fout << "	\"" << className_ << ": ";
	printQuotedText( fout, description_ );
	fout << "	\"Neutral\",\n";
	fout << "	" << className_ << "Wrapper::fieldArray_,\n";
	fout << "	sizeof(" << className_ <<
		"Wrapper::fieldArray_)/sizeof(Finfo *),\n";
	fout << "	&" << className_ << "Wrapper::create\n";
	fout << ");\n";

	fout << "\n";
	fout << "///////////////////////////////////////////////////\n";
	fout << "// Field function definitions\n";
	fout << "///////////////////////////////////////////////////\n";
	fout << "\n";
	for ( n = 0; n < fieldVec_.size(); n++ )
		fieldVec_[ n ]->printWrapperCppFuncs( className_, fout );

	fout << "\n";
	fout << "///////////////////////////////////////////////////\n";
	fout << "// Dest function definitions\n";
	fout << "///////////////////////////////////////////////////\n";
	fout << "\n";
	for ( n = 0; n < destVec_.size(); n++ )
		destVec_[ n ]->printWrapperCppFuncs( className_, fout );

	if ( synapseVec_.size() > 0 ) {
		fout << "///////////////////////////////////////////////////\n";
		fout << "// Synapse function definitions\n";
		fout << "///////////////////////////////////////////////////\n";
		for ( n = 0; n < synapseVec_.size(); n++ )
			synapseVec_[ n ]->printWrapperCppFuncs( className_, fout );
	}

	if ( connVec_.size() > 0 ) {
		fout << "///////////////////////////////////////////////////\n";
		fout << "// Connection function definitions\n";
		fout << "///////////////////////////////////////////////////\n";
		for ( n = 0; n < connVec_.size(); n++ )
			connVec_[ n ]->printWrapperCppFuncs( className_, fout );
	}

	if ( wrapperCpp_.size() > 0 ) {
		fout << "///////////////////////////////////////////////////\n";
		fout << "// Other function definitions\n";
		fout << "///////////////////////////////////////////////////\n";
		for ( n = 0; n < wrapperCpp_.size(); n++ )
			fout << wrapperCpp_[n] << "\n";
	}
}

void Mpp::addChar(const char c)
{
	if (c == '\n') {
		if (index_ > 0) {
			line_[index_] = '\0';
			string s(line_);
			if (!is_whitespace(s))
				original_.push_back(s);
			index_ = 0;
		}
	} else {
		line_[index_++] = c;
	}
}

void ASSERT( bool assertion, const string& name )
{
	static unsigned long assertNum = 0;
	assertNum++;
	if ( assertion ) {
		cout << ".";
	} else {
		cout << "\nASSERT " << assertNum << " failed for " <<
			name << "\n";
		exit( 0 );
	}
}

void runTest()
{
	Mpp mh;
	cout << "Testing Dest parsing: ";
	vector< string >& srcString = mh.msgsrcString_;
	vector< string >& destString = mh.msgdestString_;

	// Test 1: simple message destination
	destString.push_back( "single test1( double V) {" );
	destString.push_back( "x = y ;");
	destString.push_back( "}");
	Dest::parse( mh.destVec_, mh.connVec_, mh.srcVec_, destString );
	cout << "1";
	ASSERT( mh.destVec_.size() == 1, "test1: creation");
	ASSERT( mh.destVec_[0]->type_ == "single", "test1: type");
	ASSERT( mh.destVec_[0]->name_ == "test1", "test1: naming");
	ASSERT( mh.destVec_[0]->connName_ == "test1InConn", "test1: conn");
	ASSERT( mh.destVec_[0]->argtypes_.size() == 1, "test1: nargs");
	ASSERT( mh.destVec_[0]->argnames_.size() == 1, "test1: nargs");
	ASSERT( mh.destVec_[0]->argtypes_[0] == "double", "test1: argtype");
	ASSERT( mh.destVec_[0]->argnames_[0] == "V", "test1: argname");
	ASSERT( mh.destVec_[0]->lines_.size() == 1, "test1: nlines");
	ASSERT( mh.destVec_[0]->lines_[0] == "x = y ;", "test1: line");

	// Test 2: 2 arg message destination
	destString.resize( 0 );
	destString.push_back( "multi test2( int x, long y ) {" );
	destString.push_back( "x += y;");
	destString.push_back( "y = y * 3;");
	destString.push_back( "}");
	Dest::parse( mh.destVec_, mh.connVec_, mh.srcVec_, destString );
	cout << "2";
	ASSERT( mh.destVec_.size() == 2, "test2: creation");
	ASSERT( mh.destVec_[1]->type_ == "multi", "test2: type");
	ASSERT( mh.destVec_[1]->name_ == "test2", "test2: naming");
	ASSERT( mh.destVec_[1]->connName_ == "test2InConn", "test2: conn");
	ASSERT( mh.destVec_[1]->argtypes_.size() == 2, "test2: nargs");
	ASSERT( mh.destVec_[1]->argnames_.size() == 2, "test2: nargs");
	ASSERT( mh.destVec_[1]->argtypes_[0] == "int", "test2: argtype0");
	ASSERT( mh.destVec_[1]->argnames_[0] == "x", "test2: argname0");
	ASSERT( mh.destVec_[1]->argtypes_[1] == "long", "test2: argtype1");
	ASSERT( mh.destVec_[1]->argnames_[1] == "y", "test2: argname1");
	ASSERT( mh.destVec_[1]->lines_.size() == 2, "test2: nlines");
	ASSERT( mh.destVec_[1]->lines_[0] == "x += y;", "test2: line");

	// Test 3: 0 arg message destination with .cpp function define
	destString.resize( 0 );
	destString.push_back( "multi test3();" );
	Dest::parse( mh.destVec_, mh.connVec_, mh.srcVec_, destString );
	cout << "3";
	ASSERT( mh.destVec_.size() == 3, "test3: creation");
	ASSERT( mh.destVec_[2]->type_ == "multi", "test3: type");
	ASSERT( mh.destVec_[2]->name_ == "test3", "test3: naming");
	ASSERT( mh.destVec_[2]->connName_ == "test3InConn", "test3: conn");
	ASSERT( mh.destVec_[2]->argtypes_.size() == 0, "test3: nargs");
	ASSERT( mh.destVec_[2]->argnames_.size() == 0, "test3: nargs");
	ASSERT( mh.destVec_[2]->lines_.size() == 0, "test3: nlines");

	// Test 4: 1 arg message destination with 3 outgoing msgs.
	srcString.resize( 0 );
	srcString.push_back( "single out1( double V, double x);" );
	srcString.push_back( "single out2( double V);" );
	srcString.push_back( "single out3( double V, double y);" );
	destString.resize( 0 );
	destString.push_back( "single test4( double V) {" );
	destString.push_back( "out1( V, x );");
	destString.push_back( "out2( V );");
	destString.push_back( "out3( V + 1, y );");
	destString.push_back( "}");
	Src::parse( mh.srcVec_, mh.connVec_, srcString );
	Dest::parse( mh.destVec_, mh.connVec_, mh.srcVec_, destString );
	cout << "4";
	ASSERT( mh.destVec_.size() == 4, "test4: creation");
	ASSERT( mh.destVec_[3]->type_ == "single", "test4: type");
	ASSERT( mh.destVec_[3]->name_ == "test4", "test4: naming");
	ASSERT( mh.destVec_[3]->connName_ == "test4InConn", "test4: conn");
	ASSERT( mh.destVec_[3]->argtypes_.size() == 1, "test4: nargs");
	ASSERT( mh.destVec_[3]->argnames_.size() == 1, "test4: nargs");
	ASSERT( mh.destVec_[3]->argtypes_[0] == "double", "test4: argtype");
	ASSERT( mh.destVec_[3]->argnames_[0] == "V", "test4: argname");
	ASSERT( mh.destVec_[3]->lines_.size() == 3, "test4: nlines");
	ASSERT( mh.destVec_[3]->destFuncLines_.size() == 3, "test4: ndestFuncLines");
	ASSERT( mh.destVec_[3]->destFuncLines_[0] == 
		"			out1Src_.send( V, x );",
		"test4: destFuncLine0");
	ASSERT( mh.destVec_[3]->destFuncLines_[1] == 
		"			out2Src_.send( V );",
		"test4: destFuncLine1");
	ASSERT( mh.destVec_[3]->destFuncLines_[2] == 
		"			out3Src_.send( V + 1, y );",
		"test4: destFuncLine2");

	// Test 5: 2 arg message destination with .cpp function define
	// Seems there is a problem with defining 2 such funcs sequentially
	destString.resize( 0 );
	destString.push_back( "single test5( string a, string b );" );
	destString.push_back( "single test5a( string a, string b );" );
	Dest::parse( mh.destVec_, mh.connVec_, mh.srcVec_, destString );
	cout << "5";
	ASSERT( mh.destVec_.size() == 6, "test5: creation");
	ASSERT( mh.destVec_[4]->type_ == "single", "test5: type");
	ASSERT( mh.destVec_[4]->name_ == "test5", "test5: naming");
	ASSERT( mh.destVec_[5]->type_ == "single", "test5: type");
	ASSERT( mh.destVec_[5]->name_ == "test5a", "test5: naming");
	ASSERT( mh.destVec_[4]->argtypes_.size() == 2, "test5: nargs");
	ASSERT( mh.destVec_[4]->argnames_.size() == 2, "test5: nargs");
	ASSERT( mh.destVec_[5]->argtypes_.size() == 2, "test5: nargs");
	ASSERT( mh.destVec_[5]->argnames_.size() == 2, "test5: nargs");
	ASSERT( mh.destVec_[4]->lines_.size() == 0, "test5: nlines");
	ASSERT( mh.destVec_[5]->lines_.size() == 0, "test5: nlines");

	// Test 6: Message destination with if statement in it.
	destString.resize( 0 );
	destString.push_back( "multi test6( int x ) {" );
	destString.push_back( "if ( x > 10 )");
	destString.push_back( "out1( V, x );");
	destString.push_back( "else");
	destString.push_back( "out2( V );");
	destString.push_back( "}");
	Dest::parse( mh.destVec_, mh.connVec_, mh.srcVec_, destString );
	cout << "\n	6";
	ASSERT( mh.destVec_.size() == 7, "test6: creation");
	ASSERT( mh.destVec_[6]->type_ == "multi", "test6: type");
	ASSERT( mh.destVec_[6]->name_ == "test6", "test6: naming");
	ASSERT( mh.destVec_[6]->argtypes_.size() == 1, "test6: naming");
	ASSERT( mh.destVec_[6]->argnames_.size() == 1, "test6: naming");
	ASSERT( mh.destVec_[6]->lines_.size() == 4, "test6: nlines");
	ASSERT( mh.destVec_[6]->destFuncLines_.size() == 4, "test6: ndestFuncLines");
	ASSERT( mh.destVec_[6]->destFuncLines_[0] == 
		"if ( x > 10 )",
		"test6: destFuncLine0");
	ASSERT( mh.destVec_[6]->destFuncLines_[1] == 
		"			out1Src_.send( V, x );",
		"test6: destFuncLine1");
	ASSERT( mh.destVec_[6]->destFuncLines_[2] == 
		"else",
		"test6: destFuncLine2");
	ASSERT( mh.destVec_[6]->destFuncLines_[3] == 
		"			out2Src_.send( V );",
		"test6: destFuncLine3");

	// Still need to do: Internal Msgs
	cout << "\nPassed Dest parsing\n";
// const string checkForVector( const string& temp, 
// 	const string& line, unsigned int& j )
	cout << "Testing template argument parsing: ";
	string line = "single solve( vector< double >* y, vector < int > & yPrime );";
	string temp = "vector<";
	unsigned int j = line.find( "double" );
	temp = checkForVector( temp, line, j );
	ASSERT( temp == "vector<double >* " && j == line.find( "y" ),
		"test1: Single template parsing" );
	temp = "vector";
	j = line.find( "< int" );
	temp = checkForVector( temp, line, j );
	ASSERT( temp == "vector< int > & " && j == line.find( "yPrime" ),
		"test2: Second template argument parsing" );

	line = "single solve( map < double, int* >* y );";
	j = line.find( "< double" );
	temp = "map";
	temp = checkForVector( temp, line, j );
	ASSERT( temp == "map< double, int* >* " && j == line.find( "y"),
		"test3: map template parsing" );

	line = "single solve( map< string, vector< int >* >* y );";
	j = line.find( "string" );
	temp = "map<";
	temp = checkForVector( temp, line, j );
	ASSERT( temp == "map<string, vector< int >* >* " && j == line.find( "y" ),
		"test4: map nested template parsing" );

	// Test 4: 1 arg message destination with 3 outgoing msgs.
	srcString.resize( 0 );
	srcString.push_back( "single solve( vector< double >* y, vector< double >* yPrime );" );
	destString.resize( 0 );
	destString.push_back( "single solve( vector< double >* y, double t, double dt );" );
	Dest::parse( mh.destVec_, mh.connVec_, mh.srcVec_, destString );
	ASSERT( mh.destVec_.size() == 8, "test5: MsgDest generation");
	ASSERT( mh.destVec_[7]->type_ == "single", "test5: type");
	ASSERT( mh.destVec_[7]->name_ == "solve", "test5: naming");
	ASSERT( mh.destVec_[7]->connName_ == "solveInConn", "test4: conn");
	ASSERT( mh.destVec_[7]->argtypes_.size() == 3, "test5: nargs");
	ASSERT( mh.destVec_[7]->argnames_.size() == 3, "test5: nargs");
	ASSERT( mh.destVec_[7]->argtypes_[0] == "vector< double >* ", "test5: argtype0");
	ASSERT( mh.destVec_[7]->argnames_[0] == "y", "test5: argname0");
	ASSERT( mh.destVec_[7]->argtypes_[1] == "double", "test5: argtype1");
	ASSERT( mh.destVec_[7]->argnames_[1] == "t", "test5: argname1");
	ASSERT( mh.destVec_[7]->argtypes_[2] == "double", "test5: argtype2");
	ASSERT( mh.destVec_[7]->argnames_[2] == "dt", "test5: argname2");

	cout << "\nPassed Template argument parsing\n";
}
