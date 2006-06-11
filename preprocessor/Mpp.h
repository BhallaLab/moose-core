/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

enum parse_state {START,OK,LINECOMMENT,LONGCOMMENT};

// this is the moose preprocessor. It takes a .mh file and generates
// a .h, a wrapper.h, and a wrapper.cpp file.
const int LINELEN = 1000;

bool is_whitespace(const string& line);

void runTest();

class Mpp
{
	friend void runTest();
	public:
		Mpp() 
		{ 
			index_ = 0;
		}
		~Mpp()
		{ ; }

		void addChar(const char c);
		void addLine(const string& s) {
			original_.push_back( s );
		}
		void addHeader( const string& s ) {
			startstuff_.push_back( s );
		}

		void parse(); // This is the wrapper for all the parse funcs.
		/*
		void parseFields();
		void parseSimpleField( vector< string >::iterator& );
		void parseClassField( vector< string >::iterator& );
		void parseMooseClassField( vector< string >::iterator& );
		void parseSrc();
		void parseDest();
		void parseSynapse();
		void parseShared();
		void parseOrdinaryCpp();
		*/
		void parseInternalMsgs();
		void parseHeader();
		void printHeader(const string& s);
		void printWrapperH(const string& s);
		void printWrapperCpp(const string& s);
		static string copyleft;
	private:
		char line_[LINELEN];
		int index_;
		string className_;
		vector<string> original_;
		// vector<string> tokens;

		vector<string> startstuff_;
		vector<string> endstuff_;
		vector<string> pub_;
		vector<string> priv_;
		vector<string> privWrapper_;
		vector<string> fieldString_;
		vector<string> msgsrcString_;
		vector<string> msgdestString_;
		vector<string> synapseString_;
		vector<string> sharedString_;
		vector<string> author_;
		vector<string> description_;
		vector<string> wrapperCpp_;

		vector<string> headerText_;
		vector<string> includes_;

		vector< Field* > fieldVec_;
		vector< Src* > srcVec_;
		vector< Dest* > destVec_;
		vector< Synapse* > synapseVec_;
		vector< Conn* > connVec_;
		vector< Shared* > sharedVec_;
};

