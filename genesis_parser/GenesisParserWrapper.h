/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _GENESIS_PARSER_WRAPPER_H
#define _GENESIS_PARSER_WRAPPER_H

class GenesisParserWrapper: public myFlexLexer
{
    public:
		GenesisParserWrapper();

		static void readlineFunc( const Conn* c, string s );

		static void processFunc( const Conn* c );

		static void parseFunc( const Conn* c, string s );

		static void setReturnId( const Conn* c, Id i );

		static void recvCwe( const Conn* c, Id i );
		static void recvElist( const Conn* c, vector< Id > elist);
		static void recvCreate( const Conn* c, Id i );
		static void recvField( const Conn* c, string value );
		static void recvWildcardList( const Conn* c,
						vector< Id > value );
		static void recvClocks( const Conn* c, vector< double > dbls);
		static void recvMessageList( 
				const Conn* c, vector< Id > elist, string s);

//////////////////////////////////////////////////////////////////
// Helper functions
//////////////////////////////////////////////////////////////////
		void doLe( int argc, const char** argv, Id s );
		// void doPwe( int argc, const char** argv, Id s );
		void printCwe();
		char* doGet( int argc, const char** argv, Id s );
		void doSet( int argc, const char** argv, Id s );
		void doShow( int argc, const char** argv, Id s );
		void doShowMsg( int argc, const char** argv, Id s );
		void showAllFields( Id e, Id s );
		void doAdd( int argc, const char** const argv, Id s );
		bool innerAdd( Id s, Id src, const string& srcF, Id dest,
						const string& destF );
		void useClock( Id tickId, const string& path,
						const string& func, Id s );
		void step( int argc, const char** const argv );
		void showClocks( Element* e );

		bool tabCreate( int argc, const char** argv, Id s );
                bool channelTabCreate(Id s, Id elmId, char gate, string xdivs, string xmin, string xmax, string ydivs="", string ymin="", string ymax="");
		char** elementList( const string& path, Id s);
		bool fieldExists( Id eid, const string& field, Id s );
		string getFieldValue();

		string handleMultGate( int argc, const char** const argv, Id s,
			string& gate, double& gatePower );

		void getReadcellGlobals( vector< double >& globalParms );

		////////////////////////////////////////////////
		//  Utility functions
		////////////////////////////////////////////////
		// void plainCommand( int argc, const char** argv );
		// char* returnCommand( int argc, const char** argv );
		
		// This utility function follows the message to the 
		// shell, if necessary to get cwe, and gets the id
		// for the specified path.
//		static Id path2eid( const string& path, Id i );
//		Id innerPath2eid( const string& path, Id g );
//		static string eid2path( Id i );
//		static Element* getShell( Id g );

		/**
		 * This utility function directs output either to cout, or
		 * to a local buffer which can be inspected for unit tests.
		 */
		void print( const string& s, bool noNewLine = 0 );

#ifdef DO_UNIT_TESTS
		/**
		 * This function performs unit tests for the GenesisParser
		 */
		void unitTest();

		/**
		 * This is used in the unit tests and asserts that the
		 * specified command string gives rise to the specified
		 * return string.
		 */
		void gpAssert( const string& command, const string& ret );
#endif

    private:
		void loadBuiltinCommands();
		string returnCommandValue_;

		/**
		 * \todo Should use Erefs instead of Ids.
		 */
		Id returnId_;
		Id cwe_;
		Id createdElm_;
		vector< Id > elist_;
		vector< double > dbls_;

		/**
		 * This flag is true if unit tests are being done.
		 */
		bool testFlag_;
		/**
		 * This string holds parser output during unit tests
		 */
		string printbuf_;
		string fieldValue_;
};
#endif // _GENESIS_PARSER_WRAPPER_H
