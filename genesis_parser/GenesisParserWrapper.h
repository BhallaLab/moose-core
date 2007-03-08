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

		static void readlineFunc( const Conn& c, string s );

		static void processFunc( const Conn& c );

		static void parseFunc( const Conn& c, string s );

		static void setReturnId( const Conn& c, unsigned int i );

		static void recvCwe( const Conn& c, unsigned int i );
		static void recvLe( const Conn& c, vector< Id > elist );
		static void recvCreate( const Conn& c, unsigned int i );

//////////////////////////////////////////////////////////////////
// Helper functions
//////////////////////////////////////////////////////////////////
		void doLe( int argc, const char** argv, Id s );
		void doPwe( int argc, const char** argv, Id s );

// Surely this should be a local parser function
/*
		static void aliasFunc( const Conn& c, string alias, string old); {
			static_cast<GenesisParserWrapper *>( c->parent() )->
				alias( alias, old );
		}
*/

// This should also be a local parser function
/*
		static void listCommandsFunc( Conn* c ) {
			static_cast<GenesisParserWrapper *>( c->parent() )->
				listCommands( );
		}
*/

		////////////////////////////////////////////////
		//  Utility functions
		////////////////////////////////////////////////
		// void plainCommand( int argc, const char** argv );
		// char* returnCommand( int argc, const char** argv );
		
		// This utility function follows the message to the 
		// shell, if necessary to get cwe, and gets the id
		// for the specified path.
		static Id path2eid( const string& path, Id i );
		Id innerPath2eid( const string& path, Id g );
		static string eid2path( unsigned int i );
		static Element* getShell( Id g );

		/**
		 * This utility function directs output either to cout, or
		 * to a local buffer which can be inspected for unit tests.
		 */
		void print( const string& s );

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
		unsigned int returnId_;
		Id cwe_;
		Id createdElm_;
		vector< Id > elist_;

		/**
		 * This flag is true if unit tests are being done.
		 */
		bool testFlag_;
		/**
		 * This string holds parser output during unit tests
		 */
		string printbuf_;
};
#endif // _GENESIS_PARSER_WRAPPER_H
