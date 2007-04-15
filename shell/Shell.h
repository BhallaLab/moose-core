/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SHELL_H
#define _SHELL_H

extern void separateString( const string& s, vector< string>& v, 
				const string& separator );

class Shell
{
	public:
		Shell();
	
////////////////////////////////////////////////////////////////////
// Some utility functions
////////////////////////////////////////////////////////////////////
		string expandPath( const std::string& path ) const;
		static string eid2path( unsigned int eid );
		static unsigned int path2eid( const string& path,
						const string& separator );
		unsigned int innerPath2eid( const string& path,
						const string& separator ) const;
		static unsigned int parent( unsigned int eid );
		static unsigned int traversePath( 
				unsigned int start, vector< string >& );
		static string head( const string& path, const string& separator );
		static string tail( const string& path, const string& separator );
////////////////////////////////////////////////////////////////////
// Local functions for implementing basic GENESIS/MOOSE command set.
////////////////////////////////////////////////////////////////////

		static void setCwe( const Conn&, unsigned int id );
		static unsigned int getCwe( const Element* );
		static void trigCwe( const Conn& );

		static void trigLe( const Conn&, unsigned int parent );

		unsigned int create( const string& type, const string& name,
						unsigned int parent );
		void destroy( unsigned int victim );

		static void staticCreate( const Conn&, string type,
						string name, unsigned int parent );
		static void staticDestroy( const Conn&, unsigned int victim );

		static void getField( const Conn& c, unsigned int id,
						string field );
		static void setField( const Conn& c, 
						unsigned int id, string field, string value );

		static void setClock( const Conn& c, int clockNo, double dt,
				int stage );
		static void useClock( const Conn& c,
			unsigned int tickId,
			vector< unsigned int > path, string function );

		static void getWildcardList( const Conn& c,
						string path, bool ordered );

		static void listMessages( const Conn& c,
				unsigned int id, string field, bool isIncoming );

		static void copy( const Conn& c,
				unsigned int src, unsigned int parent, string name );
		static void move( const Conn& c,
				unsigned int src, unsigned int parent, string name );

		//////////////////////////////////////////////////////////
		// Some stuff for managing scheduling and simulation runs
		//////////////////////////////////////////////////////////
		static void resched( const Conn& c );
		static void reinit( const Conn& c );
		static void stop( const Conn& c );
		static void step( const Conn& c, double time );
		static void requestClocks( const Conn& c );

		//////////////////////////////////////////////////////////
		// Major input functions.
		//////////////////////////////////////////////////////////
		static void readCell( 
					const Conn& c, string filename, string cellpath);

		//////////////////////////////////////////////////////////
		// Channel setup functions.
		//////////////////////////////////////////////////////////
		static void setupAlpha( const Conn& c, unsigned int gate,
				vector< double > parms );
		static void setupTau( const Conn& c, unsigned int gate,
				vector< double > parms );
		static void tweakAlpha( const Conn& c, unsigned int gateId );
		static void tweakTau( const Conn& c, unsigned int gateId );
			/*
		void add( const string& src, const string& dest );
		void drop( const string& src, const string& dest );
		void set( const string& field, const string& value );
		void move( const string& src, const string& dest );
		void copy( const string& src, const string& dest );
		void copyShallow( 
			const string& src, const string& dest );
		void copyHalo( const string& src, const string& dest );
		*/
		/*
		void listCommands( );
		void listClasses( );
		void echo( vector< string >& s, int options );
		void remoteCommand( string arglist );
		void command( int argc, const char** argv );
		*/

	private:
		/// Current working element
		unsigned int cwe_;
		/// Most recently created element
		unsigned int recentElement_;
		vector< unsigned int > workingElementStack_;
		// True if prompts etc are to be printed.
		int isInteractive_;
		string parser_;
};

#endif // _SHELL_H
