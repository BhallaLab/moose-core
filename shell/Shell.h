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

class Shell
{
	public:
		Shell();
		string expandPath( const std::string& path ) const;
		string eid2path( unsigned int eid ) const;
		unsigned int path2eid( const string& path,
						const string& separator ) const;
		unsigned int parent( unsigned int eid ) const;
		unsigned int traversePath( 
				unsigned int start, vector< string >& ) const;
////////////////////////////////////////////////////////////////////
// Local functions for implementing basic GENESIS/MOOSE command set.
////////////////////////////////////////////////////////////////////
		unsigned int create( const string& type, const string& name,
						unsigned int parent );
		void destroy( unsigned int victim );
		void ce( unsigned int eid );
		unsigned int cwe() const {
			return cwe_;
		}

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
		void pushe( const string& newpath );
		void pope(  );
		void quit(  );
		void stop(  );
		void reset(  );
		void step( 
			const string& steptime, const string& option );
		void setClock( const string& clockNo,
			const string& dt, const string& stage );
		void showClocks();
		void useClock(
			const string& path, const string& clockNo );

		void call( const string& args );
		string get( const string& field );
		string getmsg( 
			const string& field, const string& options );
		int isa( const string& type, const string& field );
		int exists( const string& fieldstr );
		void show( const string& field );
		void showmsg( const string& field );
		void showobject( const string& classname );
		*/
		void pwe( ) const;
		void le( unsigned int eid );
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
		vector< unsigned int > workingElementStack_;
		// True if prompts etc are to be printed.
		int isInteractive_;
		string parser_;
};

#endif // _SHELL_H
