#ifndef _Shell_h
#define _Shell_h

class Element; 	// Forward declaration
class Field;	// forward declaration

unsigned int parseArgs( const string& in, vector< string >& out );

class SimDumpInfo {
	public:
		SimDumpInfo( const string& oldObject, const string& newObject,
			const string& oldFields, const string& newFields);

		// Takes info from simobjdump
		void setFieldSequence( int argc, const char** argv );

		// Sets the fields from the simundump arg list.
		bool setFields( Element* e, int argc, const char** argv);

		string oldObject() {
			return oldObject_;
		}

		string newObject() {
			return newObject_;
		}
		

	private:
		string oldObject_;
		string newObject_;
		map< string, string > fields_;
		vector< string >fieldSequence_;
};

class Shell
{
	friend class ShellWrapper;
	public:
		Shell( Element* wrapper );

		virtual ~Shell() {
		}

////////////////////////////////////////////////////////////////////
// Local functions for implementing basic GENESIS/MOOSE command set.
////////////////////////////////////////////////////////////////////

		virtual void addFuncLocal( const string& src, const string& dest );
		void dropFuncLocal( const string& src, const string& dest );
		void setFuncLocal( const string& field, const string& value );
		void createFuncLocal( const string& type, const string& path );
		void deleteFuncLocal( const string& path );
		void moveFuncLocal( const string& src, const string& dest );
		void copyFuncLocal( const string& src, const string& dest );
		void copyShallowFuncLocal( 
			const string& src, const string& dest );
		void copyHaloFuncLocal( const string& src, const string& dest );
		void ceFuncLocal( const string& newpath );
		void pusheFuncLocal( const string& newpath );
		void popeFuncLocal(  );
		void aliasFuncLocal( 
			const string& origfunc, const string& newfunc );
		void quitFuncLocal(  );

		void stopFuncLocal(  );
		void resetFuncLocal(  );
		void stepFuncLocal( 
			const string& steptime, const string& option );
		void setClockFuncLocal( const string& clockNo,
			const string& dt, const string& stage );
		void showClocksFuncLocal();
		void useClockFuncLocal(
			const string& path, const string& clockNo );

		void callFuncLocal( const string& args );
		string getFuncLocal( const string& field );
		string getmsgFuncLocal( 
			const string& field, const string& options );
		int isaFuncLocal( const string& type, const string& field );
		int existsFuncLocal( const string& fieldstr );
		void showFuncLocal( const string& field );
		void showmsgFuncLocal( const string& field );
		void showobjectFuncLocal( const string& classname );
		void pweFuncLocal(  );
		void leFuncLocal( const string& start );
		void listCommandsFuncLocal( );
		void listClassesFuncLocal( );
		void echoFuncLocal( vector< string >& s, int options );
		void commandFuncLocal( int argc, const char** argv );

////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////

		bool splitFieldString( const string& field,
			string& e, string& f );
		bool splitField( const string& fieldstr, Field& f );
		int wildcardField( const string& fieldstr, vector< Field >& f );
		Element* findElement( const string& path );
		Element* checkWorkingElement( );
		Element* shellRelativeFind( const string& path );
		Element* findDest( const string& dest, string& destChildName );

		void simobjdumpFunc( int argc, const char** argv );
		void simundumpFunc( int argc, const char** argv );
		void loadtabFunc( int argc, const char** argv );
		void readcellFunc( int argc, const char** argv );
		void setupAlphaFunc( int argc, const char** argv, bool isTau );
		void tweakFunc( int argc, const char** argv, bool isTau );
		void tabCreateFunc( int argc, const char** argv );
		void tabFillFunc( int argc, const char** argv );
		void addFieldFunc( int argc, const char** argv );

		void ok() {
			if ( isInteractive_ )
				cout << "OK\n";
		}
		void error( const string& report ) {
			if ( isInteractive_ )
				cout << "Error: " << report << "\n";
		}
		void error( const string& s1, const string& s2 ) {
			if ( isInteractive_ )
				cout << "Error: " << s1 << " " << s2 << "\n";
		}

	private:
		string workingElement_;
		vector< string > commandHistory_;
		vector< string > workingElementStack_;
		// True if prompts etc are to be printed.
		int isInteractive_;
		// Temporary for holding text that is sent to the shell for
		// output. Eventually to be directed to tty or graphics.
		string response_;
		string parser_;
		Element* wrapper_;
		Element* recentElement_; // Used in the ^ shorthand
		map< string, SimDumpInfo* > dumpConverter_;
		map< string, string > aliasMap_;
};

#endif // _Shell_h
