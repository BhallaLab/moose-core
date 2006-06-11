/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Dest
{
	friend void runTest();
	public:
		Dest( const string& type,
			const string& name, 
			const string& connName, 
			const string& argstr,
			bool sharesConn
			)
			:	type_( type ), 
				name_( name ), 
				connName_( connName ), 
				argstr_ ( argstr ),
				sharesConn_( sharesConn )
			{
				;
			}
		static void parse( vector< Dest* >&, 
			vector< Conn* >& connVec,
			vector< Src* >& srcVec,
			vector< string >& );
		// void printHeader( const string& className, ofstream& fout );
		void printPrivateWrapperH(
			const string& className, ofstream& fout );
		void printWrapperH( const string& className, ofstream& fout );
		void printWrapperCpp( const string& className, ofstream& fout );
		void printWrapperCppFuncs( 
			const string& className, ofstream& fout );
		void printConstructor( ofstream& fout );

		void addLine( const string& s ) {
			lines_.push_back( s );
		}
		void internalParse( vector< Src* >& srcVec ) ;
		
		const string& name() const {
			return name_;
		}

		bool matchSrc( const string& name ) const {
			return (
				find(
					targetnames_.begin(), targetnames_.end(), name
				)
				!= targetnames_.end()
			);
		}

		void addInternalMsg( const string& s ) {
			internalMsgs_.push_back( s );
		}

	private:
		string type_;
		string name_;
		string connName_;
		string argstr_;
		bool sharesConn_;
		// vector< string >& srclist_;
		vector< string > targetnames_;
		vector< string > destFuncLines_;
		vector< string > lines_;
		vector< string > argtypes_;
		vector< string > argnames_;
		vector< string > internalMsgs_;
};
