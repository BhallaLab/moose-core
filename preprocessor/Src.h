/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Src
{
	public:
		Src( const string& type, const string& name,
			const string& connName,
			vector< string >& argtypes,
			bool sharesConn )
			:	type_( type ), 
				name_( name ), 
				connName_( connName ),
				sharesConn_( sharesConn ),
				argtypes_ ( argtypes )
			{
				;
			}
		static void parse( vector< Src* >&, vector< Conn* >& connVec,
			vector< string >& );
		// void printHeader( const string& className, ofstream& fout );
		void printPrivateWrapperH(
			const string& className, ofstream& fout );
		void printWrapperH( const string& className, ofstream& fout );
		void printConstructorWrapperH( ofstream& fout );
		void printWrapperCpp( const string& className, ofstream& fout );
		void printWrapperCppFuncs( 
			const string& className, ofstream& fout );
		void printConstructor( ofstream& fout );

		const string& name() const {
			return name_;
		}

		void addInternalMsg( const string& s ) {
			internalMsgs_.push_back( s );
		}
	private:
		string type_;
		string name_;
		string connName_;
		bool sharesConn_;
		vector< string > argtypes_;
		vector< string > internalMsgs_;
};
