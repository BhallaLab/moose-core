/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Synapse
{
	public:
		Synapse( const string& type, const string& name,
			vector< string >& argtypes, vector< string >& argnames,
			string infoType )
			:	type_( type ), 
				name_( name ), 
				argtypes_( argtypes ),
				argnames_( argnames ),
				infoType_( infoType )
			{
				;
			}
		static void parse( vector< Synapse* >&, vector< string >& );
		void printWrapperH( const string& className, ofstream& fout );
		void printPrivateWrapperH( ofstream& fout );
		void printWrapperCpp( const string& className, ofstream& fout );
		void printWrapperCppFuncs( 
			const string& className, ofstream& fout );
		void printConstructor( ofstream& fout );
	private:
		string type_;
		string name_;
		vector< string > argtypes_;
		vector< string > argnames_;
		string infoType_;
		vector< string > internalMsgs_;
};
