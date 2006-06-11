/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Shared
{
	public:
		Shared( const string& type, const string& name,
			vector< string >& args )
			:	type_( type ), 
				name_( name ), 
				args_ ( args )
			{
				;
			}
		static void parse( vector< Shared* >&, vector< Conn* >& connVec,
			vector< string >& lines);
		/*
		void printPrivateWrapperH(
			const string& className, ofstream& fout );
		void printWrapperH( const string& className, ofstream& fout );
		void printConstructorWrapperH( ofstream& fout );
		*/
		void printWrapperCpp( const string& className, ofstream& fout );
		/*
		void printWrapperCppFuncs( 
			const string& className, ofstream& fout );
		void printConstructor( ofstream& fout );
		*/

		const string& name() const {
			return name_;
		}
	private:
		string type_;
		string name_;
		vector< string > args_;
};
