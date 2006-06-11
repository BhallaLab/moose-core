/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Conn
{
	public:
		Conn( const string& type, const string& name )
			:	type_( type ), 
				name_( name )
			{
				;
			}
		static const string& parse( vector< Conn* >&, 
			const string& name, const string& type, 
			const string& direction, bool& sharedConn );
		// void printHeader( const string& className, ofstream& fout );
		void printPrivateWrapperH(
			const string& className, ofstream& fout );
		void printConstructorWrapperH( ofstream& fout );
		void printFriendWrapperH(
			const string& className, ofstream& fout );
		void printWrapperH( const string& className, ofstream& fout );
		void printWrapperCpp( const string& className, ofstream& fout );
		void printWrapperCppFuncs( 
			const string& className, ofstream& fout );
		void printConstructor( ofstream& fout );

		const string& name() const {
			return name_;
		}

		const string& polytype() const {
			static const string single = "single";
			static const string multi = "multi";
			return ( type_ == "UniConn" ) ? single : multi;
		}

		void setShared( vector< string >& shared, const string& stype )
		{
			shared_ = shared;
			sharedType_ = stype;
		}

	private:
		string type_;
		string name_;
		vector< string > shared_;
		string sharedType_;
};
