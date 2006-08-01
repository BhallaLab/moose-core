
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class EvalField
{
	public:
		EvalField( const string& type, const string& name,
			FieldMode mode )
			:	type_( type ), 
				name_( name ), 
				mode_ ( mode )
			{
				;
			}
		static void parse( vector< EvalField* >&, vector< string >& );
		const string& name() {
			return name_;
		}
		void printWrapperH( const string& className, ofstream& fout );
		void printWrapperCpp( const string& className, ofstream& fout );
		void printWrapperCppFuncs( 
			const string& className, ofstream& fout );
		void addToGetFunc( const string& s ) {
			getFunc_.push_back( s );
		}
		void addToSetFunc( const string& s ) {
			setFunc_.push_back( s );
		}
	private:
		string type_;
		string name_;
		bool hasInitVal_;
		string initVal_;
		FieldMode mode_;
		vector< string > getFunc_;
		vector< string > setFunc_;
		bool isVector_;
		unsigned long size_;
		static const unsigned long MAX_VECTOR;
};
