
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Field
{
	public:
		Field( const string& type, const string& name,
			FieldMode mode, bool isVector, unsigned long size, 
			string initVal = "\a")
			:	type_( type ), 
				name_( name ), 
				mode_ ( mode ),
				isVector_( isVector ),
				size_( size )
			{
				if (initVal == "\a") {
					initVal_ = "";
					hasInitVal_ = 0;
				} else {
					initVal_ = initVal;
					hasInitVal_ = 1;
				}
			}
		static void parse( vector< Field* >&, vector< string >& );
		static Field* parseSimpleField(
			const string& s, const string& type, FieldMode mode, int j);
		static Field* parseClassField(
			vector< string >::iterator& i,
		    vector< string >::iterator end,
			FieldMode mode,
			unsigned int j);
		static Field* parseMooseClassField(
			const string& s, FieldMode mode, int j);
		void printPrivateHeader( ofstream& fout );
		void printWrapperH( const string& className, ofstream& fout );
		void printWrapperCpp( const string& className, ofstream& fout );
		void printWrapperCppFuncs( 
			const string& className, ofstream& fout );
		void printConstructor( ofstream& fout );
	private:
		string type_;
		string name_;
		bool hasInitVal_;
		string initVal_;
		FieldMode mode_;
		bool isVector_;
		unsigned long size_;
		static const unsigned long MAX_VECTOR;
};
