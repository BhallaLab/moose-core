/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
// #include "stdlib.h"
#include "NestFtype.h"
#include "NestFinfo.h"

/**
 * This function has to match either the name itself, or if there
 * are indices in the name, to also match the array index.
 * This is a little messy because it builds the indirection syntax
 * into the base code.
 * The match function recurses all the way down to the final field
 * definition, whether it is Value, Src or Dest finfo.
 * On the way it builds up a stack of inderection pairs, and at the
 * terminal point it saves the appropriate finfo as the origFinfo
 * of the DynamicFinfo.
 */
const Finfo* NestFinfo::match(
				Element* e, const string& s ) const 
{
	std::string::size_type openpos = name().length();
	if ( s.substr( 0, openpos ) != name() )
		return 0;

	vector< IndirectType > v;
	return parseName( v, s );
}

//
// Kchan->X_A->xmin
//
// Kchan is the element name
// X_A is the NestFinfo name for a gate.
// It traverses further down to get other fields.

const Finfo* NestFinfo::parseName( vector< IndirectType >& v,
				const string& path ) const 
{
	if ( path == name() ) {
		return this;
		/// \todo return a dynamic Finfo set up for this field alone.
		/*
		DynamicFinfo* ret = 
			new DynamicFinfo( name, 
		return ret;
		*/
	}
	v.push_back( IndirectType( ptrFunc_, 0 ) );

 	if ( path.find( name() ) == 0 ) {
		std::string::size_type pos = path.find( "[" );
		unsigned long length = name().length();
		if ( maxIndex_ != 0 && pos == name().length() ) {
			std::string::size_type endpos = path.find( "]" );
			if ( endpos == std::string::npos ) {
				cout << "PtrFinfo::match: no closing ]: " <<
						path << endl;
				return 0;
			}
			int index = atoi( path.substr( pos+1, endpos - pos - 1 ).c_str() );
			if ( index <= 0 ) {
				cout << "Error: PtrFinfo::match: Negative index in: " <<
						path << endl;
				return 0;
			}
			length = 1 + endpos;
		}

		//  foo->bar or foo[23]->bar
		if ( path.length() > length + 2 ) {
			if ( path.substr( length, 2 ) == "->" ) {
				string nest_name = path.substr( name().length() + 2 );
				// recurse.
				// return findFinfo( e, path.substr( length + 2 ) );
				// return parseName( v, path.substr( length + 2 ) );
				const Finfo* f = nestClass_->findFinfo( nest_name );
				if ( f ) {
					DynamicFinfo* df = new DynamicFinfo(
						path, f, v );
					return df;
				}
			}
		}
	}

 	if ( path.find( name() ) == 0 &&
		path.substr( name().length(), 2 ) == "->" ) {
		string nest_name = path.substr( name().length() + 2 );
		const Finfo* f = nestClass_->findFinfo( nest_name );
		if ( f ) {
			DynamicFinfo* df = new DynamicFinfo(
				path, f, v );
			return df;
		}
	}
	return 0;
}

bool NestFinfo::inherit( const Finfo* baseFinfo )
{
	return ( ftype()->isSameType( baseFinfo->ftype() ) );
}

#ifdef DO_UNIT_TESTS

#include "../builtins/Interpol.h"

/**
 * A test class for NestFinfo. We will nest the Interpol class.
 */
class NestTestClass
{
	public:
		NestTestClass()
				: dval( 8.321 )
		{
				;
		}

		static void* getPol1( void* data, unsigned int index ) {
			return &(static_cast< NestTestClass* >( data )->pol1 );
		}

		static void* getPol2( void* data, unsigned int index ) {
			return &(static_cast< NestTestClass* >( data )->pol2 );
		}

		static void setDval( const Conn& c, double val ) {
				static_cast< NestTestClass* >( c.data() )->
						dval = val;
		}
		static double getDval( const Element* e ) {
			return static_cast< NestTestClass* >( e->data() )->
					dval;
		}

	private:
		Interpol pol1;
		Interpol pol2;
		// Interpol manyPol[10]; // Later check out indexing.
		double dval;
};

void nestFinfoTest()
{
	cout << "\nTesting NestFinfo";

	static Finfo* testNestFinfos[] = 
	{
		new ValueFinfo( "dval", ValueFtype1< double >::global(), 
			GFCAST( &NestTestClass::getDval ),
			RFCAST( &NestTestClass::setDval )
		),
		new NestFinfo( "pol1", initInterpolCinfo(), 
			&NestTestClass::getPol1 ),
		new NestFinfo( "pol2", initInterpolCinfo(), 
			&NestTestClass::getPol2 ),
	};

	Cinfo nestTestClass( "nestTestClass", "Upi",
					"NestFinfo Test class",
					initNeutralCinfo(),
					testNestFinfos,
					sizeof( testNestFinfos ) / sizeof( Finfo* ),
					ValueFtype1< NestTestClass >::global() );

	Element* n1 = nestTestClass.create( "n1" );
	double dret = 0.0;
	int iret = 0;

	get< double >( n1, "dval", dret );
	ASSERT( dret == 8.321, "test get1");

	set< double >( n1, "dval", 72.1 );
	get< double >( n1, "dval", dret );
	ASSERT( dret == 72.1, "test get2");

	set< int >( n1, "pol1->xdivs", 17 );
	get< int >( n1, "pol1->xdivs", iret );
	ASSERT( iret == 17, "test nest xdivs");

	set< double >( n1, "pol1->xmin", 0 );
	get< double >( n1, "pol1->xmin", dret );
	ASSERT( dret == 0, "test nest xmin");

	set< double >( n1, "pol1->xmax", 17 );
	get< double >( n1, "pol1->xmax", dret );
	ASSERT( dret == 0, "test nest xmin");

	unsigned int k;
	for ( k = 0; k <= 17; k++ ) {
		bool setOK = 
			lookupSet< double, unsigned int >( n1, "table", k * k, k );
		assert( setOK );
	}

	for ( k = 0; k <= 17; k++ ) {
		bool getOK = 
			lookupGet< double, unsigned int >( n1, "table", dret, k );
		assert( getOK );
		assert( dret == k * k );
	}
	ASSERT( 1 , "Test set and get usig nestFinfo" );
}

#endif // DO_UNIT_TESTS
