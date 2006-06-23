/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FTYPE_H
#define _FTYPE_H

////////////////////////////////////////////////////////////////////
// Utility functions for string conversions used in assignments.
////////////////////////////////////////////////////////////////////
// Here we define string conversions. In the template we do
// only the default failure options, below we have 
// specializations for known classes.
template <class T> inline string val2str(T val)
{
			cerr << "Error: val2str conversion not defined\n";
			return "";
}

template <class T> inline T str2val(const string& s)
{
			cerr << "Error: str2val conversion not defined\n";
			return T();
			/*
			T ret;
			return (ret);
			*/
}

// Some template specializations to handle common conversions.

template<> string val2str<string>(string val);
template<> string str2val<string>(const string& s);
template<> string val2str<int>(int val) ;
template<> int str2val<int>(const string& s) ;
template<> string val2str<double>(double val) ;
template<> double str2val<double>(const string& s) ;

////////////////////////////////////////////////////////////////////
// These are the MOOSE type info classes
////////////////////////////////////////////////////////////////////
class Ftype {
	public:
		Ftype()
		{ ; }

		virtual ~Ftype()
		{ ; }

		virtual unsigned int nValues() const = 0;
		virtual bool isSameType( const Ftype* other ) const = 0;
		virtual bool strGet( Element* e, Finfo* f, string& val ) 
			const = 0;
		virtual bool strSet( Element* e, Finfo* f, const string& val )
			const = 0;
		// Can't do this here because RelayFinfo depends on Ftype.
		//virtual Finfo* makeRelayFinfo( Finfo* f, Element* e ) = 0;

		// valueComparisons apply only for Ftype1, so we provide a
		// default.
		virtual bool valueComparison(
			Field& f, const string& op, const string& val ) const {
			return 0;
		}
};

class Ftype0: public Ftype 
{
	public:
		unsigned int nValues() const {
			return 0;
		}

		bool isSameType( const Ftype* other ) const {
			return ( dynamic_cast< const Ftype0* >( other ) != 0 );
		}

		static bool set( Element* e, Finfo* f );

		bool strGet( Element* e, Finfo* f, string& val ) const {
			val = "";
			return 0;
		}

		bool strSet( Element* e, Finfo* f, const string& val ) const {
			return set( e, f );
		}
};


// This template requires that the class T provide == and < operators.
template <class T> class Ftype1: public Ftype
{
	public:
		unsigned int nValues() const {
			return 1;
		}

		bool isSameType( const Ftype* other ) const {
			return ( dynamic_cast< const Ftype1< T >* >( other ) != 0 );
		}

		static bool set( Element* e, Finfo* f, T val ) {
			if ( dynamic_cast< const Ftype1< T >* >( f->ftype() ) ) {
				void ( *func )( Conn*, T ) = 
					reinterpret_cast< void ( * )( Conn*, T ) >(
					f->recvFunc() );
				if ( func ) {
					RelayConn c( e, f );
					func( &c, val );
					return 1;
				}
			}
			return 0;
		}

		static bool set( Element* e, const string& fname, T val ) {
			return set( e, Field( e, fname ).getFinfo(), val );
			// return set( e, e->field( fname ).getFinfo(), val );
		}

		static bool get( Element* e, Finfo* f, T& ret ) {
			 ValueFinfoBase< T >* v =
			 	dynamic_cast< ValueFinfoBase< T >* >( f );
			if ( v ) {
				ret = v->value( e );
				return 1;
			}
			// Possibly it was from an ObjFinfo, in which case the
			// Finfo f is a RelayFinfo1< T >
			// Unfortunately RelayFinfos are declared after Ftype.
			/*
			RelayFinfo1< T >* r = dynamic_cast< RelayFinfo1< T >* >( f);
			if ( r ) {
				return r->get( e, ret );
			}
			*/
			return 0;
		}

		static bool get( Element* e, const string& fname, T& ret ) {
			// Field f = e->field( fname );
			// return get( e, f.operator->(), ret );
			// return get( e, e->field( fname ).getFinfo(), ret );
			return get( e, Field( e, fname ).getFinfo(), ret );
		}

		bool strGet( Element* e, Finfo* f, string& val ) const {
	//		Field temp( f, e );
			T ret;
			if ( get( e, f, ret ) ) {
				val = val2str< T >( ret );
				return 1;
			}
			return 0;
		}

		bool strSet( Element* e, Finfo* f, const string& val ) const {
			// Field temp( f, e );
			// return setField< T >( temp, str2val< T >( val ) );
			return set( e, f, str2val< T >( val ) );
		}
		virtual bool valueComparison( 
			Field& f, const string& op, const string& val ) const {
			T x;
			T y = str2val< T >( val );
			// We really only need to do two comparisons: == and <.
			// We will use this minimal set on the type T for all
			// comparisons.
			if ( get( f.getElement(), f.getFinfo(), x ) ) {
			//cout << "in Ftype::valueComparison, testing: " << x <<
			//	" " << op << " " << y << "\n";
				if ( op == "==" )
					return ( x == y );
				if ( op == "=" ) 
					// I really do mean the = symbol here. For 
					// backward compatibility this is an equality test.
					return ( x == y );
				if ( op == "!=" )
					return ( !( x == y ) );
				if ( op == "<" )
					return ( x < y );
				if ( op == "<=" )
					return ( x == y || x < y );
				if ( op == ">" )
					return ( !( x == y || x < y ) );
				if ( op == ">=" )
					return ( !( x < y ) );
			}
			return 0;
		}
};

/*
// Utility function for assignment. Possibly use later.
template < class T > bool set( Field & f, T val )
{
	return Ftype1< T >::set( f.getElement(), f.operator->(), val );
}
*/

template <class T1, class T2> class Ftype2: public Ftype
{
	public:
		unsigned int nValues() const {
			return 2;
		}

		bool isSameType( const Ftype* other ) const {
			return ( dynamic_cast< const Ftype2< T1, T2 >* >( other )
				!= 0 );
		}

		static bool set( Element* e, Finfo* f, T1 val1, T2 val2 ) {
			if ( dynamic_cast< const Ftype2< T1, T2 >* >( f->ftype()) ){
				void ( *func )( Conn*, T1, T2 ) = 
					reinterpret_cast< void ( * )( Conn*, T1, T2 ) >(
					f->recvFunc() );
				if ( func ) {
					RelayConn c( e, f );
					// SynConn< int > c( e );
					func( &c, val1, val2 );
					return 1;
				}
			}
			return 0;
		}

		static bool set( Element* e, const string& fname, 
			T1 val1, T2 val2 ) {
			// return set( e, e->field( fname ).getFinfo(), val1, val2 );
			return set( e, Field( e, fname ).getFinfo(), val1, val2 );
		}

		bool strGet( Element* e, Finfo* f, string& val ) const {
			val = "";
			return 0;
		}

		bool strSet( Element* e, Finfo* f, const string& val ) const {
			// Field temp( f, e );
			size_t i = val.find_first_of(" 	,");
			string s1;
			if ( i != string::npos && i != 0 )
				s1 = val.substr( 0, i );
			else
				s1 = "";
			string s2 = val.substr( i );
			i = s2.find_first_not_of("  ,");
			if ( i != string::npos )
				s2 = s2.substr( i );
			else
				s2 = "";

			// cout << "strSet('" << s1 << "', '" << s2 << "');\n";
			return set( e, f,
				str2val< T1 >( s1 ), 
				str2val< T2 >( s2 )
			);
		}
};

template <class T1, class T2, class T3 > class Ftype3: public Ftype
{
	public:
		unsigned int nValues() const {
			return 3;
		}

		bool isSameType( const Ftype* other ) const {
			return ( dynamic_cast< const Ftype3< T1, T2, T3 >* >( other)
				!= 0 );
		}

		static bool set( Element* e, Finfo* f, T1 val1, T2 val2, T3 val3 ) {
			if ( dynamic_cast< const Ftype2< T1, T2 >* >( f->ftype()) ){
				void ( *func )( Conn*, T1, T2, T3 ) = 
					reinterpret_cast< void ( * )( Conn*, T1, T2, T3 ) >(
					f->recvFunc() );
				if ( func ) {
					RelayConn c( e, f );
					// SynConn< int > c( e );
					func( &c, val1, val2, val3 );
					return 1;
				}
			}
			return 0;
		}
		static bool set( Element* e, const string& fname, 
			T1 val1, T2 val2, T3 val3 ) {
			// return set( e, e->field( fname ).getFinfo(), 
				// val1, val2, val3 );
			return set( e, Field( e, fname ).getFinfo(), 
				val1, val2, val3 );
		}

		bool strGet( Element* e, Finfo* f, string& val ) const {
			val = "";
			return 0;
		}

		bool strSet( Element* e, Finfo* f, const string& val ) const {
			// Field temp( f, e );
			unsigned long i = val.find_first_of(" 	,");
			string s1;
			if ( i != string::npos && i != 0 )
				s1 = val.substr( 0, i );
			else
				s1 = "";

			string s2 = val.substr( i );
			i = s2.find_first_not_of("  ,");
			if ( i != string::npos )
				s2 = s2.substr( i );
			else
				s2 = "";
			unsigned long j = s2.find_first_of(" 	,");
			string s3 = s2.substr( i );
			s2 = s2.substr( 0, j );

			i = s3.find_first_not_of("  ,");
			if ( i != string::npos )
				s3 = s3.substr( i );
			else
				s3 = "";

			// cout << "strSet('" << s1 << "', '" << s2 << "');\n";
			return set( e, f,
				str2val< T1 >( s1 ), 
				str2val< T2 >( s2 ),
				str2val< T3 >( s3 )
			);
		}
};

class MultiFtype: public Ftype
{
	public:
		MultiFtype( vector< Finfo* >& finfos )
			: finfos_( finfos )
		{ ; }

		unsigned int nValues() const {
			return 0;
		}

		bool isSameType( const Ftype* other ) const;

		bool strGet( Element* e, Finfo* f, string& val ) const {
			val = "";
			return 0;
		}
		bool strSet( Element* e, Finfo* f, const string& val ) const {
			return 0;
		}

	private:
		vector< Finfo* >& finfos_;
};


#endif // _FTYPE_H
