/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SRC_FINFO_H
#define _SRC_FINFO_H

/**
 * This set of classes define Message Sources. Their main job is to supply 
 * a type-safe send operation, and to provide typechecking for it.
 */

class SrcFinfo: public Finfo
{
	public:
		SrcFinfo( const string& name, const string& doc ); 

		~SrcFinfo() {;}

		void registerFinfo( Cinfo* c );

		///////////////////////////////////////////////////////////////

		bool strSet( const Eref& tgt, const string& field, 
			const string& arg ) const {
			return 0; // always fails
		}

		bool strGet( const Eref& tgt, const string& field, 
			string& returnValue ) const {
			return 0; // always fails
		}

		BindIndex getBindIndex() const;
		void setBindIndex( BindIndex b );

		/**
		 * Checks that the target will work for this Msg.
		 */
		bool checkTarget( const Finfo* target ) const;

		/**
		 * First checks that the target will work, then adds the Msg.
		 */
		bool addMsg( const Finfo* target, ObjId mid, Element* src ) const;



		static const BindIndex BadBindIndex;
	private:
		/**
		 * Index into the msgBinding_ vector.
		 */
		unsigned short bindIndex_;
};

/**
 * SrcFinfo0 sets up calls without any arguments.
 */
class SrcFinfo0: public SrcFinfo
{
	public:

		SrcFinfo0( const string& name, const string& doc );
		~SrcFinfo0() {;}
		
		void send( const Eref& e ) const;

		string rttiType() const {
			return "void";
		}

	private:
};

template< class A > class OpFunc1Base;
// Should specialize for major cases like doubles.

template < class T > class SrcFinfo1: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc )
			: SrcFinfo( name, doc )
			{ ; }

		void send( const Eref& e, T arg ) const 
		{
			const vector< MsgDigest >& md = e.msgDigest( getBindIndex() );
			for ( vector< MsgDigest >::const_iterator
				i = md.begin(); i != md.end(); ++i ) {
				const OpFunc1Base< T >* f = 
					dynamic_cast< const OpFunc1Base< T >* >( i->func );
				assert( f );
				for ( vector< Eref >::const_iterator
					j = i->targets.begin(); j != i->targets.end(); ++j ) {
					if ( j->dataIndex() == ALLDATA ) {
						Element* e = j->element();
						unsigned int end = e->numData();
						for ( DataId k = 0; k < end; ++k )
							f->op( Eref( e, k ), arg );
					} else  {
						f->op( *j, arg );
					}
				}
			}
		}

		string rttiType() const {
			return Conv<T>::rttiType();
		}
	private:
};


template< class A1, class A2 > class OpFunc2Base;
// Specialize for doubles.
template < class T1, class T2 > class SrcFinfo2: public SrcFinfo
{
	public:
		~SrcFinfo2() {;}

		SrcFinfo2( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		void send( const Eref& e, const T1& arg1, const T2& arg2 ) const
		{
			const vector< MsgDigest >& md = e.msgDigest( getBindIndex() );
			for ( vector< MsgDigest >::const_iterator
				i = md.begin(); i != md.end(); ++i ) {
				const OpFunc2Base< T1, T2 >* f = 
					dynamic_cast< const OpFunc2Base< T1, T2 >* >( i->func );
				assert( f );
				for ( vector< Eref >::const_iterator
					j = i->targets.begin(); j != i->targets.end(); ++j ) {
						f->op( *j, arg1, arg2 );
				}
			}
		}

		string rttiType() const {
			return Conv<T1>::rttiType() + "," + Conv< T2 >::rttiType();
		}

	private:
};

template < class T1, class T2, class T3 > class SrcFinfo3: public SrcFinfo
{
	public:
		~SrcFinfo3() {;}

		SrcFinfo3( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		// Will need to specialize for strings etc.
		void send( const Eref& e, 
			const T1& arg1, const T2& arg2, const T3& arg3 ) const
		{
		}

		string rttiType() const {
			return Conv<T1>::rttiType() + "," + Conv< T2 >::rttiType() +
				"," + Conv<T3>::rttiType();
		}

	private:
};

template < class T1, class T2, class T3, class T4 > class SrcFinfo4: public SrcFinfo
{
	public:
		~SrcFinfo4() {;}

		SrcFinfo4( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		// Will need to specialize for strings etc.
		void send( const Eref& e,
			const T1& arg1, const T2& arg2, 
			const T3& arg3, const T4& arg4 ) const
		{
		}

		string rttiType() const {
			return Conv<T1>::rttiType() + "," + Conv< T2 >::rttiType() +
				"," + Conv<T3>::rttiType() + "," + Conv< T4 >::rttiType();
		}

	private:
};

template < class T1, class T2, class T3, class T4, class T5 > class SrcFinfo5: public SrcFinfo
{
	public:
		~SrcFinfo5() {;}

		SrcFinfo5( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		// Will need to specialize for strings etc.
		void send( const Eref& e,
			const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4,
			const T5& arg5 ) const
		{
		}

		string rttiType() const {
			return Conv<T1>::rttiType() + "," + Conv< T2 >::rttiType() +
				"," + Conv<T3>::rttiType() + "," + Conv< T4 >::rttiType() +
				"," + Conv<T5>::rttiType();
		}

	private:
};


template < class T1, class T2, class T3, class T4, class T5, class T6 > class SrcFinfo6: public SrcFinfo
{
	public:
		~SrcFinfo6() {;}

		SrcFinfo6( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		void send( const Eref& e,
			const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4,
			const T5& arg5, const T6& arg6 ) const
		{
		}

		string rttiType() const {
			return Conv<T1>::rttiType() + "," + Conv< T2 >::rttiType() +
				"," + Conv<T3>::rttiType() + "," + Conv< T4 >::rttiType() +
				"," + Conv<T5>::rttiType() + "," + Conv< T6 >::rttiType();
		}

	private:
};

#endif // _SRC_FINFO_H
