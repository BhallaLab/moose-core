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
		bool addMsg( const Finfo* target, MsgId mid, Element* src ) const;



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
		
		void send( const Eref& e, ThreadId threadNum ) const;
//		void sendTo( const Eref& e, const ProcInfo* p, const ObjId& target) const;
		/// Returns compiler-independent string with type info
		void fastSend( const Eref& e, ThreadId threadNum ) const;

		string rttiType() const {
			return "void";
		}

	private:
};


// Should specialize for major cases like doubles.
template < class T > class SrcFinfo1: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc )
			: SrcFinfo( name, doc )
			{ ; }

		// Will need to specialize for strings etc.
		void send( const Eref& e, ThreadId threadNum, const T& arg ) const 
		{
			Conv< T > a( arg );
			Qinfo::addToQ( e.objId(), getBindIndex(), threadNum,
				a.ptr(), a.size());
		}

		string rttiType() const {
			return Conv<T>::rttiType();
		}
	private:
};

template <> class SrcFinfo1< double >: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		void send( const Eref& e, ThreadId threadNum, double arg ) const
		{
			Qinfo::addToQ( e.objId(), getBindIndex(), 
				threadNum, &arg, 1 );
		}

		string rttiType() const {
			return Conv< double >::rttiType();
		}

	private:
};

template <> class SrcFinfo1< unsigned int >: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		void send( const Eref& e, ThreadId threadNum, unsigned int arg ) const
		{
			double temp = arg;
			Qinfo::addToQ( e.objId(), getBindIndex(), threadNum,
				&temp, 1 );
		}

		string rttiType() const {
			return Conv< unsigned int >::rttiType();
		}

	private:
};

template <> class SrcFinfo1< int >: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		void send( const Eref& e, ThreadId threadNum, int arg ) const
		{
			double temp = arg;
			Qinfo::addToQ( e.objId(), getBindIndex(), threadNum,
				&temp, 1 );
		}

		string rttiType() const {
			return Conv< int >::rttiType();
		}

	private:
};

/*
template <> class SrcFinfo1< string >: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		// Will need to specialize for strings etc.
		void send( const Eref& e, const ProcInfo* p, const string& arg ) const
		{
			Conv< string > s( arg );

			// Qinfo( eindex, size, useSendTo );
			Qinfo q( e.index(), s.size(), 0 );
			char* buf = new char[ s.size() ];
			s.val2buf( buf );

			e.element()->asend( q, getBindIndex(), p, buf );
			delete[] buf;
		}

		void sendTo( const Eref& e, const ProcInfo* p, 
			const string& arg, const ObjId& target ) const
		{
			Conv< string > s( arg );
			Qinfo q( e.index(), s.size(), 1 );
			char* buf = new char[ s.size() ];
			s.val2buf( buf );

			e.element()->tsend( q, getBindIndex(), p, buf, target );
			delete[] buf;
		}

		string rttiType() const {
			return Conv< string >::rttiType();
		}

	private:
};
*/

// Specialize for doubles.
template < class T1, class T2 > class SrcFinfo2: public SrcFinfo
{
	public:
		~SrcFinfo2() {;}

		SrcFinfo2( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		void send( const Eref& e, ThreadId threadNum,
			const T1& arg1, const T2& arg2 ) const
		{
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Qinfo::addToQ( e.objId(), getBindIndex(), 
				threadNum,
				a1.ptr(), a1.size(),
				a2.ptr(), a2.size() );
		}

		string rttiType() const {
			return Conv<T1>::rttiType() + "," + Conv< T2 >::rttiType();
		}

	private:
};

// Specialize for doubles.
template <> class SrcFinfo2< double, double >: public SrcFinfo
{
	public:
		~SrcFinfo2() {;}

		SrcFinfo2( const string& name, const string& doc ) 
			: SrcFinfo( name, doc )
			{ ; }

		void send( const Eref& e, ThreadId threadNum,
			double arg1, double arg2 ) const
		{
			Qinfo::addToQ( e.objId(), getBindIndex(), 
				threadNum,
				&arg1, 1, &arg2, 1 );
		}

		string rttiType() const {
			return Conv< double >::rttiType() + "," + Conv<  double  >::rttiType();
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
		void send( const Eref& e, ThreadId threadNum,
			const T1& arg1, const T2& arg2, const T3& arg3 ) const
		{
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			unsigned int totSize = a1.size() + a2.size() + a3.size();
			double* data = new double[ totSize ];
			double* ptr = data;
			memcpy( ptr, a1.ptr(), a1.size() * sizeof( double ) );
			ptr += a1.size();
			memcpy( ptr, a2.ptr(), a2.size() * sizeof( double ) );
			ptr += a2.size();
			memcpy( ptr, a3.ptr(), a3.size() * sizeof( double ) );
			
			Qinfo::addToQ( e.objId(), getBindIndex(),
				threadNum, 
				data, totSize );
			delete[] data;
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
		void send( const Eref& e, ThreadId threadNum,
			const T1& arg1, const T2& arg2, 
			const T3& arg3, const T4& arg4 ) const
		{
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			Conv< T4 > a4( arg4 );
			unsigned int totSize = a1.size() + a2.size() + 
				a3.size() + a4.size();
			double* data = new double[ totSize ];
			double* ptr = data;
			memcpy( ptr, a1.ptr(), a1.size() * sizeof( double ) );
			ptr += a1.size();
			memcpy( ptr, a2.ptr(), a2.size() * sizeof( double ) );
			ptr += a2.size();
			memcpy( ptr, a3.ptr(), a3.size() * sizeof( double ) );
			ptr += a3.size();
			memcpy( ptr, a4.ptr(), a4.size() * sizeof( double ) );
			
			Qinfo::addToQ( e.objId(), getBindIndex(),
				threadNum, 
				data, totSize );
			delete[] data;
		}

		void fastSend( const Eref& e, ThreadId threadNum,
			const T1& arg1, const T2& arg2, 
			const T3& arg3, const T4& arg4 ) const
		{
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			Conv< T4 > a4( arg4 );
			unsigned int totSize = a1.size() + a2.size() + 
				a3.size() + a4.size();
			double* data = new double[ totSize ];
			double* ptr = data;
			memcpy( ptr, a1.ptr(), a1.size() * sizeof( double ) );
			ptr += a1.size();
			memcpy( ptr, a2.ptr(), a2.size() * sizeof( double ) );
			ptr += a2.size();
			memcpy( ptr, a3.ptr(), a3.size() * sizeof( double ) );
			ptr += a3.size();
			memcpy( ptr, a4.ptr(), a4.size() * sizeof( double ) );
			
			Qinfo qi( e.objId(), getBindIndex(), threadNum, 0, totSize );
			e.element()->exec( &qi, data );
			delete[] data;
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
		void send( const Eref& e, ThreadId threadNum,
			const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4,
			const T5& arg5 ) const
		{
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			Conv< T4 > a4( arg4 );
			Conv< T5 > a5( arg5 );

			unsigned int totSize = a1.size() + a2.size() + 
				a3.size() + a4.size() + a5.size();
			double* data = new double[ totSize ];
			double* ptr = data;
			memcpy( ptr, a1.ptr(), a1.size() * sizeof( double ) );
			ptr += a1.size();
			memcpy( ptr, a2.ptr(), a2.size() * sizeof( double ) );
			ptr += a2.size();
			memcpy( ptr, a3.ptr(), a3.size() * sizeof( double ) );
			ptr += a3.size();
			memcpy( ptr, a4.ptr(), a4.size() * sizeof( double ) );
			ptr += a4.size();
			memcpy( ptr, a5.ptr(), a5.size() * sizeof( double ) );
			
			Qinfo::addToQ( e.objId(), getBindIndex(),
				threadNum, 
				data, totSize );
			delete[] data;
		}

		void fastSend( const Eref& e, ThreadId threadNum,
			const T1& arg1, const T2& arg2, 
			const T3& arg3, const T4& arg4, 
			const T5& arg5 ) const
		{
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			Conv< T4 > a4( arg4 );
			Conv< T5 > a5( arg5 );
			unsigned int totSize = a1.size() + a2.size() + 
				a3.size() + a4.size() + a5.size();
			double* data = new double[ totSize ];
			double* ptr = data;
			memcpy( ptr, a1.ptr(), a1.size() * sizeof( double ) );
			ptr += a1.size();
			memcpy( ptr, a2.ptr(), a2.size() * sizeof( double ) );
			ptr += a2.size();
			memcpy( ptr, a3.ptr(), a3.size() * sizeof( double ) );
			ptr += a3.size();
			memcpy( ptr, a4.ptr(), a4.size() * sizeof( double ) );
			ptr += a4.size();
			memcpy( ptr, a5.ptr(), a5.size() * sizeof( double ) );
			
			Qinfo qi( e.objId(), getBindIndex(), threadNum, 0, totSize );
			e.element()->exec( &qi, data );
			delete[] data;
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

		// Will need to specialize for strings etc.
		void send( const Eref& e, ThreadId threadNum,
			const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4,
			const T5& arg5, const T6& arg6 ) const
		{
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			Conv< T4 > a4( arg4 );
			Conv< T5 > a5( arg5 );
			Conv< T6 > a6( arg6 );

			unsigned int totSize = a1.size() + a2.size() + 
				a3.size() + a4.size() + a5.size() + a6.size();
			double* data = new double[ totSize ];
			double* ptr = data;
			memcpy( ptr, a1.ptr(), a1.size() * sizeof( double ) );
			ptr += a1.size();
			memcpy( ptr, a2.ptr(), a2.size() * sizeof( double ) );
			ptr += a2.size();
			memcpy( ptr, a3.ptr(), a3.size() * sizeof( double ) );
			ptr += a3.size();
			memcpy( ptr, a4.ptr(), a4.size() * sizeof( double ) );
			ptr += a4.size();
			memcpy( ptr, a5.ptr(), a5.size() * sizeof( double ) );
			ptr += a5.size();
			memcpy( ptr, a6.ptr(), a6.size() * sizeof( double ) );
			
			Qinfo::addToQ( e.objId(), getBindIndex(),
				threadNum, 
				data, totSize );
			delete[] data;
		}

		string rttiType() const {
			return Conv<T1>::rttiType() + "," + Conv< T2 >::rttiType() +
				"," + Conv<T3>::rttiType() + "," + Conv< T4 >::rttiType() +
				"," + Conv<T5>::rttiType() + "," + Conv< T6 >::rttiType();
		}

	private:
};

#endif // _SRC_FINFO_H
