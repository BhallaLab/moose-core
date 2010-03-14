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
		SrcFinfo( const string& name, const string& doc, 
			BindIndex b );

		~SrcFinfo() {;}

		void registerOpFuncs(
			map< string, FuncId >& fnames, vector< OpFunc* >& funcs );

		BindIndex registerBindIndex( BindIndex current );

		BindIndex getBindIndex() const {
			return bindIndex_;
		}

		/**
		 * Checks that the target will work for this Msg.
		 */
		bool checkTarget( const Finfo* target ) const;

		/**
		 * First checks that the target will work, then adds the Msg.
		 */
		bool addMsg( const Finfo* target, MsgId mid, Id src ) const;

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

		SrcFinfo0( const string& name, const string& doc, BindIndex b );
		~SrcFinfo0() {;}

		// Will need to specialize for strings etc.
		void send( Eref e, const ProcInfo* p ) const;
		void sendTo( Eref e, const ProcInfo* p, const FullId& target) const;

	private:
};


template < class T > class SrcFinfo1: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc, 
			BindIndex b )
			: SrcFinfo( name, doc, b )
			{ ; }

		// Will need to specialize for strings etc.
		void send( Eref e, const ProcInfo* p, const T& arg ) const
		{
			// Qinfo( useSendTo, isForward, eindex, size );
			Qinfo q( 0, 1, e.index(), sizeof( T ) );
			e.element()->asend( q, getBindIndex(), p, 
				reinterpret_cast< const char* >( &arg ) );
		}

		/**
		 * We know the data index but we also need to know the target 
		 * Element or Msg, since there could be multiple ones. 
		 */
		void sendTo( Eref e, const ProcInfo* p,
			const T& arg, const FullId& target ) const
		{
			// Qinfo( useSendTo, isForward, eindex, size );
			Qinfo q( 1, 1, e.index(), sizeof( T ) );
			e.element()->tsend( q, getBindIndex(), p, 
				reinterpret_cast< const char* >( &arg ), target );
		}

	private:
};

template <> class SrcFinfo1< string >: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc, 
			BindIndex b )
			: SrcFinfo( name, doc, b )
			{ ; }

		// Will need to specialize for strings etc.
		void send( Eref e, const ProcInfo* p, const string& arg ) const
		{
			Conv< string > s( arg );

			// Qinfo( useSendTo, isForward, eindex, size );
			Qinfo q( 0, 1, e.index(), s.size() );
			char* buf = new char[ s.size() ];
			s.val2buf( buf );

			e.element()->asend( q, getBindIndex(), p, buf );
			delete[] buf;
		}

		void sendTo( Eref e, const ProcInfo* p, 
			const string& arg, const FullId& target ) const
		{
			Conv< string > s( arg );
			Qinfo q( 1, 1, e.index(), s.size() );
			char* buf = new char[ s.size() ];
			s.val2buf( buf );

			e.element()->tsend( q, getBindIndex(), p, buf, target );
			delete[] buf;
		}

	private:
};

template < class T1, class T2 > class SrcFinfo2: public SrcFinfo
{
	public:
		~SrcFinfo2() {;}

		SrcFinfo2( const string& name, const string& doc, 
			BindIndex b )
			: SrcFinfo( name, doc, b )
			{ ; }

		// This version is general but inefficient as it uses an extra
		// memcpy in val2buf.
		void send( Eref e, const ProcInfo* p,
			const T1& arg1, const T2& arg2 ) {
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Qinfo q( 0, 1, e.index(), a1.size() + a2.size() );
			char temp[ a1.size() + a2.size() ];
			a1.val2buf( temp );
			a2.val2buf( temp + a1.size() );
			e.element()->asend( q, getBindIndex(), p, temp );
		}

		void sendTo( Eref e, const ProcInfo* p,
			const T1& arg1, const T2& arg2,
			const FullId& target ) {
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Qinfo q( 1, 1, e.index(), a1.size() + a2.size() );
			char temp[ a1.size() + a2.size() ];
			a1.val2buf( temp );
			a2.val2buf( temp + a1.size() );
			e.element()->tsend( q, getBindIndex(), p, temp, target );
		}

	private:
};


template < class T1, class T2, class T3 > class SrcFinfo3: public SrcFinfo
{
	public:
		~SrcFinfo3() {;}

		SrcFinfo3( const string& name, const string& doc, 
			BindIndex b )
			: SrcFinfo( name, doc, b )
			{ ; }

		// Will need to specialize for strings etc.
		void send( Eref e, const ProcInfo* p,
			const T1& arg1, const T2& arg2, const T3& arg3 ){
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			Qinfo q( 0, 1, e.index(), a1.size() + a2.size() + a3.size() );
			char temp[ a1.size() + a2.size() + a3.size() ];
			a1.val2buf( temp );
			a2.val2buf( temp + a1.size() );
			a3.val2buf( temp + a1.size() + a2.size() );
			e.element()->asend( q, getBindIndex(), p, temp );
		}

		void sendTo( Eref e, DataId target, const ProcInfo* p,
			const T1& arg1, const T2& arg2, const T3& arg3 )
		{
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			Qinfo q( 1, 1, e.index(), a1.size() + a2.size() + a3.size() );
			char temp[ a1.size() + a2.size() + a3.size() ];
			a1.val2buf( temp );
			a2.val2buf( temp + a1.size() );
			a3.val2buf( temp + a1.size() + a2.size() );
			e.element()->tsend( q, getBindIndex(), p, temp, target );
		}

	private:
};

template < class T1, class T2, class T3, class T4 > class SrcFinfo4: public SrcFinfo
{
	public:
		~SrcFinfo4() {;}

		SrcFinfo4( const string& name, const string& doc, 
			BindIndex b )
			: SrcFinfo( name, doc, b )
			{ ; }

		// Will need to specialize for strings etc.
		void send( Eref e, const ProcInfo* p,
			const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4 ){
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			Conv< T4 > a4( arg4 );
			Qinfo q( 0, 1, e.index(), a1.size() + a2.size() + a3.size() + a4.size() );
			char temp[ a1.size() + a2.size() + a3.size() + a4.size() ];
			a1.val2buf( temp );
			a2.val2buf( temp + a1.size() );
			a3.val2buf( temp + a1.size() + a2.size() );
			a4.val2buf( temp + a1.size() + a2.size() + a3.size() );
			e.element()->asend( q, getBindIndex(), p, temp );
		}

		void sendTo( Eref e, const ProcInfo* p,
			const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4,
			const FullId& target ) {
			Conv< T1 > a1( arg1 );
			Conv< T2 > a2( arg2 );
			Conv< T3 > a3( arg3 );
			Conv< T4 > a4( arg4 );
			Qinfo q( 0, 1, e.index(), a1.size() + a2.size() + a3.size() + a4.size() );
			char temp[ a1.size() + a2.size() + a3.size() + a4.size() ];
			a1.val2buf( temp );
			a2.val2buf( temp + a1.size() );
			a3.val2buf( temp + a1.size() + a2.size() );
			a4.val2buf( temp + a1.size() + a2.size() + a3.size() );
			e.element()->tsend( q, getBindIndex(), p, temp, target );
		}

	private:
};

#endif // _SRC_FINFO_H
