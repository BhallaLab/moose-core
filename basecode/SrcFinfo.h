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
			ConnId c );

		~SrcFinfo() {;}

		void registerOpFuncs(
			map< string, FuncId >& fnames, vector< OpFunc* >& funcs );

		unsigned int registerSrcFuncIndex( unsigned int current );

		unsigned int registerConn( unsigned int current );

		ConnId getConnId() const {
			return c_;
		}
		unsigned int getFuncIndex() const {
			return funcIndex_;
		}

	private:
		ConnId c_; /// Predefined ConnId for the outgoing data.

		/// Index into a table with FuncIds, on Elements
		unsigned int funcIndex_; 
};

/**
 * SrcFinfo0 sets up calls without any arguments.
 */
class SrcFinfo0: public SrcFinfo
{
	public:

		SrcFinfo0( const string& name, const string& doc, ConnId c );
		~SrcFinfo0() {;}

		// Will need to specialize for strings etc.
		void send( Eref e ) const;
		void sendTo( Eref e, DataId target ) const;

	private:
};


template < class T > class SrcFinfo1: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc, 
			ConnId c )
			: SrcFinfo( name, doc, c )
			{ ; }

		// Will need to specialize for strings etc.
		void send( Eref e, const T& arg ) const
		{
			e.asend( getConnId(), getFuncIndex(), 
				reinterpret_cast< const char* >( &arg ), sizeof( T ) );
		}

		void sendTo( Eref e, DataId target, const T& arg ) const
		{
			e.tsend( getConnId(), getFuncIndex(), target, 
				reinterpret_cast< T* >( &arg ), sizeof( T ) );

			/*
			char temp[ sizeof( T ) + sizeof( DataId ) ];
			*reinterpret_cast< T* >( temp ) = arg;
			*reinterpret_cast< DataId* >( temp + sizeof( T ) ) = target;
			// e.tsend( c_, funcIndex_, target, reinterpret_cast< const char* >( &arg ), sizeof( T ) );
			e.tsend( getConnId(), getFuncIndex(), 
				target, temp, sizeof( T ) );
				*/
		}

	private:
};

template <> class SrcFinfo1< string >: public SrcFinfo
{
	public:
		~SrcFinfo1() {;}

		SrcFinfo1( const string& name, const string& doc, 
			ConnId c )
			: SrcFinfo( name, doc, c )
			{ ; }

		// Will need to specialize for strings etc.
		void send( Eref e, const string& arg ) const
		{
			e.asend( getConnId(), getFuncIndex(), 
				arg.c_str() , arg.length() + 1 );
		}

		void sendTo( Eref e, DataId target, const string& arg ) const
		{
			e.tsend( getConnId(), getFuncIndex(), target, 
				arg.c_str(), arg.length() + 1 );
			/*


			char* temp = new char[ arg.length() + 1 + sizeof( unsigned int ) ];
			strcpy( temp, arg.c_str() );
			*reinterpret_cast< DataId* >( temp + arg.length() + 1) = 
				target;
			// e.tsend( c_, funcIndex_, target, reinterpret_cast< const char* >( &arg ), arg.length() + 1 );
			e.tsend( getConnId(), getFuncIndex(), 
				target, temp, arg.length() + 1 );
			delete[] temp;
			*/
		}

	private:
};

template < class T1, class T2 > class SrcFinfo2: public SrcFinfo
{
	public:
		~SrcFinfo2() {;}

		SrcFinfo2( const string& name, const string& doc, 
			ConnId c )
			: SrcFinfo( name, doc, c )
			{ ; }

		// Will need to specialize for strings etc.
		void send( Eref e, const T1& arg1, const T2& arg2 ) {
			char temp[ sizeof( T1 ) + sizeof( T2 ) ];
			*reinterpret_cast< T1* >( temp ) = arg1;
			*reinterpret_cast< T2* >( temp + sizeof( T1 ) ) = arg2;
			e.asend( getConnId(), getFuncIndex(), temp,
				sizeof( T1 ) + sizeof( T2 ) );
		}

		void sendTo( Eref e, DataId target, 
			const T1& arg1, const T2& arg2 ) {
			char temp[ sizeof( T1 ) + sizeof( T2 ) + sizeof( unsigned int ) ];
			*reinterpret_cast< T1* >( temp ) = arg1;
			*reinterpret_cast< T2* >( temp + sizeof( T1 ) ) = arg2;
			*reinterpret_cast< DataId* >( temp + sizeof( T1 ) + sizeof( T2 ) ) = target;
			// e.tsend( c_, funcIndex_, target, reinterpret_cast< const char* >( &arg ), sizeof( T1 ) + sizeof( T2 ) );
			e.tsend( getConnId(), getFuncIndex(),
				target, temp, sizeof( T1 ) + sizeof( T2 ) );
		}

	private:
};


template < class T1, class T2, class T3 > class SrcFinfo3: public SrcFinfo
{
	public:
		~SrcFinfo3() {;}

		SrcFinfo3( const string& name, const string& doc, 
			ConnId c )
			: SrcFinfo( name, doc, c )
			{ ; }

		// Will need to specialize for strings etc.
		void send( Eref e, const T1& arg1, const T2& arg2, const T3& arg3 ){
			char temp[ sizeof( T1 ) + sizeof( T2 ) + sizeof( T3 ) ];
			*reinterpret_cast< T1* >( temp ) = arg1;
			*reinterpret_cast< T2* >( temp + sizeof( T1 ) ) = arg2;
			*reinterpret_cast< T3* >( temp + sizeof( T1 ) + sizeof( T2 ) ) = arg3;
			e.asend( getConnId(), getFuncIndex(), temp,
				sizeof( T1 ) + sizeof( T2 ) + sizeof( T3 ) );
		}

		void sendTo( Eref e, DataId target, 
			const T1& arg1, const T2& arg2, const T3& arg3 )
		{
			char temp[ sizeof( T1 ) + sizeof( T2 ) + sizeof( T3 ) +
				sizeof( unsigned int ) ];
			*reinterpret_cast< T1* >( temp ) = arg1;
			*reinterpret_cast< T2* >( temp + sizeof( T1 ) ) = arg2;
			*reinterpret_cast< T3* >( temp + sizeof( T1 ) + sizeof( T2 ) ) = arg3;
			*reinterpret_cast< DataId* >( temp + 
				sizeof( T1 ) + sizeof( T2 ) + sizeof( T3 ) ) = 
				target;
			// e.tsend( c_, funcIndex_, target, reinterpret_cast< const char* >( &arg ), sizeof( T1 ) + sizeof( T2 ) + sizeof( T3 ) );
			e.tsend( getConnId(), getFuncIndex(),
				target, temp, sizeof( T1 ) + sizeof( T2 ) + sizeof( T3 ) );
		}

	private:
};

#endif // _SRC_FINFO_H
