/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _EPFUNC_H
#define _EPFUNC_H
/**
 * This set of classes is derived from OpFunc, and take extra args
 * for the qinfo and Eref.
 */

template< class T > class EpFunc0: public OpFunc
{
	public:
		EpFunc0( void ( T::*func )( Eref& e, const Qinfo* q ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo0* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet0* >( s );
		}

		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q ); 
		}

	private:
		void ( T::*func_ )( Eref& e, const Qinfo* q ); 
};

template< class T, class A > class EpFunc1: public OpFunc
{
	public:
		EpFunc1( void ( T::*func )( Eref& e, const Qinfo* q, A ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return dynamic_cast< const SrcFinfo1< A >* >( s );
		}

		bool checkSet( const SetGet* s ) const {
			return dynamic_cast< const SetGet1< A >* >( s );
		}

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			A val;
			Conv< A >::buf2val( val, buf + sizeof( Qinfo ) );
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q, val ) ;
		}

	private:
		void ( T::*func_ )( Eref& e, const Qinfo* q, A ); 
};

template< class T > class RetFunc: public OpFunc
{
	public:
		RetFunc( void ( T::*func )( Eref& e, const Qinfo* q, const char* arg ) )
			: func_( func )
			{;}

		bool checkFinfo( const Finfo* s ) const {
			return 1;
		}

		bool checkSet( const SetGet* s ) const {
			return 1;
		}

		// This could do with a whole lot of optimization to avoid
		// copying data back and forth.
		void op( Eref e, const char* buf ) const {
			const Qinfo* q = reinterpret_cast< const Qinfo* >( buf );
			(reinterpret_cast< T* >( e.data() )->*func_)( e, q, buf + sizeof( Qinfo ) ) ;
		}

	private:
		void ( T::*func_ )( Eref& e, const Qinfo* q, const char* arg ); 
};

#endif //_EPFUNC_H
