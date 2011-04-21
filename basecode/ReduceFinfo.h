/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _REDUCE_FINFO_H
#define _REDUCE_FINFO_H

class ReduceFinfoBase: public SrcFinfo1< FuncId >
{
	public:
		~ReduceFinfoBase()
		{;}

		ReduceFinfoBase( const string& name, const string& doc )
			: SrcFinfo1< FuncId >( name, doc )
		{;}
		virtual ReduceBase* makeReduce( Element* e, DataId i, 
			const OpFunc* f ) const = 0;
		virtual void digestReduce( const Eref& er, const ReduceBase* r ) 
			const = 0;
};

template < class T, class F, class R > class ReduceFinfo: public ReduceFinfoBase
{
	public:
		~ReduceFinfo() {
			;
		}

		ReduceFinfo( const string& name, const string& doc, 
			void ( T::*digestFunc )( const Eref& er, const R* arg ) )
			
			: 	ReduceFinfoBase( name, doc ),
				digestFunc_( digestFunc )
		{
			;
			/*
				string trigname = "trig_" + name;
				trig_ = new SrcFinfo0( 
					trigname,
					"Trigger Reduce call"
				);
			*/
		}

		ReduceBase* makeReduce( Element* e, DataId i, 
			const OpFunc* f ) const
		{
			// Some type checking here for the return type of the func
			const GetOpFuncBase< F >* gof = 
				dynamic_cast< const GetOpFuncBase< F >* >( f );
			// I think this is safe to assert, since the typecheck
			// should have been done at the message setup stage.
			/*
			if ( !gof ) {
				cout << "ReduceFinfo::makeReduce: Error: Type mismatch\n";
				return 0;
			}
			*/
			assert( gof );
			const Eref er( e, i );
			R* ret = new R( er, this, gof );
			return ret;
		}

		void digestReduce( const Eref& er, const ReduceBase* r ) const {
			const R* reduce = dynamic_cast< const R* >( r );
			assert( reduce );
			 
			T* obj = reinterpret_cast< T* >( er.data() );

			(obj->*digestFunc_)( er, reduce );
		}

	private:
		// SrcFinfo* trig_;
		void ( T::*digestFunc_ )( const Eref& er, const R* arg );
};

#endif // _REDUCE_FINFO_H
