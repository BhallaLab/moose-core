/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef REDUCE_MAX_H
#define REDUCE_MAX_H

class ReduceFinfoBase;

/**
 * Reduces to find maximum. The argument type is templated.
 */
template< class T > class ReduceMax: public ReduceBase
{
	public:
		// The function is set up by a suitable SetGet templated wrapper.
		ReduceMax( Id srcId, const ReduceFinfoBase* rfb,
			const GetOpFuncBase< T >* gof )
			:
				ReduceBase( srcId, rfb ),
				max_( 0 ),
				gof_( gof )
		{;}

		~ReduceMax()
		{;}

		void primaryReduce( Id tgtId )
		{
			T x = gof_->reduceOp( tgtId.eref() );
			if ( max_ < x ) 
				max_ = x;
		}
		
		// Must not use other::func_
		void secondaryReduce( const ReduceBase* other )
		{
			const ReduceMax< T >* r = 
				dynamic_cast< const ReduceMax< T >* >( other );
			assert( r );
			if ( max_ < r->max_ ) 
				max_ = r->max_;
		}
		
		void tertiaryReduce( const char* data )
		{
			T x = *reinterpret_cast< const T* >( data );
			if ( max_ < x ) 
				max_ = x;
		}

		const char* data() const
		{
			return reinterpret_cast< const char* >( &max_ );
		}

		unsigned int dataSize() const
		{
			return sizeof( T );
		}

		/// Max of data values, the output of this class.
		T max() const
		{
			return max_;
		}

	private:
		T max_; /// max of data values. 

		/// OpFunc that contains function to extract data value from eref.
		const GetOpFuncBase< T >* gof_;
};

#endif // REDUCE_MAX_H
