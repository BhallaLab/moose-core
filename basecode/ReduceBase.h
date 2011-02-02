/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef REDUCE_BASE_H
#define REDUCE_BASE_H

class ReduceBase
{
	public:
		ReduceBase();
		virtual ~ReduceBase();

		/**
		 * Reduces the original data type, typically a double. To do this
		 * in a general way it just looks up the object in this
		 * interface function.
		 */
		virtual void primaryReduce( const Eref& e ) = 0;

		/**
		 * Reduces contents of other (identical) Reduce subclasses.
		 */
		virtual void secondaryReduce( const ReduceBase* other ) = 0;
	private:
};

// Here we try it without a type dependence
// The whole class is suitable to be passed around, but the func_
// must be ignored.
class ReduceStats: public ReduceBase
{
	public:
		// The function is set up by a suitable SetGet templated wrapper.
		ReduceStats( const GetOpFuncBase< double >* gof );
		~ReduceStats();

		void primaryReduce( const Eref& e );
		
		// Must not use other::func_
		void secondaryReduce( const ReduceBase* other );

	private:
		double sum_;
		double sumsq_;
		unsigned int count_;
		const GetOpFuncBase< double >* gof_;
		// double (*func_)( const Eref& e );
};
#endif // REDUCE_BASE_H
