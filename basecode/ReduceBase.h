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

class ReduceFinfoBase;

class ReduceBase
{
	public:
		ReduceBase();
		ReduceBase( ObjId srcId, const ReduceFinfoBase* rfb );
		virtual ~ReduceBase();

		/**
		 * Reduces the original data type, typically a double. To do this
		 * in a general way it just looks up the object in this
		 * interface function.
		 */
		virtual void primaryReduce( ObjId id ) = 0;

		/**
		 * Reduces contents of other (identical) Reduce subclasses.
		 */
		virtual void secondaryReduce( const ReduceBase* other ) = 0;

		/**
		 * Reduces contents of internal data form expressed as char*.
		 * Used for internode data reduction.
		 */
		virtual void tertiaryReduce( const char* data ) = 0;

		/**
		 * Expresses internal data in a char* form, used as to transfer
		 * data between nodes for internode reduction.
		 */
		virtual const char* data() const = 0;

		/**
		 * Size of internal data, in bytes.
		 */
		virtual unsigned int dataSize() const = 0;

		/**
		 * Reports whether we are on the same Eref
		 */
		bool sameEref( const ReduceBase* other ) const;

		/**
		 * Collects data from all nodes to get the simulation-wide
		 * reduced value.
		 * Returns true if the object is global, or if the object was on
		 * current node. In this situation we would expect to do the 
		 * assignResult step too.
		 * There is a problem with this: It doesn't know what to do about
		 * global fields, that is ElementFields.
		 */
		virtual bool reduceNodes();

		/**
		 * Assigns the completed calculation to the object that requested 
		 * it, using the ReduceFinfoBase::digestReduce function.
		 */
		void assignResult() const;
	private:
		/// This is the Object calling the Reduce operation.
		ObjId srcId_;

		/**
		 * This is the Finfo handling the Reduce operation. 
		 * Contains the digest call.
		 */
		const ReduceFinfoBase* rfb_;
};

/**
 * Reduces a double to its stats: mean, sdev, and # of counts.
 */
class ReduceStats: public ReduceBase
{
	public:
		// The function is set up by a suitable SetGet templated wrapper.
		ReduceStats( ObjId srcId, const ReduceFinfoBase* rfb,
			const GetOpFuncBase< double >* gof );
		~ReduceStats();

		void primaryReduce( ObjId id );
		
		// Must not use other::func_
		void secondaryReduce( const ReduceBase* other );

		
		void tertiaryReduce( const char* data );

		const char* data() const;

		unsigned int dataSize() const;

		/// Sum of data values, used for mean and sdev.
		double sum() const;

		/// Sum of squares of data values. Used for sdev.
		double sumsq() const;

		/// Number of data values reduced.
		unsigned int count() const;

	private:
		/**
		 * This little internal structure is the essential reduction data
		 * and is all that is transferred between nodes when doing internode
		 * data reduction.
		 */
		struct ReduceDataType {
			double sum_; /// Sum of data values. Used for mean and sdev.
			double sumsq_; /// Sum of squares of data values. Used for sdev
			unsigned int count_; /// Number of data values reduced.
		} data_;

		/// OpFunc that contains function to extract data value from eref.
		const GetOpFuncBase< double >* gof_;
};

/**
 * Special Reduce class to handle fieldDimension reductions.
 * It finds the max, but also grabs the dest object during primaryReduce
 * so it can later be used to assign the fieldDimension.
 */
class ReduceFieldDimension: public ReduceBase
{
	public:
		// The function is set up by a suitable SetGet templated wrapper.
		ReduceFieldDimension( ObjId srcId, const ReduceFinfoBase* rfb,
			const GetOpFuncBase< unsigned int >* gof );
		~ReduceFieldDimension();

		void primaryReduce( ObjId id );
		
		// Must not use other::func_
		void secondaryReduce( const ReduceBase* other );

		void tertiaryReduce( const char* data );

		const char* data() const;

		unsigned int dataSize() const;

		unsigned int maxIndex() const;

		bool reduceNodes();

	private:
		unsigned int maxIndex_;
		ObjId tgtId_;

		/// OpFunc that contains function to extract data value from eref.
		const GetOpFuncBase< unsigned int >* gof_;
};

#endif // REDUCE_BASE_H
