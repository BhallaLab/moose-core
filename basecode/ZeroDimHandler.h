/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZERO_DIM_HANDLER_H
#define _ZERO_DIM_HANDLER_H

/**
 * This class manages the data part of Elements having just one
 * data entry.
 */
class ZeroDimHandler: public DataHandler
{
	public:
		ZeroDimHandler( const DinfoBase* dinfo )
			: DataHandler( dinfo ), data_( 0 )
		{;}

		~ZeroDimHandler();

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const {
			return data_;
		}

		/**
		 * Returns the data at one level up of indexing.
		 * Here there isn't any.
		 */
		char* data1( DataId index ) const {
			return data_;
		}

		/**
		 * Returns the number of data entries.
		 */
		unsigned int numData() const {
			return 1;
		}

		/**
		 * Returns the number of data entries at index 1.
		 */
		unsigned int numData1() const {
			return 1;
		}

		/**
		 * Returns the number of data entries at index 2, if present.
		 * For regular Elements and 1-D arrays this is always 1.
		 */
		 unsigned int numData2( unsigned int index1 ) const {
		 	return 1;
		 }

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const {
			return 0;
		}

		/**
		 * Assign # of entries in dimension 1. 
		 * Ignore here
		 */
		void setNumData1( unsigned int size ) {
			;
		}
		/**
		 * Assigns the sizes of all array field entries at once.
		 * This is ignored for regular Elements.
		 */
		void setNumData2( const vector< unsigned int >& sizes ) {
			;
		}


		/**
		 * Looks up the sizes of all array field entries at once.
		 * Ignored in this case, as there are none.
		 */
		void getNumData2( vector< unsigned int >& sizes ) const {
			;
		}

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const;

		virtual bool isAllocated() const;

		void allocate();

	protected:

	private:
		char* data_;
};

#endif // _ZERO_DIM_HANDLER_H
