/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ONE_DIM_HANDLER_H
#define _ONE_DIM_HANDLER_H

/**
 * This class manages the data part of Elements. It handles a one-
 * dimensional array.
 */
class OneDimHandler: public DataHandler
{
	public:
		OneDimHandler( const DinfoBase* dinfo );

		~OneDimHandler();

		/**
		 * Copies contents into a 2-D array.
		 * This fails if the copy is global, and the simulation is 
		 * multinode, as we don't know how to get the other data.
		 * This works through a hack if the copy is not global. The best
		 * partitioning is not possible, so it uses the existing node
		 * partitioning and just scales up by n.
		 */
		DataHandler* copy( unsigned int n, bool toGlobal ) const;

		/**
		 * calls process on data, using threading info from the ProcInfo,
		 * and internal info about node decomposition.
		 */
		void process( const ProcInfo* p, Element* e ) const;

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const;

		/**
		 * Returns the data at one level up of indexing. In this case it
		 * just returns the data on the specified index.
		 */
		char* data1( DataId index ) const;

		/**
		 * Returns the number of data entries.
		 */
		unsigned int numData() const {
			return size_;
		}

		/**
		 * Returns the number of data entries at index 1.
		 * For regular Elements this is identical to numData.
		 */
		unsigned int numData1() const {
			return size_;
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
			return 1;
		}

		/**
		 * Assigns the size for the data dimension.
		 */
		void setNumData1( unsigned int size );

		/**
		 * Assigns the sizes of all array field entries at once.
		 * Ignore in this case, as there are none.
		 */
		void setNumData2( unsigned int start, 
			const vector< unsigned int >& sizes );

		/**
		 * Looks up the sizes of all array field entries at once.
		 * Ignore in this case, as there are no array fields.
		 */
		unsigned int getNumData2( vector< unsigned int >& sizes ) const;

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const;

		bool isAllocated() const;

		void allocate();

		bool isGlobal() const
		{
			return 0;
		}

		iterator begin() const {
			return start_;
		}

		iterator end() const {
			return end_;
		}

		/**
		 * This is relevant only for the 2 D cases like
		 * FieldDataHandlers.
		 */
		unsigned int startDim2index() const {
			return 0;
		}

		void setData( char* data, unsigned int numData );

	private:
		char* data_;
		unsigned int size_;	// Number of data entries in the whole array
		unsigned int start_;	// Starting index of data, used in MPI.
		unsigned int end_;	// Starting index of data, used in MPI.
};

#endif	// _ONE_DIM_HANDLER_H
