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
class OneDimHandler: public OneDimGlobalHandler
{
	public:
		OneDimHandler( const DinfoBase* dinfo );

		~OneDimHandler();

		DataHandler* globalize() const;

		DataHandler* unGlobalize() const;

		void assimilateData( const char* data,
			unsigned int begin, unsigned int end );

		virtual bool nodeBalance( unsigned int size );

		DataHandler* copy() const;

		DataHandler* copyExpand( unsigned int copySize ) const;

		DataHandler* copyToNewDim( unsigned int newDimSize ) const;

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const;

		/**
		 * calls process on data, using threading info from the ProcInfo,
		 * and internal info about node decomposition.
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		bool resize( vector< unsigned int > dims );

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const;

		bool isGlobal() const
		{
			return 0;
		}

		iterator begin() const {
			return iterator( this, start_ );
		}

		iterator end() const {
			return iterator( this, end_ );
		}

	protected:
		void setData( char* data, unsigned int numData );

	private:
		unsigned int start_;	// Starting index of data, used in MPI.
		unsigned int end_;	// Starting index of data, used in MPI.
};

#endif	// _ONE_DIM_HANDLER_H
