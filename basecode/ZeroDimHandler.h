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
class ZeroDimHandler: public ZeroDimGlobalHandler
{
	public:
		ZeroDimHandler( const DinfoBase* dinfo )
			: ZeroDimGlobalHandler( dinfo )
		{;}

		ZeroDimHandler( const ZeroDimHandler* other );

		~ZeroDimHandler();

		DataHandler* globalize() const;

		DataHandler* unGlobalize() const;

		void assimilateData( const char* data,
			unsigned int begin, unsigned int end );

		/**
		 * Determines how to decompose data among nodes for specified size
		 * Returns true if there is a change from the current configuration
		 */
		bool nodeBalance( unsigned int size );

		/**
		 * For copy we won't worry about global status. 
		 * Instead define function: globalize above.
		 * Version 1: Just copy as original
		 */
		DataHandler* copy() const;

		/**
		 * Version 2: Copy same dimensions but different # of entries.
		 * The copySize is the total number of targets, 
		 * here we need to figure out
		 * what belongs on the current node.
		 */
		DataHandler* copyExpand( unsigned int copySize ) const;

		/**
		 * Version 3: Add another dimension when doing the copy.
		 * Here too we figure out what is to be on current node for copy.
		 */
		DataHandler* copyToNewDim( unsigned int newDimSize ) const;


		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const;


		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		bool resize( vector< unsigned int > dims );

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const;

		bool isAllocated() const;

		bool isGlobal() const
		{
			return 0;
		}

		iterator begin() const {
			return iterator( this, 0 );
		}

		iterator end() const {
			return iterator( this, 1 );
		};

	protected:
		void setData( char* data, unsigned int numData ) {
			if ( data_ ) {
				dinfo()->destroyData( data_ );
			}
			data_ = data;
		}

	private:
};

#endif // _ZERO_DIM_HANDLER_H
