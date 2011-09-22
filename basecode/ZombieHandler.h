/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_HANDLER_H
#define _ZOMBIE_HANDLER_H

/**
 * This class wraps a DataHandler pointer of a solver, to be used by
 * a Zombie class. It uses zero space per zombie entry. Unlike the generic
 * DataHandlerWrapper, it assumes that the Zombie has a single entry,
 * and the Zombie has an arbitrary number of entries. The parent solver
 * data handler is never modified, though the data contents may be.
 */
class ZombieHandler: public DataHandler
{
	public:
		ZombieHandler( const DataHandler* parentHandler,
			unsigned int start = 0, unsigned int end = 1 );

		~ZombieHandler();

		DataHandler* globalize() const;
		DataHandler* unGlobalize() const;
		bool innerNodeBalance( unsigned int size,
			unsigned int myNode, unsigned int numNodes );

		/**
		 * Make a single identity copy, doing appropriate node 
		 * partitioning if toGlobal is false.
		 */
		DataHandler* copy( bool toGlobal ) const;

		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo) const;

		DataHandler* copyExpand( unsigned int copySize, bool toGlobal ) const;

		DataHandler* copyToNewDim( unsigned int newDimSize, bool toGlobal ) const;

		char* data( DataId index ) const;

		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		/**
		 * Returns the number of data entries.
		 */
		unsigned int totalEntries() const;

		/**
		 * Returns the number of data entries on local node
		 */
		unsigned int localEntries() const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const;

		unsigned int sizeOfDim( unsigned int dim ) const;

		bool resize( vector< unsigned int > dims );

		vector< unsigned int > dims() const;

		bool isDataHere( DataId index ) const;

		virtual bool isAllocated() const;

		bool isGlobal() const;

		iterator begin() const;

		iterator end() const;

		/**
		 * Assigns a block of data at the specified location.
		 * Returns true if all OK. No allocation.
		 */
		bool setDataBlock( const char* data, unsigned int numData,
			const vector< unsigned int >& startIndex ) const;
		bool setDataBlock( const char* data, unsigned int numData,
			DataId startIndex ) const;

		void nextIndex( DataId& index, unsigned int& linearIndex ) const;

	private:
		const DataHandler* parent_;
		unsigned int start_;
		unsigned int end_;
};

#endif // _ZOMBIE_HANDLER_H
