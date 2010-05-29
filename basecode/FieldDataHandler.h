/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FIELD_DATA_HANDLER_H
#define _FIELD_DATA_HANDLER_H

/**
 * This class manages access to array fields X in an array of objects Y.
 * Examples are synapses and clock ticks.
 * Replaces FieldElement.h
 * It is templated by the field type, the parent type and a lookup function
 * that extracts the field from the parent.
 */

template< class Parent, class Field > class FieldDataHandler: public DataHandler
{
	public:
		FieldDataHandler( const DinfoBase* dinfo,
			const DataHandler* parentDataHandler,
			Field* ( Parent::*lookupField )( unsigned int ),
			unsigned int ( Parent::*getNumField )() const,
			void ( Parent::*setNumField )( unsigned int num ) )
			: DataHandler( dinfo ),
				parentDataHandler_( parentDataHandler ),
				lookupField_( lookupField ),
				getNumField_( getNumField ),
				setNumField_( setNumField ),
				start_( 0 )

		{;}

		~FieldDataHandler()
		{;} // Don't delete data because the parent Element should do so.

		/**
		 * This really won't work, as it is just a hook to the parent
		 * Data Handler. Need the duplicated Parent for this.
		 *
		 * If n is 1, just duplicates everything. No problem.
		 * if n > 1, then operation is nasty.
		 * Scales up the data dimension from 0 to 1 if original had 1 entry,
		 * and assigns n to the new dimension. This is a problem on multi
		 * nodes as the original would have been sitting on node 0.
		 * Scales up data dimension from 1 to 2 if original had an array.
		 * 2nd dimension is now n. For multinodes does a hack by scaling
		 * up all entries by n, rather than doing a clean repartitioning.
		 */
		DataHandler* copy( unsigned int n, bool toGlobal ) const
		{
			FieldDataHandler< Parent, Field >* ret =
				new FieldDataHandler< Parent, Field >( *this );
			return ret;
		}

		void process( const ProcInfo* p, Element* e ) const 
		{
			; // Fields don't do independent process?
		}

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const
		{
			char* pa = parentDataHandler_->data( index );
			if ( pa ) {
				Field* s = ( ( reinterpret_cast< Parent* >( pa ) )->*lookupField_ )( index.field() );
				return reinterpret_cast< char* >( s );
			}
			return 0;
		}

		/**
		 * Returns the data at one level up of indexing. In this case it
		 * returns the parent of the field.
		 */
		char* data1( DataId index ) const
		{
			return parentDataHandler_->data1( index );
		}

		/**
		 * Returns the number of field entries.
		 * This runs into trouble on multinodes.
		 * I'll just return # on local node.
		 */
		unsigned int numData() const {
			unsigned int ret = 0;
			for ( DataHandler::iterator i = parentDataHandler_->begin();
				i != parentDataHandler_->end(); ++i ) {
				char* pa = parentDataHandler_->data1( i );
				ret += ( ( reinterpret_cast< Parent* >( pa ) )->*getNumField_ )();
			}

			/*
			unsigned int size = parentDataHandler_->numData1();
			unsigned int start = 
				 ( size * Shell::myNode() ) / Shell::numNodes();
			unsigned int end = 
				 ( size * ( 1 + Shell::myNode() ) ) / Shell::numNodes();

			for ( unsigned int i = start; i < end; ++i ) {
				char* pa = parentDataHandler_->data1( i );
				ret += ( ( reinterpret_cast< Parent* >( pa ) )->*getNumField_ )();
			}
			*/
			return ret;
		}

		/**
		 * Returns the number of data entries in the whole object.
		 * What is least surprising: To get the # of data entries of
		 * the parent (current version) or to go one level nested and
		 * get the # of field entries? In order to do the latter we need
		 * an index, so I think it is out of the question.
		 */
		unsigned int numData1() const {
			return parentDataHandler_->numData1();
		}

		/**
		 * Returns the number of field entries on the data entry indicated
		 * by index1, if present.
		 * e.g., return the # of synapses on a given IntFire
		 */
		unsigned int numData2( unsigned int index1 ) const
		{
			char* pa = parentDataHandler_->data1( index1 );
			if ( pa ) {
				return ( ( reinterpret_cast< Parent* >( pa ) )->*getNumField_ )();
			}
			return 0;
		}

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const {
			return 2;
		}

		/**
		 * Assigns size for first (data) dimension. This usually will not
		 * be called here, but by the parent data Element.
		 */
		void setNumData1( unsigned int size ) {
			cout << Shell::myNode() << ": FieldDataHandler::setNumData1: Error: Cannot set parent data size from Field\n";
		}

		/**
		 * Assigns the sizes of all array field entries at once.
		 */
		void setNumData2( unsigned int start, 
			const vector< unsigned int >& sizes ) {
			assert ( sizes.size() == parentDataHandler_->numData() );
			for ( DataHandler::iterator i = parentDataHandler_->begin();
				i != parentDataHandler_->end(); ++i ) {
				char* pa = parentDataHandler_->data1( i );
				( ( reinterpret_cast< Parent* >( pa ) )->*setNumField_ )( sizes[i] );
			}
			start_ = start;

/*
			unsigned int size = parentDataHandler_->numData1();
			assert( sizes.size() == size );
			unsigned int start = 
				 ( size * Shell::myNode() ) / Shell::numNodes();
			unsigned int end = 
				 ( size * ( 1 + Shell::myNode() ) ) / Shell::numNodes();

			for ( unsigned int i = start; i < end; ++i ) {
				char* pa = parentDataHandler_->data1( i );
				( ( reinterpret_cast< Parent* >( pa ) )->*setNumField_ )( sizes[i] );
			}
			*/
		}

		/**
		 * Looks up the sizes of all array field entries at once.
		 * This is messy for multinode situations, because many/most
		 * entries will be zero for the local node. So we just fill out
		 * the entries that concern the local node. 
		 * Returns the start index on the current node: this is not
		 * possible to compute just from the node#.
		 */
		unsigned int getNumData2( vector< unsigned int >& sizes ) const
		{
			sizes.assign( parentDataHandler_->numData(), 0 );
			for ( DataHandler::iterator i = parentDataHandler_->begin();
				i != parentDataHandler_->end(); ++i ) {
				char* pa = parentDataHandler_->data1( i );
				sizes[i] =  
				( ( reinterpret_cast< Parent* >( pa ) )->*getNumField_ )();
			}

			/*
			unsigned int size = parentDataHandler_->numData1();
			unsigned int start = 
				 ( size * Shell::myNode() ) / Shell::numNodes();
			unsigned int end = 
				 ( size * ( 1 + Shell::myNode() ) ) / Shell::numNodes();

			for ( unsigned int i = start; i < end; ++i ) {
				char* pa = parentDataHandler_->data1( i );
				sizes.push_back( 
				( ( reinterpret_cast< Parent* >( pa ) )->*getNumField_ )()
				);
			}
			*/
			return start_;
		}

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const {
			return parentDataHandler_->isDataHere( index );
		}

		bool isAllocated() const {
			return parentDataHandler_->isAllocated();
		}

		/**
		 * Again, this should really be done at the parent Element, not
		 * here.
		 */
		void allocate() {
			;
		}

		bool isGlobal() const
		{
			return parentDataHandler_->isGlobal();
		}

		/**
		 * This seems funny, but remember than begin() refers to the
		 * data part on the index.
		 * Don't want to permit iterating here, it will cause problems.
		 */
		iterator begin() const {
			return parentDataHandler_->begin();
		}

		// Don't want to permit iterating here, it will cause problems.
		iterator end() const {
			return parentDataHandler_->end();
		}

		const DataHandler* parentDataHandler() const {
			return parentDataHandler_;
		}

		unsigned int startDim2index() const {
			return start_;
		}

	protected:
		void setData( char* data, unsigned int numData ) {
			;
		}
	private:
		const DataHandler* parentDataHandler_;
		Field* ( Parent::*lookupField_ )( unsigned int );
		unsigned int ( Parent::*getNumField_ )() const;
		void ( Parent::*setNumField_ )( unsigned int num );
		unsigned int start_;
};

#endif	// _FIELD_DATA_HANDLER_H

