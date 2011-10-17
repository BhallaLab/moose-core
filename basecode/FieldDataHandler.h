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
 * It is templated by the field type, the parent type and a lookup function * that extracts the field from the parent.  */

template< class Parent, class Field > class FieldDataHandler: public FieldDataHandlerBase
{
	public:
		FieldDataHandler( const DinfoBase* dinfo,
			const DataHandler* parentDataHandler,
			unsigned int size,
			Field* ( Parent::*lookupField )( unsigned int ),
			unsigned int ( Parent::*getNumField )() const,
			void ( Parent::*setNumField )( unsigned int num ) )
			: 
				FieldDataHandlerBase( dinfo , parentDataHandler, size ),
				lookupField_( lookupField ),
				getNumField_( getNumField ),
				setNumField_( setNumField )
		{;}

		~FieldDataHandler()
		{;} // Don't delete data because the parent Element should do so.

		///////////////////////////////////////////////////////////////
		// Information functions
		///////////////////////////////////////////////////////////////
		// Already defined FieldDataHandlerBase::data()
		// Already defined FieldDataHandlerBase::totalEntries()
		// Already defined FieldDataHandlerBase::localEntries()
		// Already defined FieldDataHandlerBase::numDimensions()
		// Already defined FieldDataHandlerBase::sizeOfDim()
		// Already defined FieldDataHandlerBase::dims()
		// Already defined FieldDataHandlerBase::isDataHere()
		// Already defined FieldDataHandlerBase::isAllocated()
		///////////////////////////////////////////////////////////////
		// Special field access funcs
		///////////////////////////////////////////////////////////////
		/**
		 * Returns the pointer to the field entry at fieldIndex, on the
		 * parent data entry at data.
		 */
		char* lookupField( char* data, unsigned int fieldIndex ) const
		{
			if ( data ) {
				Parent* pa = reinterpret_cast< Parent* >( data );
				Field* s = ( pa->*lookupField_ )( fieldIndex );
				return reinterpret_cast< char* >( s );
			}
			return 0;
		}

		/**
		 * Returns the number of field entries on parent data entry.
		 */
		unsigned int getNumField( const char* data ) const
		{
			if ( data ) {
				const Parent* pa = reinterpret_cast< const Parent* >( data);
				return ( pa->*getNumField_ )();
			}
			return 0;
		}

		/**
		 * Assigns the number of field entries on parent data entry.
		 */
		void setNumField( char* data, unsigned int size )
		{
			if ( data ) {
				Parent* pa = reinterpret_cast< Parent* >( data );
				( pa->*setNumField_ )( size );
				if ( size > getMaxFieldEntries() ) {
					setMaxFieldEntries( size );
					// setFieldDimension( size );
					/// Here we need to request the higher powers to realloc
					/// the field dimension.
				}
			}
		}
		// Already defined FieldDataHandlerBase::setFieldArraySize()
		// Already defined FieldDataHandlerBase::getFieldArraySize()
		// Already defined FieldDataHandlerBase::biggestFieldArraySize()

		///////////////////////////////////////////////////////////////
		// Load balancing functions
		///////////////////////////////////////////////////////////////
		// Already defined FieldDataHandlerBase::innerNodeBalance()

		///////////////////////////////////////////////////////////////
		// Process and foreach
		///////////////////////////////////////////////////////////////
		// Already defined FieldDataHandlerBase::process()
		// Already defined FieldDataHandlerBase::foreach()
		// Already defined FieldDataHandlerBase::getAllData()

		///////////////////////////////////////////////////////////////
		// Data reallocation
		///////////////////////////////////////////////////////////////
		// Already defined FieldDataHandlerBase::globalize()
		// Already defined FieldDataHandlerBase::unGlobalize()
		// Already defined FieldDataHandlerBase::resize()


		/**
		 * Makes a copy of the FieldDataHandler.
		 * Needs post-processing to substitute in the new parent.
		 * Ignore the copyDepth argument as it comes from the new parent.
		 */
		DataHandler* copy( unsigned short copyDepth, bool toGlobal, 
			unsigned int n ) const
		{
			FieldDataHandler< Parent, Field >* ret =
				new FieldDataHandler< Parent, Field >( *this );
			return ret;
		}

		/**
		 * I'm dubious about this one too, because I don't see how the
		 * original lookup and set/get functions could work on a different
		 * dinfo
		 */
		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo ) const
		{
			FieldDataHandler< Parent, Field >* ret =
				new FieldDataHandler< Parent, Field >(
					dinfo,
					parentDataHandler(),
					getMaxFieldEntries(),
					lookupField_,
					getNumField_,
					setNumField_
				);
			return ret;
		}

		void assign( const char* orig, unsigned int numOrig )
		{
			// Need to iterate through parent data to do this assignment
			;
		}

		///////////////////////////////////////////////////////////////

	private:
		Field* ( Parent::*lookupField_ )( unsigned int );
		unsigned int ( Parent::*getNumField_ )() const;
		void ( Parent::*setNumField_ )( unsigned int num );
};

#endif	// _FIELD_DATA_HANDLER_H

