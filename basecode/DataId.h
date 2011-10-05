/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DATAID_H
#define _DATAID_H

/**
 * Handles lookup of data fields in an Element. In most cases all we need
 * is the index for the data object. 
 * In principle, this object should be able to handle arbitrary
 * dimensions. Alternatively, it should be able to index more than 1e10
 * data entries. However, the total address space will be unlikely to exceed
 * 64 bits for a while, so one could imagine doing different kinds of
 * bitfields out of the total.
 */

class DataId
{
	public:
		friend ostream& operator <<( ostream& s, const DataId& i );
		friend istream& operator >>( istream& s, DataId& i );

		/**
		 * Default DataId is zero index
		 */
		DataId()
			: index_( 0 )
		{;}

		/**
		 * Creates a DataId with the specified index
		 */
		DataId( unsigned long long index )
			: index_( index )
		{;}

		/**
		 * Destructor. Nothing much to do here, move along.
		 */
		~DataId()
		{;}

		unsigned long long value() const
		{
			return index_;
		}

		/**
		 * Returns index of local object
		 */
		unsigned int myIndex( unsigned int mask ) const
		{
			return mask & index_;
		}

		/** 
		 * returns index of parent object
		 */
		unsigned int parentIndex( unsigned short bitOffset ) const
		{
			return index_ >> bitOffset;
		}
		///////////////////////////////////////////////////////////////////
		// Increment operators, to be used mostly by DataHandler::nextIndex
		///////////////////////////////////////////////////////////////////

		DataId operator++() { // prefix
			return ++index_;
		}

		/*
		DataId operator++( int ) { // postfix
			return index_++;
		}
		*/

		/**
		 * Increment index, and return true if there is a rollover where
		 * the index has exceeded the range allowed.
		 */
		bool increment( unsigned int max ) {
			if ( index_ >= max - 1 ) 
				return 1;
			++index_;
			return 0;
		}

		///////////////////////////////////////////////////////////////////
		// Comparison operators
		///////////////////////////////////////////////////////////////////

		bool operator==( const DataId& other ) const {
			return index_ == other.index_;
		}

		bool operator!=( const DataId& other ) const {
			return index_ != other.index_;
		}

		bool operator<( const DataId& other ) const {
			return ( index_ < other.index_ );
		}

		///////////////////////////////////////////////////////////////////
		// Predefined values
		///////////////////////////////////////////////////////////////////
		static const DataId bad;
		static const DataId any;
		static const DataId globalField;
	private:
		unsigned long long index_;
};


#endif // _DATAID_H
