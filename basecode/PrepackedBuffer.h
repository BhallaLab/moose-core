/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * this class is used to take data that has already been converted into
 * a buffer by one or more Conv<> operations, and send it between nodes.
 * Useful when one doesn't know the data type being transferred.
 */
class PrepackedBuffer
{
	public:
		/**
		 * Constructor. Here size is the size of the data
		 */
		PrepackedBuffer( const char* data, unsigned int dataSize )
			: dataSize_( dataSize )
		{
			data_ = new char[ dataSize ];
			memcpy( data_, data, dataSize );
		}

		PrepackedBuffer( const PrepackedBuffer& other )
			: dataSize_( other.dataSize_ )
		{
			data_ = new char[ dataSize_ ];
			memcpy( data_, other.data_, dataSize_ );
		}

		~PrepackedBuffer() {
			delete[] data_;
		}

		const char* data() const {
			return data_;
		}

		/**
		 * 	Returns the size of the data contents.
		 */
		unsigned int dataSize() const {
			return dataSize_;
		}

	private:
		unsigned int dataSize_; // Size of data.
		char* data_; // Converted data.
};
