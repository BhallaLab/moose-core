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
		 * Constructor. 
		 * Here data is a pointer to the entire data block.
		 * dataSize is the size of the entire data block to be transferred,
		 * in units of doubles. Minimum size is therefore 1, corresponding
		 * to 8 bytes.
		 * If we have an array, then the whole array is packed efficiently,
		 * just that the start and end alignments are on double boundaries.
		 *  dataSize = individualDataSize * numEntries,
		 * with rounding up to convert to the double segment.
		 * numEntries is the # of array entries. For non-arrays it defaults
		 * to 0.
		 */
		PrepackedBuffer( const double* data, unsigned int dataSize, 
			unsigned int numEntries = 0 );

		PrepackedBuffer( const PrepackedBuffer& other );

		/**
		 * Constructor
		 * Here the char buffer is a serialized version of the 
		 * Prepacked buffer
		 */
		PrepackedBuffer( const double* buf );

		PrepackedBuffer();

		~PrepackedBuffer();

		const double* data() const {
			return data_;
		}

		/**
		 * looks up entry. If index exceeds numEntries_, then goes back to
		 * beginning. Lets us tile the target with repeating sequences.
		 * Most commonly just repeat one entry.
		 */
		const double* operator[]( unsigned int index ) const;

		/**
		 * 	Returns the size of the data contents, in doubles.
		 */
		unsigned int dataSize() const {
			return dataSize_;
		}

		/**
		 * Returns the size of the entire PrepackedBuffer  in terms of
		 * double*
		 */
		unsigned int size() const {
			return dataSize_ + 2; // Note that we don't need to transfer the individualDataSize_ entry.
		}

		/**
		 * Converts to a buffer. Buf must be preallocated.
		 */
		unsigned int conv2buf( double* buf ) const;

		/**
		 * Returns number of data entries: size of transferred array.
		 */
		unsigned int numEntries() const {
			return numEntries_;
		}

		/**
		 * Flag: is the data type a single value or a vector?
		 */
		bool isVector() const {
			return numEntries_ > 0;
		}
	private:
		unsigned int dataSize_; // Size of data.
		unsigned int numEntries_; // Number of data entries, if array.
		unsigned int individualDataSize_; // size of each entry.
		double* data_; // Converted data.
};
