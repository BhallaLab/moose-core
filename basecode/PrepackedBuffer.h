/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class PrepackedBuffer
{
	public:
		PrepackedBuffer( const char* data, unsigned int size )
			: size_( size )
		{
			data_ = new char[ size ];
			memcpy( data_, data, size );
		}

		PrepackedBuffer( const PrepackedBuffer& other )
			: size_( other.size_ )
		{
			data_ = new char[ size_ ];
			memcpy( data_, other.data_, size_ );
		}

		~PrepackedBuffer() {
			delete data_;
		}

		const char* data() const {
			return data_;
		}

		unsigned int size() const {
			return size_;
		}

	private:
		unsigned int size_;
		char* data_;
};
