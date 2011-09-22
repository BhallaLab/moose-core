/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _DINFO_H
#define _DINFO_H

class DinfoBase
{
	public:
		DinfoBase()
			: makeCompilerHappy( 0 )
		{;}
		virtual ~DinfoBase()
		{;}
		virtual char* allocData( unsigned int numData ) const = 0;
		virtual void destroyData( char* d ) const = 0;
		virtual unsigned int size() const = 0;

		/**
		 * Analogous to copying a vector into a bigger one. Repeat the
		 * original data as many times as possible.
		 * Destroys old data and allocates new.
		 * returns new data.
		 */
		virtual char* copyData( const char* orig, unsigned int origSize,
			unsigned int copySize ) const = 0;
		/*
		static unsigned int size( const D* value ) const = 0;
		static unsigned int serialize( char* buf, const Data* d ) const = 0;
		static unsigned int unserialize( const char* buf, Data* d ) const = 0;
		*/
		virtual bool isA( const DinfoBase* other ) const = 0;
	private:
		const int makeCompilerHappy;
};

template< class D > class Dinfo: public DinfoBase
{
	public:
		Dinfo()
		{;}
		char* allocData( unsigned int numData ) const {
			if ( numData == 0 )
				return 0;
			else 
				return reinterpret_cast< char* >( new( nothrow) D[ numData ] );
		}

		char* copyData( const char* orig, unsigned int origSize,
			unsigned int copySize ) const
		{
			if ( origSize == 0 )
				return 0;
			D* ret = new( nothrow ) D[copySize];
			if ( !ret )
				return 0;
			const D* origData = reinterpret_cast< const D* >( orig );
			for ( unsigned int i = 0; i < copySize; ++i ) {
				ret[ i ] = origData[ i % origSize ];
			}

			/*
			D* ret = new D[ numData * numCopies ];
			const D* origData = reinterpret_cast< const D* >( orig );
			for ( unsigned int i = 0; i < numData; ++i ) {
				for ( unsigned int j = 0; j < numCopies; ++j ) {
					ret[ i * numCopies + j ] = origData[ i ];
				}
			}
			*/
			return reinterpret_cast< char* >( ret );
		}

		void destroyData( char* d ) const {
			delete[] reinterpret_cast< D* >( d );
		}

		unsigned int size()  const {
			return sizeof( D );
		}

		/*
		// Will need to specialize for variable size and pointer-containing
		// D.
		static unsigned int serialize( char* buf, const Data* d ) {
			*reinterpret_cast< D* >( buf ) = *static_cast< const D* >( d );
			return sizeof( D );
		}

		static unsigned int unserialize( const char* buf, Data* d ) {
			*d = *reinterpret_cast< const D* >( buf );
			return sizeof( D );
		}
		// Possible problems of dependence here.
		// static const Cinfo* cinfo;
		*/
		bool isA( const DinfoBase* other ) const {
			return dynamic_cast< const Dinfo< D >* >( other );
		}
};

template< class D > class ZeroSizeDinfo: public Dinfo< D >
{
	public:
		unsigned int size()  const {
			return 0;
		}
};

#endif // _DINFO_H
