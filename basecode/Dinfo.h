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
		virtual Data* allocData( unsigned int numData ) const = 0;
		virtual unsigned int size() const = 0;
		virtual unsigned int serialize( char* buf, const Data* d ) const = 0;
		virtual unsigned int unserialize( const char* buf, Data* d ) const = 0;
		virtual bool isA( const DinfoBase* other ) const = 0;
	private:
		const int makeCompilerHappy;
};

template< class D > class Dinfo: public DinfoBase
{
	public:
		Dinfo()
		{;}
		Data* allocData( unsigned int numData ) const {
			return new D[ numData ];
		}

		unsigned int size() const {
			return sizeof( D );
		}

		// Will need to specialize for variable size and pointer-containing
		// D.
		unsigned int serialize( char* buf, const Data* d ) const {
			*reinterpret_cast< D* >( buf ) = *static_cast< const D* >( d );
			return sizeof( D );
		}

		unsigned int unserialize( const char* buf, Data* d ) const {
			*d = *reinterpret_cast< const D* >( buf );
			return sizeof( D );
		}
		bool isA( const DinfoBase* other ) const {
			return dynamic_cast< const Dinfo< D >* >( other );
		}
		// Possible problems of dependence here.
		// static const Cinfo* cinfo;

	private:
};

#endif // _DINFO_H
