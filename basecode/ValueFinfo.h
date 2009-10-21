/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _VALUE_FINFO_H
#define _VALUE_FINFO_H

template < class T, class F > class ValueFinfo: public Finfo
{
	public:
		~ValueFinfo() {
			delete setOpFunc_;
			delete getOpFunc_;
		}

		ValueFinfo( const string& name, const string& doc, 
			void ( T::*setFunc )( const F& ),
			const F& ( T::*getFunc )() const )
			: Finfo( name, doc )
			{
				setOpFunc_ = new OpFunc1< T, F >( setFunc );
				getOpFunc_ = new GetOpFunc< T, F >( getFunc );
			}


		void registerOpFuncs(
			map< string, FuncId >& fnames, vector< OpFunc* >& funcs ) 
		{
			string setName = "set" + name();
			map< string, FuncId >::iterator i = fnames.find( setName );
			if ( i != fnames.end() ) {
				funcs[ i->second ] = setOpFunc_;
			} else {
				unsigned int size = funcs.size();
				fnames[ setName ] = size;
				funcs.push_back( setOpFunc_ );

			}
			string getName = "get" + name();
			i = fnames.find( getName );
			if ( i != fnames.end() ) {
				funcs[ i->second ] = getOpFunc_;
			} else {
				unsigned int size = funcs.size();
				fnames[ getName ] = size;
				funcs.push_back( getOpFunc_ );
			}
		}

		// Need to think about whether an index has to be registered here.
		unsigned int registerSrcFuncIndex( unsigned int current )
		{
			return current;
		}

		// Need to think about whether an index has to be registered here.
		unsigned int registerConn( unsigned int current )
		{
			return current;
		}

	private:
		OpFunc1< T, F >* setOpFunc_;
		GetOpFunc< T, F >* getOpFunc_;
};

#endif // _VALUE_FINFO_H
