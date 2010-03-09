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
			void ( T::*setFunc )( F ),
			F ( T::*getFunc )() const )
			: Finfo( name, doc )
			{
				setOpFunc_ = new OpFunc1< T, F >( setFunc );
				getOpFunc_ = new GetOpFunc< T, F >( getFunc );
			}


		void registerOpFuncs(
			map< string, FuncId >& fnames, vector< OpFunc* >& funcs ) 
		{
			string setName = "set_" + name();
			map< string, FuncId >::iterator i = fnames.find( setName );
			if ( i != fnames.end() ) {
				funcs[ i->second ] = setOpFunc_;
			} else {
				unsigned int size = funcs.size();
				fnames[ setName ] = size;
				funcs.push_back( setOpFunc_ );

			}
			string getName = "get_" + name();
			i = fnames.find( getName );
			if ( i != fnames.end() ) {
				funcs[ i->second ] = getOpFunc_;
			} else {
				unsigned int size = funcs.size();
				fnames[ getName ] = size;
				funcs.push_back( getOpFunc_ );
			}
		}

		BindIndex registerBindIndex( BindIndex current )
		{
			return current;
		}

	private:
		OpFunc1< T, F >* setOpFunc_;
		GetOpFunc< T, F >* getOpFunc_;
};


template < class T, class F > class ReadonlyValueFinfo: public Finfo
{
	public:
		~ReadonlyValueFinfo() {
			delete getOpFunc_;
		}

		ReadonlyValueFinfo( const string& name, const string& doc, 
			F ( T::*getFunc )() const )
			: Finfo( name, doc )
			{
				getOpFunc_ = new GetOpFunc< T, F >( getFunc );
			}


		void registerOpFuncs(
			map< string, FuncId >& fnames, vector< OpFunc* >& funcs ) 
		{
			string getName = "get_" + name();
			map< string, FuncId >::iterator i = fnames.find( getName );
			if ( i != fnames.end() ) {
				funcs[ i->second ] = getOpFunc_;
			} else {
				unsigned int size = funcs.size();
				fnames[ getName ] = size;
				funcs.push_back( getOpFunc_ );
			}
		}

		// Need to think about whether an index has to be registered here.
		BindIndex registerBindIndex( BindIndex current )
		{
			return current;
		}

	private:
		GetOpFunc< T, F >* getOpFunc_;
};

/**
 * Here the value belongs to an array field within class T.
 * This is used when the assignment function for an array field 
 * should also update some information in the parent class T.
 * The function thus does not refer to the class of the array field.
 */
template < class T, class F > class UpValueFinfo: public Finfo
{
	public:
		~UpValueFinfo() {
			delete setUpFunc_;
			delete getUpFunc_;
		}

		UpValueFinfo( const string& name, const string& doc, 
			void ( T::*setFunc )( DataId, F ),
			F ( T::*getFunc )( DataId ) const )
			: Finfo( name, doc )
			{
				setUpFunc_ = new UpFunc1< T, F >( setFunc );
				getUpFunc_ = new GetUpFunc< T, F >( getFunc );
			}


		void registerOpFuncs(
			map< string, FuncId >& fnames, vector< OpFunc* >& funcs ) 
		{
			string setName = "set_" + name();
			map< string, FuncId >::iterator i = fnames.find( setName );
			if ( i != fnames.end() ) {
				funcs[ i->second ] = setUpFunc_;
			} else {
				unsigned int size = funcs.size();
				fnames[ setName ] = size;
				funcs.push_back( setUpFunc_ );

			}
			string getName = "get_" + name();
			i = fnames.find( getName );
			if ( i != fnames.end() ) {
				funcs[ i->second ] = getUpFunc_;
			} else {
				unsigned int size = funcs.size();
				fnames[ getName ] = size;
				funcs.push_back( getUpFunc_ );
			}
		}

		// Need to think about whether an index has to be registered here.
		BindIndex registerBindIndex( BindIndex current )
		{
			return current;
		}

	private:
		UpFunc1< T, F >* setUpFunc_;
		GetUpFunc< T, F >* getUpFunc_;
};

#endif // _VALUE_FINFO_H
