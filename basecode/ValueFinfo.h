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

/*
Template < class T, class A, 
	void( T::*SetFunc )( const A& ),
	const F& ( T::*GetFunc )() const > 
	class ValueFinfo< T, F, SetFunc, GetFunc >: 
	public Finfo
{
	public:
		ValueFinfo( const string& name, const string& doc)
			: Finfo( name, doc )
			{;}
		void registerOpFuncs(
			map< string, FuncId >& fnames, vector< OpFunc >& funcs ) 
		{
			string setName = "set" + name();
			map< string, FuncId >::iterator i = fnames.find( setName );
			if ( i != fnames.end() ) {
				funcs[ i.second ] = SetFunc;
			} else {
				unsigned int size = fnames.size();
				fnames[ setName ] = SetFunc;
			}

			string getName = "get" + name();
			map< string, FuncId >::iterator i = fnames.find( getName );
			if ( i != fnames.end() ) {
				funcs[ i.second ] = GetFunc;
			} else {
				unsigned int size = fnames.size();
				fnames[ getName ] = GetFunc;
			}
		}
	private:
		void ( T::*setFunc_ )( const F& );
		const F& ( T::*getFunc_ )() const;
};



template< class T, class A, void ( T::*F )( A ) >
	unsigned int async1( Eref e, const void* buf )
{
	(static_cast< T* >( e.data() )->*F)( 
		*static_cast< const A* >( buf ) );
	return sizeof( FuncId ) + sizeof ( A );
}

void getop( Eref e, const char* buf const ) {
	double ret = static_cast< Neutral* >( e.data() )->getName();
	Id src = *reinterpret_cast< Id* >( buf );
	buf += sizeof( Id );
	MsgId srcMsg = *reinterpret_cast< MsgId* >( buf );
	buf += sizeof( MsgId );
	FuncId srcFunc = *reinterpret_cast< FuncId* >( buf );
	Finfo2< MsgId, double > s( MsgId, srcFunc );
	s.sendTo( e, Id, ret );
}

*/
#endif // _VALUE_FINFO_H
