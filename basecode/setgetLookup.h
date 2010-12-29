/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SETGET_LOOKUP_H
#define _SETGET_LOOKUP_H


/**
 * This is essentially equivalent to set< T1, T2 > and even can use
 * identical interface internally.
*/
template < class T1, class T2 > bool lookupSet( 
	Eref e, const Finfo* f, T1 v, const T2& index )
{
	const LookupFtype< T1, T2 >* lf =
			dynamic_cast< const LookupFtype< T1, T2 >* >( f->ftype() );
	if ( lf ) {
		return lf->lookupSet( e, f, v, index );
	}
	cout << "Error: lookupSet( " << e.e->name() << "." << e.i << ", " << 
		f->name() << " T ): Finfo type mismatch." <<
                " Expected: " << f->ftype()->getTemplateParameters() <<
                ". Received: " << Ftype::full_type(typeid(T1)) << ", " <<
                Ftype::full_type(typeid(T2)) << "\n";

	return 0;
}

/**
 * Utility function for doing the set using a string lookup for Finfo
 */
template < class T1, class T2 > bool lookupSet( Eref e,
	const string& f, T1 v, const T2& index )
{
	const Finfo* finfo = e.e->findFinfo( f );
	if ( finfo == 0 ) {
		cout << "Error: lookupSet( " << e.e->name() << "." << e.i << ", " <<
			f << " T1, T2 ): Finfo not found\n";
		return 0;
	}
	return lookupSet<T1, T2>( e, finfo, v, index );
}

/**
* This is the utility func everyone should use for getting values
* from a lookup function such as array or map.
*/
template < class T1, class T2 > bool lookupGet(
		Eref e, const Finfo* f, T1& v, const T2& index )
{
	assert( e.e != 0 );
	assert( f != 0 );

	const LookupFtype< T1, T2 >* lf =
			dynamic_cast< const LookupFtype< T1, T2 >* >( f->ftype() );
	if ( lf ) {
		return lf->lookupGet( e, f, v, index );
	}
	cout << "Error: lookupGet( " << e.e->name() << "." << e.i << 
		", " << f->name() << " T1, T2 ): Finfo Type mismatch." <<
                " Expected: " << f->ftype()->getTemplateParameters() <<
                ". Received: " << Ftype::full_type(typeid(T1)) << ", " <<
                Ftype::full_type(typeid(T2)) << "\n";
	return 0;
}

/**
 * This is a utility function for using a string to identify the
 * Finfo in the 'get' command. Note that here the Element e is not
 * a const, because the string lookup operation may involve creation
 * of a temporary Finfo which is attached to the Element, thus
 * modifying it.
 */
template < class T1, class T2 > bool lookupGet(
	Eref e, const string& f, T1& v, const T2& index )
{
	const Finfo* finfo = e.e->findFinfo( f );
	if ( finfo == 0 ) {
		cout << "Error: lookupGet( " << e.e->name() << "." << e.i << 
			", " << f << " T1, T2 ): Finfo not found\n";
		return 0;
	}
	return lookupGet< T1, T2 >( e, finfo, v, index );
}

#endif // _SETGET_LOOKUP_H
