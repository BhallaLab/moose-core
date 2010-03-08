/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SETGET_H
#define _SETGET_H


/**
* This is the utility func everyone should use for calling zero-arg
* functions. It is named set for congruence with similar routines
* for larger numbers of arguments.
*/

extern bool set( Eref e, const Finfo* f );
extern bool set( Eref e, const string& f );

/**
* This is the utility func everyone should use for setting values.
* It does appropriate typechecking of the finfo before doing anything.
*/
template < class T > bool set( Eref e, const Finfo* f, T v )
{
	const Ftype1< T >* f1 =
			dynamic_cast< const Ftype1< T >* >( f->ftype() );
	if ( f1 ) {
		return f1->set( e, f, v );
	}
	cout << "Error: set( " << e.e->name() << "." << e.i << ", " <<
		f->name() << " T ): Finfo type mismatch\n";
	return 0;
}

/**
 * Utility function for doing the set using a string lookup for Finfo
 */
template < class T > bool set( Eref e, const string& f, T v )
{
	const Finfo* finfo = e.e->findFinfo( f );
	if ( finfo == 0 ) {
		cout << "Error: set( " << e.e->name() << "." << e.i << ", " << 
			f << " T ): Finfo not found\n";
		return 0;
	}
	return ::set( e, finfo, v );
}

/**
* This is the utility func everyone should use for getting values.
* It does appropriate typechecking of the finfo before doing anything.
* Perhaps we should not use the simple getFunc, but something
* involving the finfos Ftype and taking the Finfo also as an
* argument so that any extra args (e.g., indices) may be
* passed in.
* The other issue is the disposition of the Finfo* f. If it is a 
* DynamicFinfo created specially for this operation, it should be
* deleted. This temporary status is known to the DynamicFinfo
* because it has no Conns assigned. Alternatively we have a 
* Finfo wrapper variant that releases resources on deletion.
* Or we do not permit direct access via the Finfo pointer,
* only via the string so that we have a self-contained variant.
*/
template < class T > bool get( Eref e, const Finfo* f, T& v )
{
	assert( e.e != 0 );
	assert( f != 0 );

	const Ftype1< T >* f1 =
			dynamic_cast< const Ftype1< T >* >( f->ftype() );
	if ( f1 ) {
		return f1->get( e, f, v );
	}
	cout << "Error: get( " << e.e->name() << "." << e.i << ", " << 
		f->name() << " T ): Finfo Type mismatch\n";
	return 0;
}

/**
 * This is a utility function for using a string to identify the
 * Finfo in the 'get' command. Note that here the Element e is not
 * a const, because the string lookup operation may involve creation
 * of a temporary Finfo which is attached to the Element, thus
 * modifying it.
 */
template < class T > bool get( Eref e, const string& f, T& v )
{
	const Finfo* finfo = e.e->findFinfo( f );
	if ( finfo == 0 ) {
		cout << "Error: get( " << e.e->name() << "." << e.i << ", " << 
			f << " T ): Finfo not found\n";
		return 0;
	}
	return get( e, finfo, v );
}


/**
* This is the utility func for calling 2-argument functions.
* It does appropriate typechecking of the finfo before doing anything.
*/
template < class T1, class T2 > bool set(
				Eref e, const Finfo* f, T1 v1, T2 v2 )
{
	// make sure that the Finfo f is on the element.
	///\todo Later we can remove this check by using a safe call function.
	vector< const Finfo* > flist;
	e.e->listFinfos( flist );
	if ( find( flist.begin(), flist.end(), f ) == flist.end() ) {
		cout << "Error: set( " << e.e->name() << "." << e.i << 
			", " << f->name() << " T1, T2 ): Finfo not found\n";
		return 0;
	}

	const Ftype2< T1, T2 >* f2 =
			dynamic_cast< const Ftype2< T1, T2 >* >( f->ftype() );
	if ( f2 ) {
		return f2->set( e, f, v1, v2 );
	}
	cout << "Error: set2( " << e.e->name() << "." << e.i << ", " << 
		f->name() << " T1, T2 ): Finfo type mismatch\n";
	return 0;
}

/**
 * Utility function for doing the set using a string lookup for Finfo
 */
template < class T1, class T2 > bool set( 
			Eref e, const string& f, T1 v1, T2 v2 )
{
	const Finfo* finfo = e.e->findFinfo( f );
	if ( finfo == 0 ) {
		cout << "Error: set( " << e.e->name() << "." << e.i << ", " << 
			f << " T1, T2 ): Finfo not found\n";
		return 0;
	}
	return ::set( e, finfo, v1, v2 );
}

/**
* This is the utility func for calling 3-argument functions.
* It does appropriate typechecking of the finfo before doing anything.
*/
template < class T1, class T2, class T3 > bool set(
				Eref e, const Finfo* f, T1 v1, T2 v2, T3 v3 )
{
	// make sure that the Finfo f is on the element.
	///\todo Later we can remove this check by using a safe call function.
	vector< const Finfo* > flist;
	e.e->listFinfos( flist );
	if ( find( flist.begin(), flist.end(), f ) == flist.end() ) {
		cout << "Error: set( " << e.e->name() << "." << e.i << 
			", " << f->name() << " T1, T2, T3 ): Finfo not found\n";
		return 0;
	}

	const Ftype3< T1, T2, T3 >* f3 =
			dynamic_cast< const Ftype3< T1, T2, T3 >* >( f->ftype() );
	if ( f3 ) {
		return f3->set( e, f, v1, v2, v3 );
	}
	cout << "Error: set3( " << e.e->name() << "." << e.i << ", " << 
		f->name() << " T1, T2, T3 ): Finfo type mismatch\n";
	return 0;
}

/**
 * Utility function for doing the set using a string lookup for Finfo
 */
template < class T1, class T2, class T3 > bool set( 
			Eref e, const string& f, T1 v1, T2 v2, T3 v3 )
{
	const Finfo* finfo = e.e->findFinfo( f );
	if ( finfo == 0 ) {
		cout << "Error: set( " << e.e->name() << "." << e.i << ", " << 
			f << " T1, T2, T3 ): Finfo not found\n";
		return 0;
	}
	return ::set( e, finfo, v1, v2, v3 );
}


/**
* This is the utility func for calling 4-argument functions.
* It does appropriate typechecking of the finfo before doing anything.
*/
template < class T1, class T2, class T3, class T4 > bool set(
				Eref e, const Finfo* f, T1 v1, T2 v2, T3 v3, T4 v4 )
{
	// make sure that the Finfo f is on the element.
	///\todo Later we can remove this check by using a safe call function.
	vector< const Finfo* > flist;
	e.e->listFinfos( flist );
	if ( find( flist.begin(), flist.end(), f ) == flist.end() ) {
		cout << "Error: set( " << e.e->name() << "." << e.i << ", " <<
			f->name() << " T1, T2, T3, T4 ): Finfo not found\n";
		return 0;
	}

	const Ftype4< T1, T2, T3, T4 >* f4 =
			dynamic_cast< const Ftype4< T1, T2, T3, T4 >* >( f->ftype() );
	if ( f4 ) {
		return f4->set( e, f, v1, v2, v3, v4 );
	}
	cout << "Error: set4( " << e.e->name() << "." << e.i << ", " <<
		f->name() << " T1, T2, T3, T4 ): Finfo type mismatch\n";
	return 0;
}

/**
 * Utility function for doing the set using a string lookup for Finfo
 */
template < class T1, class T2, class T3, class T4 > bool set( 
			Eref e, const string& f, T1 v1, T2 v2, T3 v3, T4 v4 )
{
	const Finfo* finfo = e.e->findFinfo( f );
	if ( finfo == 0 ) {
		cout << "Error: set( " << e.e->name() << "." << e.i << ", " <<
			f << " T1, T2, T3, T4 ): Finfo not found\n";
		return 0;
	}
	return ::set( e, finfo, v1, v2, v3, v4 );
}
#endif // _SETGET_H
