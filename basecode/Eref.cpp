/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "sstream"
#include "Cinfo.h"

Eref Eref::root()
{
	return Eref( Element::root() );
}

void* Eref::data()
{
	return e->data( i );
}

bool Eref::operator<( const Eref& other ) const
{
	if ( e == other.e )
		return ( i < other.i );

	return ( e < other.e );
}

bool Eref::operator==( const Eref& other ) const
{
	return ( e == other.e && i == other.i );
}

Id Eref::id()
{
	Id ret = e->id();
	return ret.assignIndex( i );
}

bool Eref::add( int m1, Eref e2, int m2, unsigned int connTainerOption )
{
	assert( e != 0 && e2.e != 0 );
	assert( validMsg( m1 ) );
	assert( e2.validMsg( m2 ) );
	const Finfo* srcF = e->findFinfo( m1 );
	const Finfo* destF = e2.e->findFinfo( m2 );

	if ( srcF && destF )
		return srcF->add( *this, e2, destF, connTainerOption );
	cout << "Eref::add: Error: Could not find Finfos " <<
		srcF->name() << ", " << destF->name() << endl;
	return 0;
}

bool Eref::add( const string& f1, Eref e2, const string& f2,
	unsigned int connTainerOption )
{
	assert( e2.e != 0 );
	const Finfo* srcF = e->findFinfo( f1 );
	const Finfo* destF = e2.e->findFinfo( f2 );
	if ( !srcF ) {
		cout << "Eref::add: Error: Could not find element.srcFinfo " <<
			name() << "." << f1 << endl;
		return 0;
	}
	if ( !destF ) {
		cout << "Eref::add: Error: Could not find element.srcFinfo " <<
			e2.name() << "." << f2 << endl;
		return 0;
	}
	return srcF->add( *this, e2, destF, connTainerOption );
}

bool Eref::add( const string& f1, Eref e2, const string& f2)
{
	return add( f1, e2, f2, ConnTainer::Default );
}

bool Eref::drop( int msg, unsigned int doomed )
{
	if ( !validMsg( msg ) )
		return 0;
	if ( msg >= 0 ) {
		return e->varMsg( msg )->drop( e, doomed );
	} else {
		cout << "Not sure what to do here, as the lookup is non-sequential\n";
		vector< ConnTainer* >* ctv = e->getDest( msg );
		if ( doomed >= ctv->size() )
			return 0;
	}
	return 0;
}

/*
bool Eref::drop( int msg, const ConnTainer* doomed )
{
	if ( !validMsg( msg ) )
		return 0;
	if ( msg >= 0 ) {
		varMsg( msg )->drop( this, doomed );
		return 1;
	} else {
		cout << "Not sure what to do here in Eref::drop\n";
		return 0;
	}
}
*/

bool Eref::dropAll( int msg )
{
	if ( !validMsg( msg ) )
		return 0;
	if ( msg >= 0 ) {
		e->varMsg( msg )->dropAll( e );
		return 1;
	} else {
		vector< ConnTainer* >* ctv = e->getDest( msg );
		vector< ConnTainer* >::iterator k;
		for ( k = ctv->begin(); k != ctv->end(); k++ ) {
			bool ret = Msg::innerDrop( ( *k )->e1(), ( *k )->msg1(), *k );
			if ( ret )
				delete ( *k );
			else
				cout << "Error: Eref::dropAll(): innerDrop failed\n";
			*k = 0;
		}
		ctv->resize( 0 );
		// I could erase the entry in the dest_ map too. Later.
		return 1;
	}
}

bool Eref::dropAll( const string& finfo )
{
	const Finfo* f = e->findFinfo( finfo );
	if ( f ) {
		return dropAll( f->msg() );
	}
	return 0;
}

/**
 * Returns number dropped. Check to confirm that all went.
 * Concern in doing this is that we don't want to mess up the iterators.
 * Also need to be sure that no one else is using the iterators.
 */
bool Eref::dropVec( int msg, const vector< const ConnTainer* >& vec )
{
	if ( vec.size() == 0 )
		return 0;

	if ( !validMsg( msg ) )
		return 0;

	if ( msg >= 0 ) {
		Msg* m = e->varMsg( msg );
		assert ( m != 0 );
		vector< const ConnTainer* >::const_iterator i;
		for ( i = vec.begin(); i != vec.end(); i++ ) {
			bool ret = m->drop( ( *i )->e1(), *i );
			assert( ret );
		}
		return 1;
	} else {
		vector< ConnTainer* >* ctv = e->getDest( msg );
		assert ( ctv->size() >= vec.size() );
		vector< const ConnTainer* >::const_iterator i;
		for ( i = vec.begin(); i != vec.end(); i++ ) {
			int otherMsg = ( *i )->msg1();
			Element* otherElement = ( *i )->e1();
			Msg* om = otherElement->varMsg( otherMsg );
			assert( om );
			bool ret = om->drop( otherElement, *i );
			assert( ret );
		}
		return 1;
	}
	return 0;
}

bool Eref::validMsg( int msg ) const
{
	const Cinfo* c = e->cinfo();
	if ( msg >= 0 ) {
		if ( msg < static_cast< int >( c->numSrc() ) )
			return 1; // It is a valid msgSrc.
		return 0;
	}
	msg = -msg;
	if ( msg < static_cast< int >( c->numSrc() ) )
		return 0; // Should not have a msgDest less than the # of msgSrc.
	if ( msg < static_cast< int >( c->numFinfos() ) )
		return 1; // This is either a pure dest or a value field

	return 0;
}

string Eref::name() const
{
	if ( i == Id::AnyIndex )
		return ( e->name() + "[]" );
	if ( e->elementType() == "Array" ) {
		ostringstream s1;
		s1 << e->name() << "[" << i << "]";
		return s1.str();
	}

	return e->name();
}

string Eref::saneName(Id parent) const{
	if (e->elementType()=="Array" && parent()->elementType() == "Simple"){
		ostringstream s1;
		s1 << e->name() << "[" << i << "]";
		return s1.str();
	}
	return e->name();
}
