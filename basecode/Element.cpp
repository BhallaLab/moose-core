/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"

// static Element* UNKNOWN_NODE = reinterpret_cast< Element* >( 1L );
// const unsigned int BAD_ID = ~0;
// const unsigned int MAX_ID = 1000000;
// const unsigned int MIN_NODE = 1;
// const unsigned int MAX_NODE = 65536; // Dream on.

/**
 * The normal base element constructor needs an id for the element
 * as soon as it is made, and puts the element onto the elementList.
 * This function is called on all nodes, so it must handle cases where
 * there are gaps between the previous and current id assigned to this
 * node. Those gaps are referred to the master node by default.
 * This checks if an id is already in use.
 */
Element::Element( Id id )
	: id_( id )
{
	id.setElement( this );

/*
	assert( id < MAX_ID );
	assert( id() == 0 );

	unsigned int prevSize = elementList().size();
	if ( id > prevSize ) {
		elementList().resize( id + 1 );
		for ( unsigned int i = prevSize; i < id; i++ )
			elementList()[ i ] = UNKNOWN_NODE;
	} else {
		assert( elementList()[ id ] == 0 ) ; // Do not overwrite.
	}
	elementList()[ id ] = this;
	*/
}

/**
 * This variant is used for making dummy elements without messing
 * with the elementList. The id is hardcoded to zero here, so that
 * the destructor does not attempt to clear the location in the
 * ElementList. Note that the argument is just ignored.
 */
Element::Element( bool ignoreId )
{ ; }

/**
 * The virtual destructor for Elements cleans up the entry on the
 * elementList. The special case of zero id (from above) is not
 * deleted.
 */
Element::~Element()
{
	if ( !id_.zero() )
		id_.setElement( 0 );
}

bool Element::isTarget( const Element* tgt ) const
{
	unsigned int n = numMsg();
	for ( unsigned int i = 0; i < n; ++i )
		if ( msg( i )->isTarget( this, tgt ) )
			return 1;
	return 0;
}

void Element::setId( Id id )
{
	id_ = id;
}

/**
 * Here we work with a single big array of all ids. Off-node elements
 * are represented by their postmasters. When we hit a postmaster we
 * put the id into a special field on it. Note that this is horrendously
 * thread-unsafe.
 * \todo: I need to replace the off-node case with a wrapper Element
 * return. The object stored here will continue to be the postmaster,
 * and when this is detected it will put the postmaster ptr and the id
 * into the wrapper element. The wrapper's own id will be zero so it
 * can be safely deleted.
 */
/*
Element* Element::element( unsigned int id )
{
	if ( id < elementList().size() ) {
		Element* ret = elementList()[ id ];
		if ( ret == 0 )
			return 0;
		if ( ret == UNKNOWN_NODE )
			// don't know how to handle this yet. It should trigger
			// a request to the master node to update the elist.
			// We then get into managing how many entries are unknown...
			assert( 0 );
		if ( ret->className() == "PostMaster" ) {
			set< unsigned int >( ret, "targetId", id );
		}
		return elementList()[ id ];
	}
	return 0;
}
*/

/**
 * Returns the most recently created element.
 * It is a static function. Deprecated.
 */
/*
Element* Element::lastElement()
{
	assert ( elementList().size() > 0 );
	return Element::element( elementList().size() - 1 );
}
*/

/**
 * Function for accessing the element list in an initialization-sequence
 * independent manner
 */
/*
vector< Element* >& Element::elementList()
{
	static vector< Element* > elementList;

	return elementList;
}
*/

/**
 * Returns the next available id and allocates space for it.
 * Later can be refined to mop up freed ids. 
 * Should only be called on master node.
 */
/*
unsigned int Element::nextId()
{
	elementList().push_back( 0 );
	return elementList().size() - 1;
}
*/

/**
 * Returns the current high id. Used to find the most recently created
 * object. Should only be called on master node.
 */
/*
unsigned int Element::lastId()
{
	return elementList().size() - 1;
}
*/

#if 0
bool Element::add( int m1, Element* e2, int m2 )
{
	assert( e2 != 0 );
	assert( validMsg( m1 ) );
	assert( validMsg( m2 ) );
	const Finfo* srcF = findFinfo( m1 );
	const Finfo* destF = e2->findFinfo( m2 );

	if ( srcF && destF )
		return srcF->add( this, e2, destF );
	cout << "Element::add: Error: Could not find Finfos " <<
		srcF->name() << ", " << destF->name() << endl;
	return 0;
}

bool Element::add( const string& f1, Element* e2, const string& f2 )
{
	assert( e2 != 0 );
	const Finfo* srcF = findFinfo( f1 );
	const Finfo* destF = e2->findFinfo( f2 );
	if ( !srcF ) {
		cout << "Element::add: Error: Could not find element.srcFinfo " <<
			name() << "." << f1 << endl;
		return 0;
	}
	if ( !destF ) {
		cout << "Element::add: Error: Could not find element.srcFinfo " <<
			e2->name() << "." << f2 << endl;
		return 0;
	}
	return srcF->add( this, e2, destF );
}

bool Element::drop( int msg, unsigned int doomed )
{
	if ( !validMsg( msg ) )
		return 0;
	if ( msg >= 0 ) {
		return varMsg( msg )->drop( this, doomed );
	} else {
		cout << "Not sure what to do here, as the lookup is non-sequential\n";
		vector< ConnTainer* >* ctv = getDest( msg );
		if ( doomed >= ctv->size() )
			return 0;
	}
	return 0;
}

/*
bool Element::drop( int msg, const ConnTainer* doomed )
{
	if ( !validMsg( msg ) )
		return 0;
	if ( msg >= 0 ) {
		varMsg( msg )->drop( this, doomed );
		return 1;
	} else {
		cout << "Not sure what to do here in Element::drop\n";
		return 0;
	}
}
*/

bool Element::dropAll( int msg )
{
	if ( !validMsg( msg ) )
		return 0;
	if ( msg >= 0 ) {
		varMsg( msg )->dropAll( this );
		return 1;
	} else {
		vector< ConnTainer* >* ctv = getDest( msg );
		vector< ConnTainer* >::iterator k;
		for ( k = ctv->begin(); k != ctv->end(); k++ ) {
			bool ret = Msg::innerDrop( ( *k )->e1(), ( *k )->msg1(), *k );
			if ( ret )
				delete ( *k );
			else
				cout << "Error: Element::dropAll(): innerDrop failed\n";
			*k = 0;
		}
		ctv->resize( 0 );
		// I could erase the entry in the dest_ map too. Later.
		return 1;
	}
}

bool Element::dropAll( const string& finfo )
{
	const Finfo* f = findFinfo( finfo );
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
bool Element::dropVec( int msg, const vector< const ConnTainer* >& vec )
{
	if ( vec.size() == 0 )
		return 0;

	if ( !validMsg( msg ) )
		return 0;

	if ( msg >= 0 ) {
		Msg* m = varMsg( msg );
		assert ( m != 0 );
		vector< const ConnTainer* >::const_iterator i;
		for ( i = vec.begin(); i != vec.end(); i++ ) {
			bool ret = m->drop( ( *i )->e1(), *i );
			assert( ret );
		}
		return 1;
	} else {
		vector< ConnTainer* >* ctv = getDest( msg );
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

bool Element::validMsg( int msg ) const
{
	const Cinfo* c = cinfo();
	if ( msg > 0 && msg < static_cast< int >( c->numSrc() ) )
		return 1;
	if ( msg < 0 && -msg < static_cast< int >( c->numSrc() ) )
		return 0;
	if ( msg < 0 && -msg < static_cast< int >( c->numFinfos() ) )
		return 1;

	return 0;
}
#endif
