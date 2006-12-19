/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <fstream>
#include "header.h"

//////////////////////////////////////////////////////////////////
// Define and initialize the global root and classes Elements.
//////////////////////////////////////////////////////////////////
Element* Element::classes_;
Element* Element::root_ = Element::initializeRoot(); 

//////////////////////////////////////////////////////////////////
// Element initialization for moose classes. 
// Note that Elements are themselves pure virtual base classes,
// but we need to do this initialization to set up fields which
// are used by all derived classes
//////////////////////////////////////////////////////////////////


Finfo* Element::fieldArray_[] = 
{
	new ValueFinfo< string >("name", 
		&Element::getName, &Element::setName, "string"),
	new Dest0Finfo( "child_in",
		&Element::deleteFunc, &Element::getChildIn, "")
};

const Cinfo Element::cinfo_(
	"Element",
	"Upinder S. Bhalla, NCBS",
	"Element class. Abstract base class for MOOSE objects",
	"Element",
	Element::fieldArray_,
	sizeof(Element::fieldArray_)/sizeof(Finfo *),
	&Element::create
);

//////////////////////////////////////////////////////////////////
// Element functions here.
//////////////////////////////////////////////////////////////////

// Looks up the parent of this element.
Element* Element::parent() const {
	if ( childIn_.nTargets() > 0 )
		return childIn_.target(0)->parent();
	else if ( this == root() )
		return root();
	else
		return 0;
}

// Creates the new root Element.
// Called for the static initialization of Element::Root
// So it cannot assume anything about other statically initialized
// fields.
Element* Element::initializeRoot()
{
	Element* r = new Neutral("root");
	classes_ = new Neutral("classes");
	r->adoptChild(classes_);
	return r;
}

// Places new relays on the Element.
// For now this is simple. Later we do a bit of sorting:
// New field-modifying relays are inserted near the front so
// that searches for field matches are faster. 
// Solvers are first.
/*
void Element::AddRelay(const relay& r)
{
	relays.push_back(r);
}
*/

// Ask Element for finfo for named field.
// If given, use this as ret and wrap the other ret in it.
Field Element::field( const string& name )
{
	Field ret;
	for ( long i = relays_.size() - 1; i >= 0; i-- ) {
		ret = relays_[ i ]->match( name );
		if ( ret.good() ) {
			ret.setElement( this );
			return ret;
		}		
	}

	ret = this->cinfo()->field( name );
	if ( ret.good() )
		ret.setElement( this );
	return ret;
}

// Looks for a RelayFinfo holding the valueFinfo, and checks if
// it has a vacant slot for either sending or receiving.
// Returns the Finfo if found. Used to associate the trigger and
// return value messages in a value relay.
Finfo* Element::findVacantValueRelay( 
	Finfo* valueFinfo, bool isSending )
{
	Field ret;
	for ( long i = relays_.size() - 1; i >= 0; i-- ) {
		RelayFinfo* r = dynamic_cast< RelayFinfo* >( relays_[ i ] );
		if ( r &&  r->innerFinfo() == valueFinfo ) {
			if ( isSending ) {
				if (r->inConn( this )->nTargets() == 0 &&
					r->outConn( this )->nTargets() > 0 )
					return r;
			} else {
				if (r->outConn( this )->nTargets() == 0 &&
					r->inConn( this )->nTargets() > 0 )
					return r;
			}
		}
	}
	return 0;
}

void Element::listFields( vector< Finfo* >& ret )
{
	for ( long i = relays_.size() - 1; i >= 0; i-- )
		ret.push_back( relays_[ i ] );

//	cout << "Num relays for " << name() << " = " << ret.size() << "\n";
	const_cast< Cinfo* >( this->cinfo() )->listFields( ret );
}

// Returns a field identifying the finfo/element for the source
// of a message.
// The conn is the conn on the src. It also has src parent info.
// The func is the recvfunc of the destination of the message, which
// at message creation time is stored in the src MsgSrcs and
// in certain kinds of Finfos and relays.
// Using both conn and func we can identify the src.
// First scan the builtin messages as most cases emanate from there.
Field Element::lookupSrcField( Conn* conn, RecvFunc func )
{
	Finfo* f = cinfo()->findRemoteMsg( conn, func );
	if ( f )
		return Field( f, this );

	vector< Finfo* >::iterator i;
	for (i = relays_.begin(); i != relays_.end(); i++ ) {
		if ( ( *i )->outConn( conn->parent() ) == conn && 
			(*i)->matchRemoteFunc( this, func ) )
			return Field( *i, this );
	}

	return Field();
}

// Returns a field identifying the finfo/element for the destination
// of a message.
// The conn is the conn on the dest itself. It also has parent info.
// The func is the recvfunc, which also originates from the dest.
// We need both because relays may share the same func.
// First scan the builtin messages as most cases emanate from there.
Field Element::lookupDestField( Conn* conn, RecvFunc func )
{
	Finfo* f = const_cast< Finfo* >( cinfo()->findMsg( conn, func ) );
	if ( f )
		return Field( f, this );

	vector< Finfo* >::iterator i;
	for (i = relays_.begin(); i != relays_.end(); i++ ) {
		if ( (*i)->recvFunc() == func &&
			( *i )->inConn( conn->parent() ) == conn )
			return Field( *i, this );
	}

	return Field();
}

const string Element::path() const
{
	if (this == Element::root())
		return "/";
	string path = name();
	const Element *e;
	for (e = parent(); e != Element::root(); e = e->parent())
		path = e->name() + "/" + path;
	path = "/" + path;
	return path;
}

// Finds an element relative to the current one based on path n
Element* Element::relativeFind( const string& n )
{
	if (n == "/" || n == "")
			return this;
	unsigned long pos = n.find('/');
	string head = n.substr(0, pos);
	if (head == "") { //   starts with a slash: /bar
		return root()->relativeFind( n.substr(pos + 1, string::npos));
	}
	if (head == "..") { //   ../bar
		if ( this == Element::root() ) // oopsy.
			return 0;
		return parent()->relativeFind( n.substr(pos + 1, string::npos));
	}
	if (head == ".") { // ./bar/zod
			return relativeFind(n.substr(pos + 1, string::npos));
	}

	// We have to give up here because this element is childless
	return 0;
}


Element* Element::shallowCopy( Element* parent ) const {
	return cinfo()->create( name_, parent, this);
}

// Objects may need to do operations relative to the entire copied
// tree. Solvers are an example. This is done after the entire tree
// is built, using each object's handlePostCopy function.
void handlePostCopyOnTree(map<const Element*, Element*>& tree)
{
	map<const Element*, Element*>::iterator i;
	for (i = tree.begin(); i != tree.end(); i++)
		i->second->handlePostCopy();
}

Element* Element::deepTreeCopy(Element* pa, 
	map<const Element*, Element*> &tree) const {
	Element *e = internalDeepCopy(pa, tree);
	if ( e ) {
		duplicateMessagesOnTree(tree);
		handlePostCopyOnTree(tree);
		return e;
	}
	return 0;
}

Element* Element::deepCopy(Element* pa) const {
	map<const Element*, Element*> tree;
	return deepTreeCopy(pa, tree);
}

Element* Element::internalDeepCopy(Element* pa,
	map<const Element*, Element*>& tree) const
{
	Element* e = cinfo()->create( name_, pa, this );
	if (e)
		tree[this] = e;
	return e;
}

bool Element::descendsFrom( const Element* pa ) const
{
	if ( pa == root() || pa == this )
		return 1;

	for ( const Element* e = this; e != root(); e = e->parent() )
		if ( e == pa )
			return 1;
	return 0;
}

void Element::duplicateMessagesOnTree(map<const Element*, Element*>& tree) const
{
	map<const Element*, Element*>::iterator ti;

	for (ti = tree.begin(); ti != tree.end(); ti++) {
		// Corresponding parts of tree hould be of identical classes.
		if ( ti->first->cinfo() != ti->second->cinfo() ) {
			cout << "Error: duplicateMessgesOnTree(): Tree mismatch\n";
			return;
		}
	}
	for (ti = tree.begin(); ti != tree.end(); ti++) {
		vector< Finfo* > finfos;
		ti->first->cinfo()->listFields( finfos );
		for (unsigned int i = 0; i < finfos.size(); i++ ) {
			Element* temp = const_cast< Element* >( ti->first );
			Field f1( finfos[i], temp );
			vector< Field > dests;
			f1.dest( dests );
			// Scan dests for entries in tree but avoid kids and 
			// predefined msgs.
			// Connect up to corresponding entries in tree
			// f1 = ti->first->GetCinfo()->GetField(++i);
			// f2 = ti->second->GetCinfo()->GetField(i);
		}
	}
}

/*
void element::buildMatchingTree(Element* pa, Filter *fil,
	map<const element*, element*> &tree) const {
	if (fil->do_filter(this) && fil->do_filter(pa))
		tree[this] = pa;
}
*/

//////////////////////////////////////////////////////////////////
// Utility functions here
//////////////////////////////////////////////////////////////////
//
Element* traverseSrcToTick( Field& f )
{
	static const Cinfo* tickCinfo = Cinfo::find( "ClockTick" );
	if ( f.getElement()->cinfo()->isA( tickCinfo ) )
		return f.getElement();
	vector< Field > srcList;
	vector< Field >::iterator i;
	f.src( srcList );
	Element* ret;
	for ( i = srcList.begin(); i != srcList.end(); i++ ) {
		ret = traverseSrcToTick( *i );
		if ( ret )
				return ret;
	}
	return 0;
}
