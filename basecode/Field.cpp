/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

Finfo* Field::dummy_ = new FinfoDummy("dummy");

//////////////////////////////////////////////////////////////////
// Field member functions
//////////////////////////////////////////////////////////////////

Field::Field()
	: f_( dummy_ ), e_( Element::root() )
{
	;
}

Field::Field(Finfo *f, Element* e )
{
	if (f)
		f_ = f->copy();
	else
		f_ = dummy_;
	if (e)
		e_ = e;
	else
		e_ = Element::root();
}

Field::Field(const Field& other)
{
	f_ = other.f_->copy();
	e_ = other.e_;
}

// Requires an absolute path
Field::Field(const string& path)
{
	f_ = dummy_;
	e_ = Element::root();
	unsigned long pos = path.rfind( '/' );
	if ( pos == string::npos )
		return;

	string e = path.substr( 0, pos );
	if ( pos == 0 )
	string e = "/";
	string f = path.substr( pos + 1 );
	e_ = Element::root()->relativeFind( e );
	if ( !e_ ) {
		e_ = Element::root();
		return;
	}
	f_ = e_->cinfo()->field( f ).f_->copy();
}

Field::Field( Element* e, const string& finfoName )
{
	if ( e )
		f_ = e->field( finfoName ).f_->copy();
	else
		f_ = dummy_;
	e_ = e;
}

Field::~Field()
{
	f_->destroy();
}
// This operation may alter the field.
bool Field::add( Field& other ) {
	bool ret = f_->add( e_, other );
	if ( ret ) {
		refreshFinfo();
		other.refreshFinfo();
	}
	return ret;
}

// The source field calls this function to drop a msg to the dest.
// Note that the Finfo->drop() goes the other way around so that
// its virtual function can be used in messy cases like the synapse.
bool Field::drop( Field& other ) {
	bool ret = other->drop( other.e_, *this );
	// bool ret = f_->drop( e_, other );
	if ( ret ) {
		refreshFinfo();
		other.refreshFinfo();
	}
	return ret;
}

void Field::src( vector< Field >& list ) {
	f_->src( list, e_ );
}

void Field::dest( vector< Field >& list ) {
	f_->dest( list, e_ );
}

/*
unsigned long Field::nSrc() const {
	if ( f_->inConn( e_ ) )
		return f_->inConn( e_ )->nTargets();
	return 0;
}
unsigned long Field::nDest() const {
	if ( f_->outConn( e_ ) )
		return f_->outConn( e_ )->nTargets();
	return 0;
}

Field Field::src ( unsigned long i ) const 
{
	if ( e_ &&  i < f_->inConn( e_ )->nTargets() ) {
		Conn* tgt = f_->inConn( e_ )->target( i );
		if ( tgt ) {
			return tgt->parent()->lookupSrcField( tgt, f_->recvFunc() );
		} else {
			cerr << "Warning: Field::src: Null target " << i <<
				" for " << e_->name() << "." << f_->name() << "\n";
		}
	}
	cerr << "Error: Field::src( " << i <<
		"): Attempt to look up field out of range\n";
	return Field();
}

Field Field::dest ( unsigned long i ) const 
{
	if ( e_ &&  i < f_->outConn( e_ )->nTargets() ) {
		Conn* tgt = f_->outConn( e_ )->target( i );
		if ( tgt ) {
			return tgt->parent()->
				lookupDestField( tgt, f_->targetFunc( e_, i ) );
		} else {
			cerr << "Warning: Field::dest: Null target " << i <<
				" for " << e_->name() << "." << f_->name() << "\n";
		}
	}
	cerr << "Error: Field::dest( " << i <<
		"): Attempt to look up field out of range\n";
	return Field();
}
*/

const Field& Field::operator=( const Field& other )
{
	f_->destroy();
	f_ = other.f_->copy();
	e_ = other.e_;

	return *this;
}

bool Field::refreshFinfo()
{
	Field temp = e_->field( f_->name() );
	if ( temp.f_ != f_ ) {
		f_->destroy();
		f_ = temp.f_->copy();
		return 1;
	}
	return 0;
}

string Field::name() const
{
	return e_->name() + "." + f_->name();
}

string Field::path() const
{
	return e_->path() + "/" + f_->name();
}

bool Field::set( const string& value )
{
	return f_->strSet( e_, value );
}

bool Field::get( string& value )
{
	return f_->strGet( e_, value );
}

bool Field::valueComparison( const string& op, const string& value )
{
	return f_->ftype()->valueComparison( *this, op, value );
}

Finfo* Field::respondToAdd( Finfo* f )
{
	return f_->respondToAdd( e_, f );
}

//////////////////////////////////////////////////////////////////
// Other utility functions.
//////////////////////////////////////////////////////////////////

void Field::appendRelay( Finfo* f )
{
	e_->appendRelay( f );
}

/*
Field lookupSrcField( Conn* c, RecvFunc func ) 
{
	if (c) {
		Element *p = c->parent();

		// What to do about multiple hits?
		Finfo* f = p->cinfo()->findRemoteMsg( p, func );
		if (p && f)
			return Field(f, p);
	}
	return Field();
}

Field lookupDestField( Conn* c, RecvFunc func ) 
{
	if (c) {
		Element *p = c->parent();
		// What to do about multiple hits ?
		Finfo* f = const_cast< Finfo* >( p->cinfo()->findMsg( func ) );
		if (p && f)
			return Field(f, p);
	}
	return Field();
}
*/
