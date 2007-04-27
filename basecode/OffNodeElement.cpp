/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "OffNodeElement.h"

OffNodeElement::OffNodeElement( unsigned int id, unsigned int node )
		: Element( 0 ), destId_( id ), node_( 0 )
{
}

/**
 * The base Element virtual function destroys the contents of the id
 * lookup. Here we want to leave it untouched. Messy, because virtual
 * destructors are not meant to be overridden. So here we use the
 * special case where id_ is zero, in which case the base class
 * destructor does not remove the id lookup. We don't need to do
 * any stuff with the destructor because all this is handled by setting
 * the base Element id to zero.
 */
OffNodeElement::~OffNodeElement()
{
		;
}

		/// Returns the name of the element
const std::string& OffNodeElement::name( ) const
{
		static string ret = "";
		return ret;
}

		/// Sets the name of the element.
void OffNodeElement::setName( const std::string& name )
{ ; }

		/// Returns the name of the element class
const std::string& OffNodeElement::className( ) const
{
	static string ret = "";
	return ret;
}

/// Looks up the specific indexed conn
vector< Conn >::const_iterator OffNodeElement::lookupConn( unsigned int i ) const
{
		return static_cast< vector< Conn >::const_iterator >( 0 );
		// return post_->lookupConn( i );
}

/// Looks up the specific indexed conn, allows modification.
vector< Conn >::iterator OffNodeElement::lookupVariableConn( unsigned int i )
{
		return static_cast< vector< Conn >::iterator >( 0 );
		// return post_->lookupVariableConn( i );
}

		/// Finds the index of the specified conn
unsigned int OffNodeElement::connIndex( const Conn* ) const
{
		return 0;
}

unsigned int OffNodeElement::connDestRelativeIndex(
				const Conn& c, unsigned int slot ) const
{
		return 0;
}
		
		/// Returns the size of the conn vector.
unsigned int OffNodeElement::connSize() const 
{
		return 0;
}

vector< Conn >::const_iterator OffNodeElement::connSrcBegin( unsigned int src ) const
{
		return static_cast< vector< Conn >::const_iterator >( 0 );
}

vector< Conn >::const_iterator OffNodeElement::connSrcEnd( unsigned int src ) const
{
		return static_cast< vector< Conn >::const_iterator >( 0 );
}

vector< Conn >::const_iterator OffNodeElement::connSrcVeryEnd( unsigned int src ) const
{
		return static_cast< vector< Conn >::const_iterator >( 0 );
}

unsigned int OffNodeElement::nextSrc( unsigned int src ) const
{
		return 0;
}

vector< Conn >::const_iterator OffNodeElement::connDestBegin( unsigned int dest ) const
{
		return static_cast< vector< Conn >::const_iterator >( 0 );
}

vector< Conn >::const_iterator OffNodeElement::connDestEnd( unsigned int dest ) const
{
		return static_cast< vector< Conn >::const_iterator >( 0 );
}

void OffNodeElement::connect( unsigned int myConn,
			Element* targetElement, unsigned int targetConn)
{
		;
}

void OffNodeElement::disconnect( unsigned int connIndex )
{
		;
}

void OffNodeElement::deleteHalfConn( unsigned int connIndex ) 
{
		;
}


bool OffNodeElement::isMarkedForDeletion() const
{
		return 0;
}

bool OffNodeElement::isGlobal() const
{
		return 0;
}

void OffNodeElement::prepareForDeletion( bool stage )
{
		;
}

unsigned int OffNodeElement::insertConnOnSrc(
				unsigned int src, FuncList& rf,
				unsigned int dest, unsigned int nDest)
{
		return 0;
}

unsigned int OffNodeElement::insertConnOnDest( unsigned int dest, unsigned int nDest)
{
		return 0;
}

void* OffNodeElement::data() const
{
		return 0;
}

const Finfo* OffNodeElement::findFinfo( const string& name )
{
		return 0;
}

const Finfo* OffNodeElement::findFinfo( unsigned int connIndex ) const
{
		return 0;
}

unsigned int OffNodeElement::listFinfos(
			vector<	const Finfo* >& flist ) const
{
		return 0;
}

unsigned int OffNodeElement::listLocalFinfos( vector< Finfo* >& flist )
{
		return 0;
}


void OffNodeElement::addFinfo( Finfo* f )
{
		;
}

bool OffNodeElement::isConnOnSrc(
			unsigned int src, unsigned int conn ) const
{
		return 0;
}

bool OffNodeElement::isConnOnDest(
			unsigned int dest, unsigned int conn ) const
{
		return 0;
}


Element* OffNodeElement::copy( Element* parent, const string& newName )
				const
{
		return 0;
}

bool OffNodeElement::isDescendant( const Element* ancestor ) const
{
		return 0;
}


Element* OffNodeElement::innerDeepCopy( 
				map< const Element*, Element* >& tree )
				const
{
		return 0;
}

void OffNodeElement::replaceCopyPointers(
						map< const Element*, Element* >& tree )
{
		;
}


void OffNodeElement::copyMsg( map< const Element*, Element* >& tree )
{
		;
}

Element* OffNodeElement::innerCopy() const
{
		return 0;
}

bool OffNodeElement::innerCopyMsg(
				Conn& c, const Element* orig, Element* dup )
{
		return 0;
}

///////////////////////////////////////////////////////////////////
// Here are the real functions.
///////////////////////////////////////////////////////////////////

unsigned int OffNodeElement::destId() const
{
	return destId_;
}

unsigned int OffNodeElement::node() const
{
	return node_;
}

void OffNodeElement::setPost( Element* post )
{
	post_ = post;
}

Element* OffNodeElement::post() const
{
	return post_;
}

void OffNodeElement::setFieldName( const string& name )
{
	fieldName_ = name;
}

const string& OffNodeElement::fieldName() const 
{
	return fieldName_;
}
