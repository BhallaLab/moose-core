/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "ArrayWrapperElement.h"
#include<sstream>

string itos(int i)	// convert int to string
{
	stringstream s;
	s << i;
	return s.str();
}

ArrayWrapperElement::ArrayWrapperElement( Element* arrayElement, unsigned int index )
	: SimpleElement( Id(), arrayElement->name() + "[" + itos(index) + "]" , 0, 0, 0 ),
	arrayElement_( arrayElement ), index_( index )
{
	//arrayElement->id().assignIndex(index);
	//id().setIndex(index);
	assert( index < arrayElement->numEntries() );
}


ArrayWrapperElement::~ArrayWrapperElement( )
{
	;
}

void* ArrayWrapperElement::data() const
{
	return static_cast< void* >(
		static_cast< char* >( arrayElement_->data( ) )
		+ index_ * arrayElement_->cinfo()->ftype()->size() );
}

const Finfo* ArrayWrapperElement::findFinfo( const string& name )
{
	return arrayElement_->findFinfo( name );
}

unsigned int ArrayWrapperElement::listFinfos( 
				vector< const Finfo* >& flist ) const
{
	return arrayElement_->listFinfos( flist );
}


unsigned int ArrayWrapperElement::numEntries( ) const
{
	return 0;
	//return arrayElement_->numEntries();
}

unsigned int ArrayWrapperElement::index( ) const
{
	return index_;
}

Id ArrayWrapperElement::id( ) const
{
	return arrayElement_->id().assignIndex(index_);
}

const std::string& ArrayWrapperElement::className( ) const
{
	return arrayElement_->className();
}

vector< Conn >::const_iterator
	ArrayWrapperElement::connSrcBegin( unsigned int src ) const
{
	return arrayElement_->connSrcBegin(src);
}

vector< Conn >::const_iterator
	ArrayWrapperElement::connSrcEnd( unsigned int src ) const
{
	return arrayElement_->connSrcEnd(src);
}

const Finfo* ArrayWrapperElement::getThisFinfo( ) const
{
	return arrayElement_->getThisFinfo( );
}

vector< Conn >::const_iterator
	ArrayWrapperElement::connSrcVeryEnd( unsigned int src ) const
{
	return arrayElement_->connSrcVeryEnd(src);
}

void ArrayWrapperElement::getElementPosition(int& nx, int& ny){
	(static_cast<ArrayElement *> (arrayElement_))->getElementPosition(nx, ny, index_);
}

void ArrayWrapperElement::getElementCoordinates(double& xcoord, double& ycoord){
	(static_cast<ArrayElement *> (arrayElement_))->getElementCoordinates(xcoord, ycoord, index_);
}
		









