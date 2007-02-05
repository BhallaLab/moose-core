/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

vector< Element* > Element::elementList;

Element::Element()
{
	id_ = elementList.size();
	elementList.push_back( this );
}

Element::~Element()
{
	elementList[ id_ ] = 0;
}
