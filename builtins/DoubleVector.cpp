/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "DoubleVector.h"

//////////////////////////////////////////////////////////////////
// DoubleArrays are Elements that can have children. They have no 
// data fields. They equivalent to the GENESIS neutrals.
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// DoubleArray initialization
//////////////////////////////////////////////////////////////////

Finfo* DoubleArray::fieldArray_[] = 
{
	new ArrayFinfo<double, DoubleArray::valFunc >("value", "double")
};

const Cinfo DoubleArray::cinfo_(
	"DoubleArray",
	"Upinder S. Bhalla, NCBS",
	"DoubleArray class. Holds an array of doubles.",
	"Element",
	DoubleArray::fieldArray_,
	sizeof(DoubleArray::fieldArray_)/sizeof(Finfo *),
	&DoubleArray::create
);


//////////////////////////////////////////////////////////////////
// DoubleArray functions
//////////////////////////////////////////////////////////////////

DoubleArray::~DoubleArray()
{
	;
}

// In this function we take the proto's int value and its name.
Element* DoubleArray::create(
			const string& name, Element* parent, const Element* proto)
{
	DoubleArray* ret = new DoubleArray(name);
	const DoubleArray* p = dynamic_cast<const DoubleArray *>(proto);
	if (p)
		ret->value_ = p->value_;
	return ret;
	/*
	if (parent->adoptChild(ret)) {
		return ret;
	} else {
		delete ret;
		return 0;
	}
	return 0;
	*/
}
