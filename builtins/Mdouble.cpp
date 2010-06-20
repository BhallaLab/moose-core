/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "Mdouble.h"

const Cinfo* Mdouble::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Mdouble, double > thisFinfo(
			"this",
			"Access function for entire Mdouble object.",
			&Mdouble::setThis,
			&Mdouble::getThis
		);
		static ValueFinfo< Mdouble, double > valueFinfo(
			"value",
			"Access function for value field of Mdouble object,"
			"which happens also to be the entire contents of the object.",
			&Mdouble::setThis,
			&Mdouble::getThis
		);

	static Finfo* mDoubleFinfos[] = {
		&thisFinfo,	// Value
		&valueFinfo,	// Value
	};

	static Cinfo mDoubleCinfo (
		"Mdouble",
		Neutral::initCinfo(),
		mDoubleFinfos,
		sizeof( mDoubleFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Mdouble >()
	);

	return &mDoubleCinfo;
}

static const Cinfo* mDoubleCinfo = Mdouble::initCinfo();

Mdouble::Mdouble()
	: value_( 0.0 )
{
	;
}

Mdouble::Mdouble( double val )
	: value_( val )
{
	;
}

void Mdouble::setThis( double v )
{
	value_ = v;
}

double Mdouble::getThis() const
{
	return value_;
}

/*
const Mdouble& Mdouble::operator=( const Mdouble& other )
{
	value_ = other.value_;
	return *this;
}

double Mdouble::operator=( const Mdouble& other )
{
	return ( value_ = other.value_ );
}

double Mdouble::operator=( const double& other )
{
	return ( value_ = other );
}
*/
