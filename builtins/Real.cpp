/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "Real.h"

const Cinfo* Real::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Real, double > thisFinfo(
			"this",
			"Access function for entire real object.",
			&Real::setThis,
			&Real::getThis
		);

	static Finfo* realFinfos[] = {
		&thisFinfo,	// Value
	};

	static Cinfo realCinfo (
		"Real",
		Neutral::initCinfo(),
		realFinfos,
		sizeof( realFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Real >()
	);

	return &realCinfo;
}

static const Cinfo* realCinfo = Real::initCinfo();

Real::Real()
	: value_( 0.0 )
{
	;
}

Real::Real( double val )
	: value_( val )
{
	;
}

void Real::setThis( double v )
{
	value_ = v;
}

double Real::getThis() const
{
	return value_;
}

/*
const Real& Real::operator=( const Real& other )
{
	value_ = other.value_;
	return *this;
}

double Real::operator=( const Real& other )
{
	return ( value_ = other.value_ );
}

double Real::operator=( const double& other )
{
	return ( value_ = other );
}
*/
