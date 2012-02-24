/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "Annotator.h"

const Cinfo* Annotator::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Annotator, double > x(
			"x",
			"x field. Typically display coordinate x",
			&Annotator::setX,
			&Annotator::getX
		);
		static ValueFinfo< Annotator, double > y(
			"y",
			"y field. Typically display coordinate y",
			&Annotator::setY,
			&Annotator::getY
		);
		static ValueFinfo< Annotator, double > z(
			"z",
			"z field. Typically display coordinate z",
			&Annotator::setZ,
			&Annotator::getZ
		);

		static ValueFinfo< Annotator, string > notes(
			"notes",
			"A string to hold some text notes about parent object",
			&Annotator::setNotes,
			&Annotator::getNotes
		);

		static ValueFinfo< Annotator, string > color(
			"color",
			"A string to hold a text string specifying display color."
			"Can be a regular English color name, or an rgb code rrrgggbbb",
			&Annotator::setColor,
			&Annotator::getColor
		);

		static ValueFinfo< Annotator, string > textColor(
			"textColor",
			"A string to hold a text string specifying color for text label"
			"that might be on the display for this object."
			"Can be a regular English color name, or an rgb code rrrgggbbb",
			&Annotator::setTextColor,
			&Annotator::getTextColor
		);

		static ValueFinfo< Annotator, string > icon(
			"icon",
			"A string to specify icon to use for display",
			&Annotator::setIcon,
			&Annotator::getIcon
		);

	static Finfo* annotatorFinfos[] = {
		&x,	// Value
		&y,	// Value
		&z,	// Value
		&notes,	// Value
		&color,	// Value
		&textColor,	// Value
		&icon,	// Value
	};

	static Cinfo annotatorCinfo (
		"Annotator",
		Neutral::initCinfo(),
		annotatorFinfos,
		sizeof( annotatorFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Annotator >()
	);

	return &annotatorCinfo;
}

static const Cinfo* annotatorCinfo = Annotator::initCinfo();

Annotator::Annotator()
	: x_( 0.0 ), y_( 0.0 ), z_( 0.0 ), 
		notes_( "" ), color_( "white" ), textColor_( "black" ), 
		icon_( "sphere" )
{
	;
}

double Annotator::getX() const
{
	return x_;
}

void Annotator::setX( double v )
{
	x_ = v;
}

double Annotator::getY() const
{
	return y_;
}

void Annotator::setY( double v )
{
	y_ = v;
}

double Annotator::getZ() const
{
	return z_;
}

void Annotator::setZ( double v )
{
	z_ = v;
}

string Annotator::getNotes() const
{
	return notes_;
}

void Annotator::setNotes( string v )
{
	notes_ = v;
}

string Annotator::getColor() const
{
	return color_;
}

void Annotator::setColor( string v )
{
	color_ = v;
}

string Annotator::getTextColor() const
{
	return textColor_;
}

void Annotator::setTextColor( string v )
{
	textColor_ = v;
}

string Annotator::getIcon() const
{
	return icon_;
}

void Annotator::setIcon( string v )
{
	icon_ = v;
}
