/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/////////////////////////////////////////////////////////////////////////////////////
// The class TextBox is derived from the OSG tutorial and code at		   //
// http://www.cs.clemson.edu/~malloy/courses/3dgames-2007/tutor/web/text/text.html //
/////////////////////////////////////////////////////////////////////////////////////

#include "Constants.h"

#ifndef TEXTBOX_H
#define TEXTBOX_H

class TextBox
{
 public:
	TextBox();
	~TextBox();
	
	void setText( const std::string& text );
	void setFont( const std::string& font );
	void setColor( osg::Vec4d color );
	void setPosition( osg::Vec3d position ); // (x,y,z) where z is always 0; (0,0,0) is bottom-left
	void setTextSize( unsigned int size );

	osg::ref_ptr< osg::MatrixTransform > getGroup();
	std::string getText();
 private:
	osg::ref_ptr< osg::MatrixTransform > matrixTransform_; // protects HUD from transformations of parent node
	osg::ref_ptr< osg::Projection > projection_; // provides surface to render text
	osg::ref_ptr< osg::Geode > textGeode_; // hosts the text drawable
	osg::ref_ptr< osgText::Text > text_;
};

#endif // TEXTBOX_H