/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Projection>
#include <osg/MatrixTransform>
#include <osg/Transform>
#include <osgText/Text>

#include "TextBox.h"

TextBox::TextBox()
	:
	matrixTransform_( new osg::MatrixTransform ),
	projection_( new osg::Projection ),
	textGeode_( new osg::Geode ),
	text_( new osgText::Text )
{
	matrixTransform_->setReferenceFrame( osg::Transform::ABSOLUTE_RF );
	matrixTransform_->addChild( projection_ );
	
	projection_->setMatrix( osg::Matrix::ortho2D( 0, WINDOW_WIDTH, 0, WINDOW_HEIGHT ) );
	projection_->addChild( textGeode_ );

	textGeode_->addDrawable( text_ );
	textGeode_->setDataVariance( osg::Object::STATIC );
	
	text_->setAxisAlignment( osgText::Text::SCREEN );
	text_->setText( "..." );
}

TextBox::~TextBox()
{
	matrixTransform_ = NULL;
	projection_ = NULL;
	textGeode_ = NULL;
	text_ = NULL;
}

void TextBox::setText( const std::string& text )
{
	text_->setText( text );
}

void TextBox::setFont( const std::string& font )
{
	text_->setFont( font );
}

void TextBox::setColor( osg::Vec4d color )
{
	text_->setColor( color );
}

void TextBox::setPosition( osg::Vec3d position )
{
	text_->setPosition( position );
}

void TextBox::setTextSize( unsigned int size )
{
	text_->setCharacterSize( size );
}

osg::ref_ptr< osg::MatrixTransform > TextBox::getGroup()
{
	return matrixTransform_;
}

std::string TextBox::getText()
{
	return text_->getText().createUTF8EncodedString();
}
