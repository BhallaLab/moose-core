/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2015 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "../utility/Vec.h"
#include "SwcSegment.h"
#include "Spine.h"
#include "Neuron.h"

const Cinfo* Spine::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< Spine, double > shaftLength (
			"shaftLength",
			"Length of spine shaft..",
			&Spine::setShaftLength,
			&Spine::getShaftLength
		);
		static ElementValueFinfo< Spine, double > shaftDiameter (
			"shaftDiameter",
			"Diameter of spine shaft..",
			&Spine::setShaftDiameter,
			&Spine::getShaftDiameter
		);
		static ElementValueFinfo< Spine, double > headLength (
			"headLength",
			"Length of spine head...",
			&Spine::setHeadLength,
			&Spine::getHeadLength
		);
		static ElementValueFinfo< Spine, double > headDiameter (
			"headDiameter",
			"Diameter of spine head..",
			&Spine::setHeadDiameter,
			&Spine::getHeadDiameter
		);
		static ElementValueFinfo< Spine, double > totalLength (
			"totalLength",
			"Length of entire spine",
			&Spine::setTotalLength,
			&Spine::getTotalLength
		);
		static ElementValueFinfo< Spine, double > angle (
			"angle",
			"angle of spine around shaft. Longitude. 0 is away from soma",
			&Spine::setAngle,
			&Spine::getAngle
		);
		static ElementValueFinfo< Spine, double > inclination (
			"inclination",
			"inclination of spine with ref to shaft. Normal is 0.",
			&Spine::setInclination,
			&Spine::getInclination
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////

	static Finfo* spineFinfos[] = {
		&shaftLength,		// Readonly Value
		&shaftDiameter,		// Readonly Value
		&headLength,		// Readonly Value
		&headDiameter,		// Readonly Value
		&totalLength,		// Readonly Value
	};

	static string doc[] = 
	{
			"Name", "Spine",
			"Author", "Upi Bhalla",
			"Description", "Spine wrapper, used to change its morphology "
			"typically by a message from an adaptor. The Spine class "
			"takes care of a lot of resultant scaling to electrical, "
			"chemical, and diffusion properties. "
	};
	static Dinfo< Spine > dinfo;
	static Cinfo spineCinfo (
		"Spine",
		Neutral::initCinfo(),
		spineFinfos,
		sizeof( spineFinfos ) / sizeof ( Finfo* ),
		&dinfo,
		doc,
		sizeof(doc)/sizeof( string ),
		true // This IS a FieldElement, not be be created directly.
	);

	return &spineCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* spineCinfo = Spine::initCinfo();

Spine::Spine()
	: parent_( 0 )
{;}

Spine::Spine( const Neuron* parent )
	: parent_( parent )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

double Spine::getShaftLength( const Eref& e ) const
{
	const vector< Id >& sl = parent_->spineIds( e.fieldIndex() );
	if ( sl.size() > 0 && 
					sl[0].element()->cinfo()->isA( "CompartmentBase" ) )
		return Field< double >::get( sl[0], "length" );
	return 0.0;
}

void Spine::setShaftLength( const Eref& e, double len )
{
	vector< Id > sl = parent_->spineIds( e.fieldIndex() );
	if ( sl.size() > 1 && 
			sl[0].element()->cinfo()->isA( "CompartmentBase" ) ) 
	{ 
		double origDia = Field< double >::get( sl[0], "diameter" );
		double dx = Field< double >::get( sl[0], "x" );
		double dy = Field< double >::get( sl[0], "y" );
		double dz = Field< double >::get( sl[0], "z" );
		SetGet2< double, double >::set( 
			sl[0], "setGeomAndElec", len, origDia );

		dx = Field< double >::get( sl[0], "x" ) - dx;
		dy = Field< double >::get( sl[0], "y" ) - dy;
		dz = Field< double >::get( sl[0], "z" ) - dz;

		SetGet3< double, double, double >::set( sl[1], "displace",
			dx, dy, dz );
		// Here we've set the electrical and geometrical stuff. Now to
		// do the diffusion. Chem doesn't come into the picture for the
		// spine shaft.	
		// Assume the scaleDiffusion function propagates changes into the
		// VoxelJunctions used by the Dsolve.
		parent_->scaleShaftDiffusion( e.fieldIndex(), len, origDia );
	}
}

double Spine::getShaftDiameter( const Eref& e ) const
{
	vector< Id > sl = parent_->spineIds( e.fieldIndex() );
	if ( sl.size() > 0 && 
				sl[0].element()->cinfo()->isA( "CompartmentBase" ) )
		return Field< double >::get( sl[0], "diameter" );
	return 0.0;
}

void Spine::setShaftDiameter( const Eref& e, double dia )
{
	vector< Id > sl = parent_->spineIds( e.fieldIndex() );
	if ( sl.size() > 1 && 
					sl[0].element()->cinfo()->isA( "CompartmentBase") )
	{
		double origLen = Field< double >::get( sl[0], "length" );
		SetGet2< double, double >::set( 
			sl[0], "setGeomAndElec", origLen, dia );
		// Dia is changing, leave the coords alone.
		parent_->scaleShaftDiffusion( e.fieldIndex(), origLen, dia );
	}
}

double Spine::getHeadLength( const Eref& e ) const
{
	vector< Id > sl = parent_->spineIds( e.fieldIndex() );
	if ( sl.size() > 1 && 
					sl[1].element()->cinfo()->isA( "CompartmentBase" ) )
		return Field< double >::get( sl[1], "length" );
	return 0.0;
}

void Spine::setHeadLength( const Eref& e, double len )
{
	vector< Id > sl = parent_->spineIds( e.fieldIndex() );
	if ( sl.size() > 1 && 
					sl[1].element()->cinfo()->isA( "CompartmentBase") ) 
	{
		double origDia = Field< double >::get( sl[1], "diameter" );
		double origLen = Field< double >::get( sl[1], "length" );
		SetGet2< double, double >::set( 
			sl[1], "setGeomAndElec", len, origDia );
		// Here we've set the electrical and geometrical stuff. Now to
		// do the diffusion.
		// Assume the scaleDiffusion function propagates changes into the
		// VoxelJunctions used by the Dsolve.
		parent_->scaleHeadDiffusion( e.fieldIndex(), len, origDia );
		// Now scale the chem stuff. The PSD mesh is assumed to scale only
		// with top surface area of head, so it is not affected here.
		parent_->scaleBufAndRates( e.fieldIndex(), len/origLen, 1.0 );
	}
}

double Spine::getHeadDiameter( const Eref& e ) const
{
	vector< Id > sl = parent_->spineIds( e.fieldIndex() );
	if ( sl.size() > 1 && 
			sl[1].element()->cinfo()->isA( "CompartmentBase" ) )
		return Field< double >::get( sl[1], "diameter" );
	return 0.0;
}

void Spine::setHeadDiameter( const Eref& e, double dia )
{
	vector< Id > sl = parent_->spineIds( e.fieldIndex() );
	if ( sl.size() > 1 && 
			sl[0].element()->cinfo()->isA( "CompartmentBase") )
	{
		double origLen = Field< double >::get( sl[1], "length" );
		double origDia = Field< double >::get( sl[1], "diameter" );
		SetGet2< double, double >::set( 
			sl[1], "setGeomAndElec", origLen, dia );
		parent_->scaleHeadDiffusion( e.fieldIndex(), origLen, dia );
		parent_->scaleBufAndRates( e.fieldIndex(), 
						1.0, dia/origDia );
	}
}

double Spine::getTotalLength( const Eref& e ) const
{
	return getHeadLength( e ) + getShaftLength( e );
}

void Spine::setTotalLength( const Eref& e, double len )
{	
	double shaftLen = getShaftLength( e );
	double headLen = getHeadLength( e );
	double totLen = shaftLen + headLen;
	setShaftLength( e, shaftLen * len / totLen );
	setHeadLength( e, headLen * len / totLen );
}

double Spine::getAngle( const Eref& e ) const
{
	return 0;
}

void Spine::setAngle( const Eref& e, double theta )
{	
	;
}

double Spine::getInclination( const Eref& e ) const
{
	return 0;
}

void Spine::setInclination( const Eref& e, double theta )
{	
	;
}
