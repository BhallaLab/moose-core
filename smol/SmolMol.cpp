/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "SmolHeader.h"
#include "SmolMol.h"
#include "Pool.h"
#include "ElementValueFinfo.h"
#include "DataHandlerWrapper.h"

const Cinfo* SmolMol::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< SmolMol, double > x(
			"x",
			"X coordinate of molecule",
			&SmolMol::setX,
			&SmolMol::getX
		);

		static ElementValueFinfo< SmolMol, double > y(
			"y",
			"Y coordinate of molecule",
			&SmolMol::setY,
			&SmolMol::getY
		);

		static ElementValueFinfo< SmolMol, double > z(
			"z",
			"Z coordinate of molecule",
			&SmolMol::setZ,
			&SmolMol::getZ
		);

		static ElementValueFinfo< SmolMol, MolecState > state(
			"state",
			"State of molecule: cytosolic, inner face of memb, outer face, etc",
			&SmolMol::setState,
			&SmolMol::getState
		);

		static ReadOnlyElementValueFinfo< SmolMol, unsigned int > species(
			"species",
			"Species identifer for this molecule",
			&SmolMol::getSpecies
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		/*
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< SmolMol >( &SmolMol::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< SmolMol >( &SmolMol::reinit ) );

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo reacDest( "reacDest",
			"Handles reaction input",
			new OpFunc2< SmolMol, double, double >( &SmolMol::reac )
		);

		static DestFinfo setSize( "setSize",
			"Separate finfo to assign size, should only be used by compartment."
			"Defaults to SI units of volume: m^3",
			new EpFunc1< SmolMol, double >( &SmolMol::setSize )
		);
		*/

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////
		/*
		static SrcFinfo1< double > nOut( 
				"nOut", 
				"Sends out # of molecules on each timestep"
		);
		*/

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

		/*
		static Finfo* reacShared[] = {
			&reacDest, &nOut
		};
		static SharedFinfo reac( "reac",
			"Connects to reaction",
			reacShared, sizeof( reacShared ) / sizeof( const Finfo* )
		);
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);
		*/

	static Finfo* smolMolFinfos[] = {
		&x,				// Value
		&y,				// Value
		&z,				// Value
		&state,			// Value
		&species,		// ReadOnlyValue
	};

	static Cinfo smolMolCinfo (
		"SmolMol",
		Neutral::initCinfo(),
		smolMolFinfos,
		sizeof( smolMolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SmolMol >()
	);

	return &smolMolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* smolMolCinfo = SmolMol::initCinfo();

SmolMol::SmolMol()
{;}

SmolMol::~SmolMol()
{;}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void SmolMol::setX( const Eref& e, const Qinfo* q, double v )
{
	// S_[ convertIdToMolIndex( e.id() ) ] = v;
}

double SmolMol::getX( const Eref& e, const Qinfo* q ) const
{
	return 0.0;
	// return S_[ convertIdToMolIndex( e.id() ) ];
}

void SmolMol::setY( const Eref& e, const Qinfo* q, double v )
{
	// S_[ convertIdToMolIndex( e.id() ) ] = v;
}

double SmolMol::getY( const Eref& e, const Qinfo* q ) const
{
	return 0.0;
	// return S_[ convertIdToMolIndex( e.id() ) ];
}

void SmolMol::setZ( const Eref& e, const Qinfo* q, double v )
{
	// S_[ convertIdToMolIndex( e.id() ) ] = v;
}

double SmolMol::getZ( const Eref& e, const Qinfo* q ) const
{
	return 0.0;
	// return S_[ convertIdToMolIndex( e.id() ) ];
}

void SmolMol::setState( const Eref& e, const Qinfo* q, MolecState v )
{
	// S_[ convertIdToMolIndex( e.id() ) ] = v;
}

MolecState SmolMol::getState( const Eref& e, const Qinfo* q ) const
{
	return MSsoln;
	// return S_[ convertIdToMolIndex( e.id() ) ];
}

unsigned int SmolMol::getSpecies( const Eref& e, const Qinfo* q ) const
{
//	return species_[ convertIdToMolIndex( e.id() ) ];
	return 0;
}

/////////////////////////////////////////////////////////////////////////
// Zombification operations.
/////////////////////////////////////////////////////////////////////////

void SmolMol::zombify( Element* solver, Element* orig )
{
	cout << "zombifying " << orig->getName() << endl;
}

void SmolMol::unzombify( Element* zombie )
{
}

/////////////////////////////////////////////////////////////////////////
// Utility func for MolecState
/////////////////////////////////////////////////////////////////////////
ostream& operator <<( ostream&s, const MolecState& ms )
{
	int temp = ms;
	// s << reinterpret_cast< int >( ms );
	s << temp;
	return s;
}

istream& operator >>( istream&s, MolecState& ms )
{
	int temp;
	s >> temp;
	// ms = temp;
	// ms = reinterpret_cast< MolecState >( temp );
	return s;
}
