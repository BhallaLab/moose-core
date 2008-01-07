/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <cmath>
#include <fstream>
#include "moose.h"
#include "Shell.h"
#include "ReadCell.h"
#include "../element/Neutral.h"

// Following headers are required for accessing static Cinfo initializers.
// Alternatively, one could use non-static Cinfo's below.
#include "../biophysics/Compartment.h"
#include "../biophysics/HHChannel.h"
#include "../builtins/Interpol.h"
#include "../biophysics/HHGate.h"
#include <queue>
#include "../biophysics/SynInfo.h"
#include "../biophysics/HSolveStruct.h"
#include "../biophysics/SynChan.h"
#include "../biophysics/SpikeGen.h"
#include "../biophysics/Nernst.h"
#include "../biophysics/CaConc.h"

static const Cinfo* comptCinfo = initCompartmentCinfo();
static const Cinfo* chanCinfo = initHHChannelCinfo();
static const Cinfo* gateCinfo = initHHGateCinfo();
static const Cinfo* synchanCinfo = initSynChanCinfo();
static const Cinfo* spikegenCinfo = initSpikeGenCinfo();
static const Cinfo* nernstCinfo = initNernstCinfo();
static const Cinfo* caconcCinfo = initCaConcCinfo();

ReadCell::ReadCell( const vector< double >& globalParms )
		: RM_( 10.0 ), CM_( 0.01 ), RA_( 1.0 ), EREST_ACT_( -0.065 ),
		dendrDiam( 0.0 ), aveLength( 0.0 ),
		spineSurf( 0.0 ), spineDens( 0.0 ),
		spineFreq( 0.0 ), membFactor( 0.0 ),
		numCompartments_( 0 ), numChannels_( 0 ), numOthers_( 0 ),
		cell_( 0 ), currCell_( 0 ),
		lastCompt_( 0 ), protoCompt_( 0 ),
		numProtoCompts_( 0 ), numProtoChans_( 0 ),
		numProtoOthers_( 0 ), graftFlag_( 0 ),
		polarFlag_( 0 ), relativeCoordsFlag_( 0 )
{
		Id libId;
		bool ret = lookupGet< Id, string >(
					Element::root(), "lookupChild", libId, "library" );

		if ( !ret || libId.bad() ) {
			cout << "Warning: ReadCell: No library for channels\n";
			return;
		}

		// We aren't using index 4, which is ELEAK.
		assert( globalParms.size() == 5 );

		if ( globalParms[0] != 0.0 )
			CM_ = globalParms[0];
		if ( globalParms[1] != 0.0 )
			RM_ = globalParms[1];
		if ( globalParms[2] != 0.0 )
			RA_ = globalParms[2];
		if ( globalParms[3] != 0.0 )
			EREST_ACT_ = globalParms[3];

		Element* lib = libId();

		vector< Id > chanIds;
		vector< Id >::iterator i;
		ret = get< vector< Id > >( lib, "childList", chanIds);
		assert( ret );
		for ( i = chanIds.begin(); i != chanIds.end(); i++ )
			chanProtos_.push_back( ( *i )() );
}

/**
 * The readcell function implements the old GENESIS cellreader
 * functionality. Although it is really a parser operation, I
 * put it here in biophysics because the cell format is indpendent
 * of parser and is likely to remain a legacy for a while.
 */

Element* ReadCell::start( const string& cellpath )
{
	// Warning: here is a parser dependence in the separator.
	Id cellId( cellpath, "/" );
	// There should not be an existing object of this name.
	// In the old GENESIS it deleted it. Here we will complain
	
	if ( !cellId.bad() ) {
		cout << "Warning: cell '" << cellpath << "' already exists.\n";
		return 0;
	}

	string cellname = "cell";

	string::size_type pos = cellpath.find_last_of( "/" );
	Element* cellpa;
	if ( pos == string::npos ) {
		cellpa = Element::root(); // actually should be cwe
		cellname = cellpath;
	} else if ( pos == 0 ) {
		cellpa = Element::root();
		cellname = cellpath.substr( 1 );
	} else {
		//cout << cellpath.substr( 0, pos ) << endl;
		cellId = Id( cellpath.substr( 0, pos  ), "/" );
		if ( cellId.bad() ) {
			cout << "Warning: cell path '" << cellpath <<
					"' not found.\n";
			return 0;
		}
		cellpa = cellId();
		cellname = cellpath.substr( pos + 1 );
	}
	
	if ( graftFlag_ ) {
		return Neutral::create( "Compartment",
			cellname, cellpa, Id::scratchId() );
	} else {
		return Neutral::create( "Cell",
			cellname, cellpa, Id::scratchId() );
	}
}

void ReadCell::read( const string& filename, const string& cellpath )
{
	ifstream fin( filename.c_str() );
	cell_ = start( cellpath );
	if ( !cell_ ) return;
	currCell_ = cell_;

	string line;
	unsigned int lineNum = 0;
	string::size_type pos;
	ParseStage parseMode = DATA;
	while ( getline( fin, line ) ) {
		lineNum++;
		if ( line.length() == 0 )
				continue;
		pos = line.find_first_not_of( "\t " );
		if ( pos == string::npos )
				continue;
		else
			line = line.substr( pos );
		if ( line.substr( 0, 2 ) == "//" )
				continue;
		if ( (pos = line.find("//")) != string::npos ) 
			line = line.substr( 0, pos );
		if ( line.substr( 0, 2 ) == "/*" ) {
				parseMode = COMMENT;
		} else if ( line.find( "*/" ) != string::npos ) {
				parseMode = DATA;
				continue;
		} else if ( line[0] == '*' ) {
				parseMode = SCRIPT;
		}

		if ( parseMode == COMMENT ) {
			pos = line.find( "*/" );
			if ( pos != string::npos ) {
				parseMode = DATA;
				if ( line.length() > pos + 2 )
					line = line.substr( pos + 2 );
			}
		}
		if ( parseMode == DATA )
				readData( line, lineNum );
		else if ( parseMode == SCRIPT ) {
				readScript( line, lineNum );
				parseMode = DATA;
		}
	}
	
	cout << filename << " read: " <<
			numCompartments_ << " compartments, " << 
			numChannels_ << " channels, " << 
			numOthers_ << " others\n";
}

void ReadCell::readData( const string& line, unsigned int lineNum )
{
	vector< string > argv;
	parseString( line, argv, "\t " ); 
	if ( argv.size() < 6 ) {
			cout << "Readfile: Error on line " << lineNum << endl;
			cout << "Too few arguments in line: " << argv.size() <<
					", should be > 6\n";
			return;
	}
	string name = argv[0];
	string parent = argv[1];
	double x = 1.0e-6 * atof( argv[2].c_str() );
	double y = atof( argv[3].c_str() );
	double z = atof( argv[4].c_str() );
	double d = 1.0e-6 * atof( argv[5].c_str() );
	if ( polarFlag_ ) {
		double r = x;
		double theta = y * PI / 180.0;
		double phi = z * PI / 180.0;
		x = r * sin( phi ) * cos ( theta );
		y = r * sin( phi ) * sin ( theta );
		z = r * cos( phi );
	} else {
		y *= 1.0e-6;
		z *= 1.0e-6;
	}

	double length;
	Element* compt =
		buildCompartment( name, parent, x, y, z, d, length, argv );
	buildChannels( compt, argv, d, length );
}

Element* ReadCell::buildCompartment( 
				const string& name, const string& parent,
				double x, double y, double z, double d, double& length,
				vector< string >& argv )
{
	static const Finfo* axial = comptCinfo->findFinfo( "axial" );
	static const Finfo* raxial = comptCinfo->findFinfo( "raxial" );
	static const Finfo* xFinfo = comptCinfo->findFinfo( "x" );
	static const Finfo* yFinfo = comptCinfo->findFinfo( "y" );
	static const Finfo* zFinfo = comptCinfo->findFinfo( "z" );
	static const Finfo* dFinfo = comptCinfo->findFinfo( "diameter" );
	static const Finfo* lengthFinfo = comptCinfo->findFinfo( "length" );
	static const Finfo* RmFinfo = comptCinfo->findFinfo( "Rm" );
	static const Finfo* CmFinfo = comptCinfo->findFinfo( "Cm" );
	static const Finfo* RaFinfo = comptCinfo->findFinfo( "Ra" );
	static const Finfo* initVmFinfo = comptCinfo->findFinfo( "initVm" );
	static const Finfo* EmFinfo = comptCinfo->findFinfo( "Em" );
	static const Finfo* VmFinfo = comptCinfo->findFinfo( "Vm" );

	Element* pa;
	if ( parent == "." ) { // Shorthand: use the previous compartment.
			pa = lastCompt_;
	} else if ( parent == "none" || parent == "nil" ) {
			pa = Element::root();
	} else {
		Id paId( currCell_->id().path() + "/" + parent );
		if ( paId.bad() ) {
			cout << "Error: ReadCell: could not find parent compt '" <<
					parent << "' for child '" << name << "'\n";
			return 0;
		}
		pa = paId();
	}
	if ( pa == 0 )
		return 0;
	Id childId;
	bool ret = lookupGet< Id, string >(
				currCell_, "lookupChild", childId, name );
	assert( ret );
	if ( !childId.bad() ) {
		cout << "Error: ReadCell: duplicate child on parent compt '" <<
				parent << "' for child '" << name << "'\n";
		return 0;
	}

	Element* compt;
	if ( graftFlag_ && ( parent == "none" || parent == "nil" ) ) {
		compt = currCell_;
	} else {
		if ( protoCompt_ ) {
			compt = protoCompt_->copy( currCell_, name );
			numCompartments_ += numProtoCompts_;
			numChannels_ += numProtoChans_;
			numOthers_ += numProtoOthers_;
		} else {
			compt = Neutral::create( "Compartment",
				name, currCell_, Id::scratchId() );
			if ( !graftFlag_ )
				++numCompartments_;
		}
	}
	lastCompt_ = compt;

	if ( pa != Element::root() ) {
		double dx, dy, dz;
		get< double >( pa, xFinfo, dx );
		get< double >( pa, yFinfo, dy );
		get< double >( pa, zFinfo, dz );
		if ( relativeCoordsFlag_ == 1 ) {
			x += dx;
			y += dy;
			z += dz;
		}
		dx = x - dx;
		dy = y - dy;
		dz = z - dz;

		length = sqrt( dx * dx + dy * dy + dz * dz );
		axial->add( pa, compt, raxial );
	} else {
		length = sqrt( x * x + y * y + z * z ); 
		// or it coult be a sphere.
	}

	set< double >( compt, xFinfo, x );
	set< double >( compt, yFinfo, y );
	set< double >( compt, zFinfo, z );
	set< double >( compt, dFinfo, d );

	set< double >( compt, lengthFinfo, length );
	double Rm = RM_ / ( d * length * PI );
	set< double >( compt, RmFinfo, Rm );
	double Ra = RA_ * length * 4.0 / ( d * d * PI );
	set< double >( compt, RaFinfo, Ra );
	double Cm = CM_ * ( d * length * PI );
	set< double >( compt, CmFinfo, Cm );
	set< double >( compt, initVmFinfo, EREST_ACT_ );
	set< double >( compt, EmFinfo, EREST_ACT_ );
	set< double >( compt, VmFinfo, EREST_ACT_ );

	return compt;
}

void ReadCell::readScript( const string& line, unsigned int lineNum )
{
	vector< string > argv;
	parseString( line, argv, "\t " ); 

	if ( argv[0] == "*cartesian" ) {
		polarFlag_ = 0;
		return;
	}
	if ( argv[0] == "*polar" ) {
		polarFlag_ = 1;
		return;
	}
	if ( argv[0] == "*relative" ) {
		relativeCoordsFlag_ = 1;
		return;
	}
	if ( argv[0] == "*absolute" ) {
		relativeCoordsFlag_ = 0;
		return;
	}

	if ( argv[0] == "*set_global" ) {
		if ( argv.size() != 3 ) {
			cout << "Error: readCell: Bad line: " << lineNum <<
					": " << line << endl;
			return;
		}
		if ( argv[1] == "RM" )
				RM_ = atof( argv[2].c_str() );
		if ( argv[1] == "RA" )
				RA_ = atof( argv[2].c_str() );
		if ( argv[1] == "CM" )
				CM_ = atof( argv[2].c_str() );
		if ( argv[1] == "EREST_ACT" )
				EREST_ACT_ = atof( argv[2].c_str() );
	}

	if ( argv[0] == "*start_cell" ) {
		if ( argv.size() == 1 ) {
			graftFlag_ = 0;
			currCell_ = cell_;
		} else if ( argv.size() == 2 ) {
			graftFlag_ = 1;
			currCell_ = start( argv[1] );
		} else {
			cout << "Error: readCell: Bad line: " << lineNum <<
					": " << line << endl;
			return;
		}
	}

	if ( argv[0] == "*compt" ) {
		if ( argv.size() != 2 ) {
			cout << "Error: readCell: Bad line: " << lineNum <<
					": " << line << endl;
			return;
		}

		Id protoId( argv[1] );
		if ( protoId.bad() ) {
			cout << "Error: readCell: Bad path: " << lineNum <<
					": " << line << endl;
			return;
		}
		
		protoCompt_ = protoId();
		countProtos( );
		return;
	}

	if ( argv[0] == "*makeproto" ) {
		return; // Should traverse tree below and drop process messages.
	}
}

Element* ReadCell::findChannel( const string& name )
{
	vector< Element* >::iterator i;
	for ( i = chanProtos_.begin(); i != chanProtos_.end(); i++ )
		if ( (*i)->name() == name )
			return (*i);
	return 0;
}

double calcSurf( double len, double dia )
{
	double area = 0.0;
	if ( len == 0.0 ) // Spherical
		area = dia * dia * PI;
	else
		area = len * dia * PI;

	return area;
}

bool ReadCell::buildChannels( Element* compt, vector< string >& argv,
				double diameter, double length)
{
	if ( ( argv.size() % 2 ) == 1 ) {
		cout << "Error: readCell: Bad number of arguments in channel list\n";
		return 0;
	}
	for ( unsigned int j = 6; j < argv.size(); j++ ) {
		// Here we explicitly set compt fields by scaling from the 
		// specific value applied here.
		string chan = argv[j];

		double value = atof( argv[ ++j ].c_str() );
		if ( chan == "RA" ) {
			double temp;
			if ( length >= 0.0 ) // spherical flag. Assume length = dia.
				temp = 8.0 * value / (diameter * PI );
			else
				temp = 4.0 * value * length / (diameter * diameter * PI );
			set< double >( compt, "Ra", temp );
		} else if ( chan == "RM" ) {
			set< double >( compt, "Rm", value * calcSurf( length, diameter ) );
		} else if ( chan == "CM" ) {
			set< double >( compt, "Cm", value * calcSurf( length, diameter ) );
		} else if ( chan == "Rm" ) {
			set< double >( compt, "Rm", value );
		} else if ( chan == "Ra" ) {
			set< double >( compt, "Ra", value );
		} else if ( chan == "Cm" ) {
			set< double >( compt, "Cm", value );
		} else {
			Element* chanElm = findChannel( chan );
			if ( chanElm == 0 ) {
				cout << "Error: readCell: Channel '" << chan <<
						"' not found\n";
				continue;
			}

			if ( !addChannel( compt, chanElm, value, diameter, length ) )
				continue;
		}
	}
	return 1;
}


bool ReadCell::addChannel( 
			Element* compt, Element* proto, double value, 
			double dia, double length )
{

	Element* chan = proto->copy( compt, "" );
	assert( chan != 0 );

	if ( addHHChannel( compt, chan, value, dia, length ) ) return 1;
	if ( addSynChan( compt, chan, value, dia, length ) ) return 1;
	if ( addSpikeGen( compt, chan, value, dia, length ) ) return 1;
	if ( addCaConc( compt, chan, value, dia, length ) ) return 1;
	if ( addNernst( compt, chan, value ) ) return 1;
	return 0;
}

bool ReadCell::addHHChannel( 
		Element* compt, Element* chan, 
		double value, double dia, double length )
{
	static const Finfo* chanSrcFinfo = comptCinfo->findFinfo( "channel" );
	static const Finfo* hhChanDestFinfo = chanCinfo->findFinfo( "channel" );
	static const Finfo* gbarFinfo = chanCinfo->findFinfo( "Gbar" );
	
	if ( chan->className() == "HHChannel" ) {
		bool ret = chanSrcFinfo->add( compt, chan, hhChanDestFinfo );
		assert( ret );
			
		if ( value > 0 ) {
			value *= dia * length * PI;
		} else {
			value = - value;
		}

		if ( !graftFlag_ )
			++numChannels_;
		return set< double >( chan, gbarFinfo, value );
	}
	return 0;
}

bool ReadCell::addSynChan( 
		Element* compt, Element* chan, 
		double value, double dia, double length )
{
	static const Finfo* chanSrcFinfo = 
		comptCinfo->findFinfo( "channel" );
	static const Finfo* synChanDestFinfo = 
		synchanCinfo->findFinfo( "channel" );
	static const Finfo* synGbarFinfo = 
		synchanCinfo->findFinfo( "Gbar" );

	if ( chan->className() == "SynChan" ) {
		bool ret = chanSrcFinfo->add( compt, chan, synChanDestFinfo );
		assert( ret );
		
		if ( value > 0 ) {
			value *= dia * length * PI;
		} else {
			value = - value;
		}

		if ( !graftFlag_ )
			++numChannels_;
		return set< double >( chan, synGbarFinfo, value );
	}
	return 0;
}

bool ReadCell::addSpikeGen( 
		Element* compt, Element* chan, 
		double value, double dia, double length )
{
	static const Finfo* vmSrcFinfo =
		comptCinfo->findFinfo( "VmSrc" );
	static const Finfo* vmDestFinfo =
		spikegenCinfo->findFinfo( "Vm" );
	static const Finfo* threshFinfo =
		spikegenCinfo->findFinfo( "threshold" );
	if ( chan->className() == "SpikeGen" ) {
		bool ret = vmSrcFinfo->add( compt, chan, vmDestFinfo  );
		assert( ret );
		if ( !graftFlag_ )
			++numOthers_;
		return set< double >( chan, threshFinfo, value );
	}
	return 0;
}


// This has tricky messaging, need to complete later.
bool ReadCell::addCaConc( 
		Element* compt, Element* chan, 
		double value, double dia, double length )
{
		/*
	static const Finfo* concSrcFinfo = caconcCinfo->findFinfo( "concSrc" );
	static const Finfo* currentFinfo = caconcCinfo->findFinfo( "current" );
		*/
	static const Finfo* bFinfo = caconcCinfo->findFinfo( "B" );
	if ( chan->className() == "CaConc" ) {
		// assert( vmSrcFinfo->add( compt, chan, vmDestFinfo  ) );
		if ( !graftFlag_ )
			++numOthers_;
		return set< double >( chan, bFinfo, value );
	}
	return 0;
}

// This has tricky messaging, need to complete later.
bool ReadCell::addNernst( 
		Element* compt, Element* chan, double value )
{
	if ( !graftFlag_ )
		++numOthers_;
	return 0;
}

/**
 * Count elements under a tree.
 */
void ReadCell::countProtos( )
{
	if ( protoCompt_ == 0 )
		return;
	
	numProtoCompts_ = 1; // protoCompt_ itself
	numProtoChans_ = 0;
	numProtoOthers_ = 0;

	vector< vector< Id > > cstack;
	cstack.push_back( Neutral::getChildList( protoCompt_ ) );
	while ( !cstack.empty() ) {
		vector< Id >& child = cstack.back();
		
		if ( child.empty() ) {
			cstack.pop_back();
			if ( !cstack.empty() )
				cstack.back().pop_back();
		} else {
			const Id& curr = child.back();
			const Cinfo* currCinfo = curr()->cinfo();
			
			if ( currCinfo->isA( comptCinfo ) )
				++numProtoCompts_;
			else if ( currCinfo->isA( chanCinfo ) ||
			          currCinfo->isA( synchanCinfo ) )
				++numProtoChans_;
			else if ( currCinfo->isA( spikegenCinfo ) ||
			          currCinfo->isA( caconcCinfo ) ||
			          currCinfo->isA( nernstCinfo ) )
				++numProtoOthers_;
			
			cstack.push_back( Neutral::getChildList( curr() ) );
		}
	}
}
