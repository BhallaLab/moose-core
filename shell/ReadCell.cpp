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
#include <sstream>
#include "moose.h"
#include "Shell.h"
#include "ReadCell.h"
#include "../element/Neutral.h"
#include "../utility/utility.h"

// Following headers are required for accessing static Cinfo initializers.
// Alternatively, one could use non-static Cinfo's below.
#include "../biophysics/Compartment.h"
#include "../biophysics/HHChannel.h"
#include "../builtins/Interpol.h"
#include "../biophysics/HHGate.h"
#include <queue>
#include "../biophysics/SynInfo.h"
#include "../biophysics/SynChan.h"
#include "../biophysics/SpikeGen.h"
#include "../biophysics/Nernst.h"
#include "../biophysics/CaConc.h"


// Defined in GenesisParserWrapper.cpp
extern void do_add( int argc, const char** const argv, Id s );

static const Cinfo* comptCinfo = initCompartmentCinfo();
static const Cinfo* chanCinfo = initHHChannelCinfo();
static const Cinfo* synchanCinfo = initSynChanCinfo();
static const Cinfo* spikegenCinfo = initSpikeGenCinfo();
static const Cinfo* nernstCinfo = initNernstCinfo();
static const Cinfo* caconcCinfo = initCaConcCinfo();
double calcSurf(double, double);

ReadCell::ReadCell(
	const vector< double >& globalParms,
	IdGenerator idGen )
	:
	idGen_( idGen ), RM_( 10.0 ), CM_( 0.01 ), RA_( 1.0 ), EREST_ACT_( -0.065 ), ELEAK_( -0.065 ),
	dendrDiam( 0.0 ), aveLength( 0.0 ),
	spineSurf( 0.0 ), spineDens( 0.0 ),
	spineFreq( 0.0 ), membFactor( 0.0 ),
	numCompartments_( 0 ), numChannels_( 0 ), numOthers_( 0 ),
	cell_( 0 ), currCell_( 0 ),
	lastCompt_( 0 ), protoCompt_( 0 ),
	numProtoCompts_( 0 ), numProtoChans_( 0 ),
	numProtoOthers_( 0 ), graftFlag_( 0 ),
	polarFlag_( 0 ), relativeCoordsFlag_( 0 ),
	doubleEndpointFlag_( 0 )
{
	Id libId;
	bool ret = lookupGet< Id, string >(
				Element::root(), "lookupChild", libId, "library" );

	if ( !ret || libId.bad() ) {
		cerr << "Warning: ReadCell: No library for channels\n";
		cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
		return;
	}

	assert( globalParms.size() == 5 );

	if ( !isEqual(globalParms[0], 0.0 ))
		CM_ = globalParms[0];
	if ( !isEqual(globalParms[1], 0.0 ))
		RM_ = globalParms[1];
	if ( !isEqual(globalParms[2], 0.0 ))
		RA_ = globalParms[2];
	if ( !isEqual(globalParms[3], 0.0 ))
		EREST_ACT_ = globalParms[3];
        if ( !isEqual(globalParms[4], 0.0 ))
                ELEAK_ = globalParms[4];
        else
                ELEAK_ = EREST_ACT_;
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
 * put it here in Shell because the cell format is independent
 * of parser and is likely to remain a legacy for a while.
 */
void ReadCell::read(
	const string& filename, 
	const string& cellname,
	Id pa )
{
	filename_ = filename;

	PathUtility pathUtil(Property::getProperty(Property::SIMPATH));

	ifstream fin( filename.c_str() );
	for (unsigned int i = 0; i < pathUtil.size() && !fin; ++i )
	{
		string path = pathUtil.makeFilePath(filename, i);
		fin.clear( );
		fin.open( path.c_str() );
	}
        if (!fin){
            cerr << "ReadCell::read -- could not open file " << filename << endl;
            return;
        }
        
	cell_ = Neutral::create( "Cell", cellname, pa, idGen_.next() );

	if ( !cell_ ) {
		cerr << "Error: ReadCell::read: unable to create cell " <<
			cellname << " on " << pa << "." << pa.node() << endl;
		cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
		return;
	}

	currCell_ = cell_;

	innerRead( fin );
}

void ReadCell::innerRead( ifstream& fin )
{
    string read, line;
	lineNum_ = 0;
	string::size_type pos;
	ParseStage parseMode = DATA;
	while ( getline( fin, read ) ) {
		lineNum_++;
                line = trim(read);
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
				readData( line );
		else if ( parseMode == SCRIPT ) {
				readScript( line );
				parseMode = DATA;
		}
	}
	
	cout << " innerRead: " <<
			numCompartments_ << " compartments, " << 
			numChannels_ << " channels, " << 
			numOthers_ << " others\n";
}

void ReadCell::readData( const string& line )
{
	vector< string > argv;
	parseString( line, argv, "\t " ); 
	if ( argv.size() < 6 ) {
			cerr << "Error: ReadCell: Too few arguments in line: " << argv.size()
				<<	", should be > 6\n";
			cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
			return;
	}
	
	double x0 = 0.0, y0 = 0.0, z0 = 0.0, x, y, z, d;
	int argOffset = 0;
	string name = argv[0];
	string parent = argv[1];
	
	if ( doubleEndpointFlag_ ) {
		argOffset = 3;
		
		x0 = 1.0e-6 * atof( argv[2].c_str() );
		y0 = atof( argv[3].c_str() );
		z0 = atof( argv[4].c_str() );
		if ( polarFlag_ ) {
			double r = x0;
			double theta = y0 * M_PI / 180.0;
			double phi = z0 * M_PI / 180.0;
			x0 = r * sin( phi ) * cos ( theta );
			y0 = r * sin( phi ) * sin ( theta );
			z0 = r * cos( phi );
		} else {
			y0 *= 1.0e-6;
			z0 *= 1.0e-6;
		}
	}
	
	x = 1.0e-6 * atof( argv[argOffset + 2].c_str() );
	y = atof( argv[argOffset + 3].c_str() );
	z = atof( argv[argOffset + 4].c_str() );
	if ( polarFlag_ ) {
		double r = x;
		double theta = y * M_PI / 180.0;
		double phi = z * M_PI / 180.0;
		x = r * sin( phi ) * cos ( theta );
		y = r * sin( phi ) * sin ( theta );
		z = r * cos( phi );
	} else {
		y *= 1.0e-6;
		z *= 1.0e-6;
	}

	d = 1.0e-6 * atof( argv[argOffset + 5].c_str() );

	double length;
	Element* compt =
		buildCompartment( name, parent, x0, y0, z0, x, y, z, d, length, argv );
	if ( compt )
		buildChannels( compt, argv, d, length );
}

Element* ReadCell::buildCompartment( 
				const string& name, const string& parent,
				double x0, double y0, double z0,
				double x, double y, double z,
				double d, double& length,
				vector< string >& argv )
{
// BUG: the comptCinfo raxial and axial are wrong from symcompartments.
	static const Finfo* axial = comptCinfo->findFinfo( "axial" );
	static const Finfo* raxial = comptCinfo->findFinfo( "raxial" );
	static const Finfo* x0Finfo = comptCinfo->findFinfo( "x0" );
	static const Finfo* y0Finfo = comptCinfo->findFinfo( "y0" );
	static const Finfo* z0Finfo = comptCinfo->findFinfo( "z0" );
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
		string paPath = currCell_->id().path() + "/" + parent;
		// Id paId = Id::localId( currCell_->id().path() + "/" + parent );
		Id paId = Id::localId( paPath );
		if ( paId.bad() ) {
			cerr << "Error: ReadCell: could not find parent compt '" <<
					parent << "' for child '" << name << "'\n";
			cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
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
		if ( name[ name.length() - 1 ] == ']' ) {
			string::size_type pos = name.rfind( '[' );
			if ( pos == string::npos ) {
				cerr << "Error: ReadCell: bad child name:" << name << endl;
				cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
				return 0;
			}
			unsigned int index = 
				atoi( name.substr( pos + 1, name.length() - pos ).c_str() );
			if ( childId.index() == index ) {
				cerr << "Error: ReadCell: duplicate child on parent compt '" <<
						parent << "' for child '" << name << "'\n";
				cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
				return 0;
			}
		} else {
			cerr << "Error: ReadCell: duplicate child on parent compt '" <<
					parent << "' for child '" << name << "'\n";
			cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
			return 0;
		}
	}

	Element* compt;
	if ( graftFlag_ && ( parent == "none" || parent == "nil" ) ) {
		compt = currCell_;
	} else {
		if ( protoCompt_ ) {
			compt = protoCompt_->copy( currCell_, name, idGen_ );
			numCompartments_ += numProtoCompts_;
			numChannels_ += numProtoChans_;
			numOthers_ += numProtoOthers_;
		} else {
			compt = Neutral::create( "Compartment",
				name, currCell_->id(), idGen_.next() );
			if ( !graftFlag_ )
				++numCompartments_;
		}
	}
	lastCompt_ = compt;

	if ( pa != Element::root() ) {
		double px, py, pz, dx, dy, dz;
		get< double >( pa, xFinfo, px );
		get< double >( pa, yFinfo, py );
		get< double >( pa, zFinfo, pz );
		
		if ( !doubleEndpointFlag_ ) {
			x0 = px;
			y0 = py;
			z0 = pz;
		}
		if ( relativeCoordsFlag_ == 1 ) {
			x += px;
			y += py;
			z += pz;
			if ( doubleEndpointFlag_ ) {
				x0 += px;
				y0 += py;
				z0 += pz;
			}
		}
		dx = x - x0;
		dy = y - y0;
		dz = z - z0;

		length = sqrt( dx * dx + dy * dy + dz * dz );
		Eref( pa ).add( axial->msg(), compt, raxial->msg(), 
			ConnTainer::Default );
		// axial->add( pa, compt, raxial );
	} else {
		length = sqrt( x * x + y * y + z * z ); 
		// or it coult be a sphere.
	}

	set< double >( compt, x0Finfo, x0 );
	set< double >( compt, y0Finfo, y0 );
	set< double >( compt, z0Finfo, z0 );
	set< double >( compt, xFinfo, x );
	set< double >( compt, yFinfo, y );
	set< double >( compt, zFinfo, z );
	set< double >( compt, dFinfo, d );

	set< double >( compt, lengthFinfo, length );

	double Rm = RM_ / calcSurf(length, d);
	set< double >( compt, RmFinfo, Rm );
	double Ra;
        if (length > 0)
            Ra = RA_ * length * 4.0 / ( d * d * M_PI );
        else
            Ra = RA_ * 8.0 / ( d * M_PI );
	set< double >( compt, RaFinfo, Ra );
	double Cm = CM_ * calcSurf(length, d);
	set< double >( compt, CmFinfo, Cm );
	set< double >( compt, initVmFinfo, EREST_ACT_ );
	set< double >( compt, EmFinfo, ELEAK_ );
	set< double >( compt, VmFinfo, EREST_ACT_ );

	return compt;
}

void ReadCell::readScript( const string& line )
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

	if ( argv[0] == "*set_global" || argv[0] == "*set_compt_param" ) {
		if ( argv.size() != 3 ) {
			cerr << "Error: readCell: Bad line: " << lineNum_ <<
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
                if (argv[1] == "ELEAK" )
                                ELEAK_ = atof( argv[2].c_str() );
	}

	if ( argv[0] == "*start_cell" ) {
		if ( argv.size() == 1 ) {
			graftFlag_ = 0;
			currCell_ = cell_;
		} else if ( argv.size() == 2 ) {
			graftFlag_ = 1;
			currCell_ = startGraftCell( argv[1] );
			assert( currCell_ != 0 );
		} else {
			cerr << "Error: readCell: Bad line: " << lineNum_ <<
					": " << line << endl;
			return;
		}
	}

	if ( argv[0] == "*compt" ) {
		if ( argv.size() != 2 ) {
			cerr << "Error: readCell: Bad line: " << lineNum_ <<
					": " << line << endl;
			return;
		}

		Id protoId( argv[1] );
		if ( protoId.bad() ) {
			cerr << "Error: readCell: Bad path: " << lineNum_ <<
					": " << line << endl;
			return;
		}
		
		protoCompt_ = protoId();
		countProtos( );
		return;
	}
	
	if ( argv[0] == "*double_endpoint" ) {
		doubleEndpointFlag_ = 1;
	}

	if ( argv[0] == "*double_endpoint_off" ) {
		doubleEndpointFlag_ = 0;
	}
	
	if ( argv[0] == "*makeproto" ) {
		return; // Should traverse tree below and drop process messages.
	}
}


Element* ReadCell::startGraftCell( const string& cellpath )
{
	// Warning: here is a parser dependence in the separator.
	Id cellId = Id::localId( cellpath );
	
	if ( ! cellId.bad() ) {
		if ( ! cellId.isGlobal() ) {
			cerr << "Warning: ReadCell: cell '" << cellpath << "' already exists.\n";
			cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
			return 0;
		}
		
		return cellId();
	}
	
	string cellname;
	
	string::size_type pos = cellpath.find_last_of( "/" );
	Id parentId;
	if ( pos == string::npos ) {
		cerr << "Error: ReadCell: *start_cell should be given absolute path.\n";
		cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
		return 0;
	} else if ( pos == 0 ) {
		parentId = Element::root()->id();
		cellname = cellpath.substr( 1 );
	} else {
		parentId = Id( cellpath.substr( 0, pos  ), "/" );
		if ( parentId.bad() ) {
			cerr << "Error: ReadCell: cell path '" << cellpath
				<< "' not found.\n";
			cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
			return 0;
		}
		
		cellname = cellpath.substr( pos + 1 );
	}
	
	//~ if ( parentId.isGlobal() && Shell::myNode() == 0 ) {
		//~ return Shell::createGlobal( 
			//~ "Compartment", cellname, parentId, Id::newId() );
	//~ } else {
		//~ return Neutral::create(
			//~ "Compartment", cellname, parentId, Id::newId() );
	//~ }
	
	return Neutral::create( "Compartment", cellname, parentId, idGen_.next() );
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
	if ( isEqual(len, 0.0) ) // Spherical
		area = dia * dia * M_PI;
	else
		area = len * dia * M_PI;

	return area;
}

bool ReadCell::buildChannels( Element* compt, vector< string >& argv,
				double diameter, double length)
{
	bool isArgOK;
	int argStart;
	vector< Element* > goodChannels;
	
	if ( doubleEndpointFlag_ ) {
		isArgOK = ( argv.size() % 2 ) == 1;
		argStart = 9;
	} else {
		isArgOK = ( argv.size() % 2 ) == 0;
		argStart = 6;
	}
	
	if ( !isArgOK ) {
		cerr << "Error: readCell: Bad number of arguments in channel list\n";
		cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
		return 0;
	}
	for ( unsigned int j = argStart; j < argv.size(); j++ ) {
		// Here we explicitly set compt fields by scaling from the 
		// specific value applied here.
		string chan = argv[j];

		double value = atof( argv[ ++j ].c_str() );
		if ( chan == "RA" ) {
			double temp;
			if ( isEqual(length, 0.0) ) // spherical flag. Assume length = dia.
				temp = 8.0 * value / (diameter * M_PI );
			else
				temp = 4.0 * value * length / (diameter * diameter * M_PI );
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
		} else if ( chan == "kinModel" ) {
			// Need 3 args here: 
			// lambda, name of proto, method
			// We already have lambda from value. Note it is in microns
			if ( j + 2 < argv.size() ) {
				string protoName = argv[ ++j ];
				string method = argv[ ++j ];
				addKinModel( compt, value * 1.0e-6, protoName, method );
			} else {
				cerr << "Error: readCell: kinModel needs 3 args\n";
				break;
			}
		} else if ( chan == "m2c" ) {
			// Need 5 args here: 
			// scale factor, mol, moloffset, chan, chanoffset
			// We already have scale factor from value.
			if ( j + 4 < argv.size() ) {
				addM2C( compt, value, argv.begin() + j + 1 ); 
				j += 4;
			} else {
				cerr << "Error: readCell: m2c adaptor needs 5 args\n";
				break;
			}
		} else if ( chan == "c2m" ) {
			// Need another 5 args here: 
			// scale factor, chan, chanoffset, mol, moloffset
			if ( j + 4 < argv.size() ) {
				addC2M( compt, value, argv.begin() + j + 1 ); 
				j += 4;
			} else {
				cerr << "Error: readCell: c2m adaptor needs 5 args\n";
				break;
			}
		} else {
			Element* chanElm = findChannel( chan );
			if ( chanElm == 0 ) {
				cerr << "Error: readCell: Channel '" << chan <<
						"' not found\n";
				cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
				continue;
			}
			
			Element* copy = addChannel( compt, chanElm, value, diameter, length );
			if ( copy != 0 ) {
				goodChannels.push_back( copy );
			} else {
				cerr << "Error: readCell: Could not add " << chan
					<< " in " << compt->name() << ".";
				cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
			}
		}
	}
	
	for ( unsigned i = 0; i < goodChannels.size(); i++ )
		addChannelMessage( goodChannels[ i ] );
	
	return 1;
}

void ReadCell::addKinModel( Element* compt, double lambda, 
	string name, string method )
{
	/*
	cout << "addKinModel on " << compt->name() <<
		" name= " << name << ", lambda = " << lambda <<
		", using " << method << endl;
		*/

	Element* kinElm = findChannel( name );
	if ( kinElm == 0 ) {
		cerr << "Error:readCell: KinProto '" << name << "' not found\n";
		cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
		return;
	}
	
	Element* kph = Neutral::create( "KinPlaceHolder", "kinModel", 
		compt->id(), Id::childId( compt->id() ) );
	set< Id, double, string >( kph, "setup", 
		kinElm->id(), lambda, method );
}

void ReadCell::addM2C( Element* compt, double scale, 
	vector< string >::iterator args )
{
	/*
	cout << "addM2C on " << compt->name() << 
		" scale= " << scale << 
		" mol= " << *args << ", moloff= " << *(args+1) << 
		" chan= " << *(args + 2) << ", chanoff= " << *(args+3) << endl;
		*/

	string molName = *args++;
	double molOffset = atof( ( *args++ ).c_str() );
	string chanName = *args++;
	double chanOffset = atof( ( *args ).c_str() );
	string adaptorName = molName + "_2_" + chanName;

	Element* chan = findChannel( chanName );
	if ( chan == 0 ) {
		cerr << "Error:readCell: addM2C ': channel" << chanName << 
			"' not found\n";
		cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
		return;
	}

	Element* adaptor = Neutral::create( "Adaptor", adaptorName,
		compt->id(), Id::childId( compt->id() ) );
	
	Eref( adaptor ).add( "outputSrc", Eref( chan ), "Gbar" );
	set< string, double, double, double >( adaptor, "setup",
		molName, scale, molOffset, chanOffset );
}

void ReadCell::addC2M( Element* compt, double scale, 
	vector< string >::iterator args )
{
	/*
	cout << "addC2M on " << compt->name() << 
		" scale= " << scale << 
		" chan= " << *args << ", chanoff= " << *(args+1) << 
		" mol= " << *(args + 2) << ", moloff= " << *(args+3) <<  endl;
		*/

	string chanName = *args++;
	double chanOffset = atof( ( *args++ ).c_str() );
	string molName = *args++;
	double molOffset = atof( ( *args++ ).c_str() );
	string adaptorName = "Ca_2_" + molName;

	Element* chan = findChannel( chanName );
	if ( chan == 0 ) {
		cerr << "Error:readCell: addC2M ': channel" << chanName << 
			"' not found\n";
		cerr << "File: " << filename_ << " Line: " << lineNum_ << endl;
		return;
	}

	Element* adaptor = Neutral::create( "Adaptor", adaptorName,
		compt->id(), Id::childId( compt->id() ) );
	
	Eref( adaptor ).add( "inputRequest", Eref( chan ), "Ca" );
	set< string, double, double, double >( adaptor, "setup",
		molName, scale, chanOffset, molOffset );
}

Element* ReadCell::addChannel( 
			Element* compt, Element* proto, double value, 
			double dia, double length )
{
	Element* copy = proto->copy( compt, "", idGen_ );
	assert( copy != 0 );
	
	if ( addHHChannel( compt, copy, value, dia, length ) ) return copy;
	if ( addSynChan( compt, copy, value, dia, length ) ) return copy;
	if ( addSpikeGen( compt, copy, value, dia, length ) ) return copy;
	if ( addCaConc( compt, copy, value, dia, length ) ) return copy;
	if ( addNernst( compt, copy, value ) ) return copy;
	
	return 0;
}

bool ReadCell::addHHChannel( 
		Element* compt, Element* chan, 
		double value, double dia, double length )
{
	static const Finfo* chanSrcFinfo = comptCinfo->findFinfo( "channel" );
	static const Finfo* hhChanDestFinfo = chanCinfo->findFinfo( "channel" );
	static const Finfo* gbarFinfo = chanCinfo->findFinfo( "Gbar" );
	
	if (( chan->className() == "HHChannel" )|| ( chan->className() == "HHChannel2D" )){
#ifdef DEBUG
            // DEBUG
            if ( chan->className() == "HHChannel2D" ) {
                cout << "name:" << chan->name() << ", path:" << chan->id().path() << endl;
            }
#endif
		bool ret = Eref( compt ).add( chanSrcFinfo->msg(), chan, hhChanDestFinfo->msg(), ConnTainer::Default );
		// bool ret = chanSrcFinfo->add( compt, chan, hhChanDestFinfo );
		assert( ret );
			
		if ( value > 0 ) {
                    value *= calcSurf(length, dia);
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
		// bool ret = chanSrcFinfo->add( compt, chan, synChanDestFinfo );

		bool ret = Eref( compt ).add( chanSrcFinfo->msg(), 
			chan, synChanDestFinfo->msg(), ConnTainer::Default );

		assert( ret );
		
		if ( value > 0 ) {
                    value *= calcSurf(length, dia);
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
		// bool ret = vmSrcFinfo->add( compt, chan, vmDestFinfo  );
		bool ret = Eref( compt ).add( vmSrcFinfo->msg(), 
			chan, vmDestFinfo->msg(), ConnTainer::Default );
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
        static const Finfo* thicknessFinfo = caconcCinfo->findFinfo("thick");
        double thick;
        get<double>(chan, thicknessFinfo, thick);
        if (thick > dia/2.0) thick = 0.0;
	if ( chan->className() == "CaConc" ) {
		// assert( vmSrcFinfo->add( compt, chan, vmDestFinfo  ) );

		if ( value > 0.0 ) {
                    double vol;
                    if (length > 0.0){
                        if (thick > 0.0){
                            vol = M_PI * length * (dia - thick) * thick;
                        } else {
                            vol = dia * dia * M_PI * length / 4.0;
                        }
                    } else { // spherical
                        if (thick > 0.0){
                            double inner_dia = dia - 2 * thick;
                            vol = M_PI * ( dia * dia * dia - inner_dia * inner_dia * inner_dia) / 6.0; 
                        } else {
                            vol = M_PI * dia * dia * dia / 6.0;
                        }
                    }
                    if ( vol > 0.0 ) // Scale by volume.
                        value /= vol;
		} else {
			value = - value;
		}

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

void ReadCell::addChannelMessage( Element* chan )
{
	Id sli( "/shell/sli" );
	
	vector< const char* > argVector;
	string argString;
	string token;
	vector< string > tokens;
	
	/*
	 * Get extended Finfos on channel. (We're looking for fields added using
	 * "addfield"
	 */
	vector< Finfo* > chanFinfo;
	chan->listLocalFinfos( chanFinfo );
	
	vector< Finfo* >::iterator cfinfo;
	for ( cfinfo = chanFinfo.begin(); cfinfo != chanFinfo.end(); cfinfo++ ) {
		// Ignore a Finfo if its name does not begin with "addmsg"..
		const string& name = ( *cfinfo )->name();
		if ( name.find( "addmsg", 0 ) != 0 )
			continue;
		
		// ..or if the "addmsg" is not followed by a positive integer.
		unsigned int dummyInt;
		stringstream remaining( name.substr( 6 ) );
		if ( ( remaining >> dummyInt ).fail() )
			continue;
		
		// Get the string contained in the field..
		if ( ( *cfinfo )->strGet( chan, argString ) == false )
			continue;
		
		// ..extract tokens from the string..
		tokens.clear();
		tokens.push_back( "addmsg" );
		stringstream ss( argString );
		while ( ss >> token )
			tokens.push_back( token );
		
		/*
		 * Convert token[ 1 ] and token[ 2 ] from relative paths to absolute ones.
		 * Ignore if the tokens are not valid paths. (Possible if using the new
		 * syntax: "addmsg src_elm/msg dest_elm/msg")
		 */
		if ( tokens.size() >= 3 ) {
			Id token1( chan->id().path() + "/" + tokens[ 1 ] );
			Id token2( chan->id().path() + "/" + tokens[ 2 ] );
			
			if ( token1.good() )
				tokens[ 1 ] = token1.path();
			if ( token2.good() )
				tokens[ 2 ] = token2.path();
		}
		
		// Get C-style strings
		argVector.clear();
		for ( unsigned i = 0; i < tokens.size(); i++ )
			argVector.push_back( tokens[ i ].c_str() );
		
		// Request parser to add the message
		do_add( argVector.size(), &argVector[ 0 ], sli );
		// do_add is defined in GenesisParserWrapper.cpp
	}
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
