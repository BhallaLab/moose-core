/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <fstream>
#include <math.h>
#include "header.h"
#include "Shell.h"
#include "ShellWrapper.h"
#include "../builtins/Interpol.h"

// Read a cell model into MOOSE.
// Comments are as in C++
//
// Special command lines start with a '*'
// Commands include:
// Coordinate system:
// 		*cartesian	*polar	*cylindrical
// Measurement reference:
//		*relative *absolute
// Compartment type:
//		*symmetric *asymmetric
//
// Value assignment:
//		*set_global <name> <value>
//
// Main data lines are of form
// name  parent  x       y       z       d       ch      dens ...
// name is name of new compartment
// arent = name of preceding compartment, or . 
// 		if it is the same as the last one
// x y z are the coordinates, in whatever system is chosen
// 

enum CoordMode { CARTESIAN, POLAR, CYLINDRICAL };
static const double PI = 3.1415926535;

class ReadCell
{
	public:
		ReadCell( const string& fileName, const string& cellName,
			const string& options );
		double calculateCoords( Element* compt, Element* parent,
			double x, double y, double z );
		bool readComptLine( const string& line );
		bool assignCompartmentParameter( 
			Element* compt, const string& name,
			double value, double area) ;
		static void chopLine( 
			const string& line, vector< string >& args );
		void makeChannel( 
			Element* compt, const string& name, double density,
			double area, double volume, 
			vector< pair< Element*, string > >& addList );
		void addReadCellMsg( Element* chan, const string& command );
		bool isComment( string& line );
		void setGlobal( const string& line );
		void setupAlphaFunc( int argc, const char** argv, bool setupTau );
		void tweakFunc( int argc, const char** argv, bool setupTau );
		void printStats();

	private:
		Element* cell_;
		Element* lastCompt_;
		Element* library_;
		CoordMode coordMode_;
		bool isRelative_;
		bool isSymmetric_;
		unsigned long lineNum_;
		bool inComment_;
		double RM_;
		double CM_;
		double RA_;
		double EREST_ACT_;
		double ELEAK_;
		//double X0_;
		//double Y0_;
		//double Z0_;
};

ReadCell::ReadCell( const string& fileName, const string& cellName,
	const string& options )
	: cell_( 0 ), lastCompt_( 0 ), coordMode_( CARTESIAN ),
		isRelative_( 0 ), isSymmetric_( 0 ), lineNum_( 0 ),
		inComment_( 0 ), RM_( 1.0 ), CM_( 1.0 ), RA_( 1.0 ), 
		EREST_ACT_( -0.065 ), ELEAK_( -0.065 )
{
	cell_ = Element::root()->relativeFind( cellName );
	if ( !cell_ ) {
		cerr << "Error:ReadCell::ReadCell: parent cell not found: '" <<
			cellName << "'\n";
		return;
	}
	library_ = Element::root()->relativeFind( "library" );
	if ( !library_ ) {
		cerr << "Error:ReadCell::ReadCell: /library not found \n";
		return;
	}
	ifstream f( fileName.c_str() );
	string line;
	lineNum_ = 0;
	std::getline( f, line );
	while ( f.good() ) {
		lineNum_++;
		if ( isComment( line ) == 0 ) {
			if ( line[ 0 ] == '*' ) {
				if ( line == "*cartesian" )
					coordMode_ = CARTESIAN;
				else if ( line == "*polar" )
					coordMode_ = POLAR;
				else if ( line == "*cylindrical" )
					coordMode_ = CYLINDRICAL;
				else if ( line == "*relative" )
					isRelative_ = 1;
				else if ( line == "*absolute" )
					isRelative_ = 0;
				else if ( line == "*symmetric" )
					isSymmetric_ = 1;
				else if ( line == "*asymmetric" )
					isSymmetric_ = 0;
				else if ( line.substr( 0, 11 ) == "*set_global" )
					setGlobal( line.substr( 12 ) );
				else
					cerr << "Error:ReadCell::ReadCell: line " <<
						lineNum_ << ": Unknown option '" <<
						line << "'\n";
			} else {
				readComptLine( line );
			}
		}
		std::getline( f, line );
	}
}

// Returns 1 if the whole line is a comment, otherwise cut out
// the comment parts and return 0.
bool ReadCell::isComment( string& line )
{
	size_t p;
	if ( inComment_ ) {
		p = line.find( "*/" );
		if ( p == string::npos )
			return 1;

		inComment_ = 0;
		if ( p + 2 >= line.length() )
			return 1;
		line = line.substr( p + 2 );
		p = line.find_first_not_of( " \t" );
		if ( p == string::npos )
			return 1;
		line = line.substr( p );
		return 0;
	} else {
		p = line.find( "/*" );
		if ( p != string::npos )
			inComment_ = 1;
		else 
			p = line.find( "//" );

		if ( p != string::npos ) {
			line = line.substr( 0, p );
			p = line.find_first_not_of( " \t" );
			if ( p == string::npos )
				return 1;
			line = line.substr( p );
			return 0;
		}
		return 0;
	}
}

void ReadCell::chopLine( const string& line, vector< string >& args )
{
	unsigned long p = line.find_first_not_of( " \t" );
	if ( p == string::npos )
		return;
	string s = line;
	string temp;
	while ( p != string::npos ) {
		s = s.substr( p );
		p = s.find_first_of( " \t" );
		temp = s.substr( 0, p );
		args.push_back( temp );
		if ( p == string::npos )
			break;
		s = s.substr( p );
		p = s.find_first_not_of( " \t" );
	}
}


void ReadCell::setGlobal( const string& line )
{
	vector< string > args;
	chopLine( line, args );
	if ( args.size() != 2 ) {
		cerr << "Syntax Error: ReadCell: *set_global " << line << "\n";
		return;
	}
	double value = atof( args[1].c_str() );
	if ( args[0] == "RM" ) RM_ = value;
	else if ( args[0] == "CM" ) CM_ = value;
	else if ( args[0] == "RA" ) RA_ = value;
	else if ( args[0] == "EREST_ACT" ) EREST_ACT_ = value;
	else if ( args[0] == "ELEAK" ) ELEAK_ = value;
	else {
		cerr << "ReadCell: Unknown field: *set_global " << line << "\n";
	}
}

bool ReadCell::readComptLine( const string& line )
{
	static const Cinfo* comptCi = Cinfo::find( "Compartment" );
	// static const Cinfo* symComptCi = Cinfo::find( "SymCompartment" );
	static const Cinfo* symComptCi = comptCi;
	vector< string > args;
	chopLine( line, args );
	if ( args.size() == 0 )
		return 0;
	if ( args.size() < 6 || ( args.size() % 2 ) != 0 ) {
		cerr << "Error: ReadCell::readComptLine: line " << lineNum_ <<
			": Wrong number of args in '" << line << "'\n";
		return 0;
	}

	Element* compt;	
	Element* parent;
	if ( isSymmetric_ )
		compt = symComptCi->create( args[0], cell_ );
	else
		compt = comptCi->create( args[0], cell_ );
	
	if ( args[1] == "." ) // Parent compt
		parent = lastCompt_;
	else if ( args[1] == "none" ) // Parent compt
		parent = 0;
	else 
		parent = cell_->relativeFind( args[1] );

	if ( args[1] != "none" ) {
		if ( !parent ) {
			cerr << "Error: ReadCell::readComptLine: line " << lineNum_ <<
				": parent compt not found: '" << args[1] << "'\n";
			return 0;
		}
		Field temp( compt, "raxial" );
		parent->field( "axial" ).add( temp );
	}
	
	double length = calculateCoords( compt, parent,
		atof( args[ 2 ].c_str() ), 
		atof( args[ 3 ].c_str() ), 
		atof( args[ 4 ].c_str() ) ); 
	double diameter = atof( args[ 5 ].c_str() );
	double surfaceArea = length * diameter * PI * 1.0e-12 ;
	double crossSectionArea = diameter * diameter * PI * 1.0e-12 / 4.0;

	Ftype1< double >::set( compt, "diameter", diameter );
	Ftype1< double >::set( compt, "Em", EREST_ACT_ );
	Ftype1< double >::set( compt, "initVm", EREST_ACT_ );
	Ftype1< double >::set( compt, "Rm", RM_ / surfaceArea );
	Ftype1< double >::set( compt, "Cm", CM_ * surfaceArea );
	Ftype1< double >::set( 
		compt, "Ra", RA_ * length * 1.0e-6 / crossSectionArea );

	vector< pair< Element*, string > > addList;
	for ( unsigned long i = 6 ; i < args.size(); i += 2 )
		makeChannel( compt, args[ i ], atof( args[ i + 1 ].c_str() ), 
			surfaceArea, crossSectionArea * length * 1.0e-6, addList );
	
	vector< pair< Element*, string > >::iterator j;
	for ( j = addList.begin(); j != addList.end(); j++ )
		addReadCellMsg( j->first, j->second );

	lastCompt_ = compt;
	return 1;
}

// Returns the length of the compartment.
double ReadCell::calculateCoords( Element* compt, Element* parent,
	double x, double y, double z )
{
	double length = 0.0;
	if ( parent == 0 ) {
		length = sqrt( x * x + y * y + z * z );
		Ftype1< double >::set( compt, "length", length );
		return length;
	}
	double paX = 0.0;
	double paY = 0.0;
	double paZ = 0.0;
	double myX, myY, myZ;
	switch ( coordMode_ ) {
		case CARTESIAN :
			if ( isRelative_ ) {
				length = sqrt( x * x + y * y + z * z );
				myX = paX + x;
				myY = paY + y;
				myZ = paZ + z;
			} else {
				length = sqrt( x * x + y * y + z * z );
				myX = x;
				myY = y;
				myZ = z;
			}
			break;
		case POLAR :
			if ( isRelative_ ) {
				length = x;
				myX = paX + x * cos( y * PI / 180.0 ) * 
					cos( z * PI / 180.0 ) ;
				myY = paY + x * sin( y * PI / 180.0 ) * 
					cos( z * PI / 180.0 ) ;
				myZ = paZ + x * sin( z * PI / 180.0 );
			} else {
				length = x;
				myX = x * cos( y * PI / 180.0 ) * cos( z * PI / 180.0 );
				myY = x * sin( y * PI / 180.0 ) * cos( z * PI / 180.0 );
				myZ = x * sin( z * PI / 180.0 );
			}
			break;
		case CYLINDRICAL :
			if ( isRelative_ ) {
				length = sqrt ( x * x + z * z );
				myX = paX + x * cos( y * PI / 180.0 );
				myY = paY + x * sin( y * PI / 180.0 );
				myZ = paZ + z;
			} else {
				length = x;
				myX = x * cos( y * PI / 180.0 );
				myY = x * sin( y * PI / 180.0 );
				myZ = z;
			}
			break;
		default:
			break;
	}
	Ftype1< double >::set( compt, "length", length );
	return length;
	/*
	Ftype1< double >::set( compt, "x", mX );
	Ftype1< double >::set( compt, "y", mY );
	Ftype1< double >::set( compt, "z", mZ );
	*/
}

static map< string, string >& readCellSrcLookup()
{
	static map< string, string > src;

	if ( src.size() > 0 )
		return src;
	
	src[ "I_Ca" ] = "IkOut";	// for Ca current
	src[ "CONCEN" ] = "concOut";	// for Ca current

	return src;
}

static map< string, string >& readCellDestLookup()
{
	static map< string, string > dest;

	if ( dest.size() > 0 )
		return dest;
	
	dest[ "I_Ca" ] = "currentIn";	// for Ca current
	dest[ "CONCEN" ] = "concenIn";	// for Ca current

	return dest;
}

void ReadCell::addReadCellMsg( Element* chan, const string& command )
{
	vector < string > args;
	chopLine( command, args );
	Element* src = chan->relativeFind( args[0] );
	if ( !src ) {
		cerr << "Error: addReadCellMsg: Failed to find src " << 
			chan->path() << "/" << args[ 0 ] << "\n";
		return;	
	}
	Element* dest = chan->relativeFind( args[1] );
	if ( !dest ) {
		cerr << "Error: addReadCellMsg: Failed to find dest " << 
			chan->path() << "/" << args[ 1 ] << "\n";
		return;	
	}

	map< string, string >::iterator i;
	i = readCellSrcLookup().find( args[2] );
	if ( i != readCellSrcLookup().end() ) {
		string srcMsg = i->second;
		i = readCellDestLookup().find( args[2] );
		if ( i != readCellDestLookup().end() ) {
			string destMsg = i->second;
			Field f( dest, destMsg );
			if ( ! src->field( srcMsg ).add( f ) )
				cerr << "Error: addReadCellMsg: Failed to add msg: " << 
					src->path() << " " << args[ 2 ] << " to \n" <<
					dest->path() << " " << args[ 3 ] << "\n";
		} else {
				cerr << "Error: addReadCellMsg: MsgDest type unknown: "
					<< src->path() << " " << args[ 2 ] << " to \n" <<
					dest->path() << " " << args[ 3 ] << "\n";
		}
	} else {
		cerr << "Error: addReadCellMsg: MsgSrc type unknown: " <<
			src->path() << " " << args[ 2 ] << " to \n" <<
			dest->path() << " " << args[ 3 ] << "\n";
	}
}

bool ReadCell::assignCompartmentParameter( 
	Element* compt, const string& name, double value, double area ) 
{
	return Ftype1< double >::set( compt, name, value );
}

void ReadCell::makeChannel( 
	Element* compt, const string& name, double density, double area, 
	double volume, vector< pair< Element*, string > >& addList )
{
	Element* chanProto = library_->relativeFind( name );
	if ( !chanProto ) {
		// See if it is a direct assignment of Rm, Cm, or Ra
		if ( !assignCompartmentParameter( compt, name, density, area ) )
			cerr << "Error: ReadCell::makeChannel: line " << lineNum_ <<
				": channel not found: '" << name << "'\n";
		return;
	}
	Element* chan = chanProto->shallowCopy( compt );
	//Element* chan = chanProto->cinfo()->create( chanProto->name(), compt, chanProto );
	if ( chan->cinfo()->name() == "HHChannel" ) {
		Field temp( chan, "channel" );
		compt->field( "channel" ).add( temp );
		if ( density > 0 ) {
			Ftype1< double >::set( chan, "Gbar", density * area );
		} else {
			Ftype1< double >::set( chan, "Gbar", -density );
		}
	} else if ( chan->cinfo()->name() == "CaConc" ) {
		// Field temp( chan, "channel" );
		// compt->field( "channel" ).add( temp );
		if ( density > 0 ) {
			Ftype1< double >::set( chan, "B", density / volume ) ;
		} else {
			Ftype1< double >::set( chan, "B", -density );
		}
	}

	pair < Element*, string > p;
	string temp;
	if ( Ftype1< string >::get( chanProto, "addmsg1", temp ) ) {
		pair< Element* , string > pr1( chan, temp );
		addList.push_back( pr1 );
		// addReadCellMsg( chan, temp );
		if ( Ftype1< string >::get( chanProto, "addmsg2", temp ) ) {
			pair< Element* , string > pr2( chan, temp );
			addList.push_back( pr2 );
	//		addReadCellMsg( chan, temp );
		}
	}
}

void ReadCell::printStats()
{
	vector< Element* > ret;
	int nCompt = cell_->wildcardRelativeFind( "##[TYPE=Compartment]", ret, 1 );
	int nHHChan = cell_->wildcardRelativeFind( "##[TYPE=HHChannel]", ret, 1 );
	int nSynChan = cell_->wildcardRelativeFind( "##[TYPE=SynChannel]", ret, 1 );
	int nCaConc = cell_->wildcardRelativeFind( "##[TYPE=CaConc]", ret, 1 );
	int nNernst = cell_->wildcardRelativeFind( "##[TYPE=Nernst]", ret, 1 );
	cout << "ReadCell read " << nCompt << " Compartments, " <<
		nHHChan << " HHChannels, " <<
		nSynChan << " Synaptic Channels, " <<
		nCaConc << " CaConcens, " <<
		nNernst << " Nernst objects\n";
}

//////////////////////////////////////////////////////////////////
// Here we have several Shell functions that deal with ReadCell
//////////////////////////////////////////////////////////////////

void Shell::readcellFunc( int argc, const char** argv )
{
	if (argc < 3 ) {
		error( string("usage: ") + argv[ 0 ] +
		" filename cellname options");
		return;
	}
	if ( argc == 3 ) {
		ReadCell r( argv[ 1 ], argv[ 2 ], "" );
		r.printStats();
	} else if ( argc > 3 ) {
		ReadCell r( argv[ 1 ], argv[ 2 ], argv[ 3 ] );
		r.printStats();
	}
}

// y = (A + B * x) / (C + {exp({(x + D) / F})})
// Need to add in the other one here
void setupInterpol( Interpol& pol, double a, double b, 
	double c, double d, double f )
{
	static const double SINGULAR = 1.0e-6;
	double dx = pol.localGetDx();
	int xdivs = pol.localGetXdivs();
	double x = pol.localGetXmin();
	double lastTemp = 0.0;
	double temp;
	double temp2;
	for ( int i = 0; i <= xdivs; i++ ) {
		if ( fabs( f ) < SINGULAR ) {
			temp = 0.0;
			pol.setTableValue( 0.0, i );
		} else {
			temp2 = c + exp( ( x + d ) / f );
			if ( fabs( temp2 ) < SINGULAR ) {
				temp = lastTemp;
				pol.setTableValue( temp, i );
			} else {
				temp = ( a + b * x ) / temp2;
				pol.setTableValue( temp, i );
			}
		}
		lastTemp = temp;
		x += dx;
	}
}

void sumInterpol( Interpol& A, Interpol& B )
{
	int xdivs = A.localGetXdivs();
	for ( int i = 0; i <= xdivs; i++ ) {
		double temp = B.getTableValue( i ) + A.getTableValue( i ); 
		B.setTableValue( temp, i );
	}
}

void tauTweakInterpol( Interpol& A, Interpol& B )
{
	static const double SINGULAR = 1.0e-8;
	int xdivs = A.localGetXdivs();
	for ( int i = 0; i <= xdivs; i++ ) {
		double temp = A.getTableValue( i ); 
		if ( fabs( temp ) < SINGULAR ) {
			A.setTableValue( 0.0, i );
			B.setTableValue( 0.0, i );
		} else {
			A.setTableValue( B.getTableValue( i ) / temp, i );
			B.setTableValue( 1.0 / temp, i );
		}
	}
}

// call Element TABFILL gate xdivs calc_mode
void Shell::tabFillFunc( int argc, const char** argv )
{
	if ( argc < 6 ) {
		error( "Usage: call element TABFILL gate xdivs calc_mode" );
		return;
	}

	Element *chan = findElement( argv[ 1 ] );
	if ( !chan ) {
		error( "Failed to find channel-element: ", argv[ 1 ] );
		return;
	}
	Element *gate = chan->relativeFind( argv[ 3 ] );
	if ( !gate ) {
		error( "Failed to find gate: ", 
			string( argv[ 1 ] ) + "/" + argv[ 3 ] );
		return;
	}
	Ftype2< int, int >::set( gate, "tabFillIn", 
		atoi( argv[ 4 ] ), atoi( argv[ 5 ] ) );
		/*
	Ftype2< int, int >::set( gate, "A.tabFillIn", 
		atoi( argv[ 4 ] ), atoi( argv[ 5 ] ) );
	Ftype2< int, int >::set( gate, "B.tabFillIn", 
		atoi( argv[ 4 ] ), atoi( argv[ 5 ] ) );
		*/
}

// call Element TABCREATE gate xdivs xmin xmax
void Shell::tabCreateFunc( int argc, const char** argv )
{
	if ( argc < 7 ) {
		error( "Usage: call element TABCREATE gate xdivs xmin xmax" );
		return;
	}

	Element *chan = findElement( argv[ 1 ] );
	if ( !chan ) {
		error( "Failed to find channel-element: ", argv[ 1 ] );
		return;
	}
	Element *gate = chan->relativeFind( argv[ 3 ] );
	if ( !gate ) {
		string gateFieldName = argv[3];
		gate = Cinfo::find( "HHGate" )->create( argv[ 3 ], chan );
		if ( strlen( argv[ 3 ] ) == 1 ) {
			gateFieldName[0] = tolower( gateFieldName[0] );
			gateFieldName = gateFieldName + "Gate";
		}
		Field temp( gate, "gate" );
		if ( !chan->field( gateFieldName ).add( temp ) )
			error("Shell::tabCreateFunc:: Failed to add message from channel to gate: ", gate->path() );
	}
	if ( !gate ) {
		error( "Failed to find gate: ", 
			string( argv[ 1 ] ) + "/" + argv[ 3 ] );
		return;
	}
	Ftype1< int >::set( gate, "A.xdivs", atoi( argv[ 4 ] ) );
	Ftype1< int >::set( gate, "B.xdivs", atoi( argv[ 4 ] ) );
	Ftype1< double >::set( gate, "A.xmin", atof( argv[ 5 ] ) );
	Ftype1< double >::set( gate, "B.xmin", atof( argv[ 5 ] ) );
	Ftype1< double >::set( gate, "A.xmax", atof( argv[ 6 ] ) );
	Ftype1< double >::set( gate, "B.xmax", atof( argv[ 6 ] ) );
}

void Shell::setupAlphaFunc( int argc, const char** argv, bool setupTau )
{
	static const int DEFAULT_XDIVS = 3000;
	static const double DEFAULT_XMIN = -0.1;
	static const double DEFAULT_XMAX = 0.05;

	if (argc < 12 ) {
		error( string("usage: ") + argv[ 0 ] +
		" channel-element gate AA AB AC AD AF BA BB BC BD BF -size n -range min max");
		return;
	}
	Element *chan = findElement( argv[ 1 ] );
	if ( !chan ) {
		error( "Failed to find channel-element: ", argv[ 1 ] );
		return;
	}
	Element *gate = chan->relativeFind( argv[ 2 ] );
	if ( !gate ) {
		string gateFieldName = argv[2];
		gate = Cinfo::find( "HHGate" )->create( argv[ 2 ], chan );
		if ( strlen( argv[ 2 ] ) == 1 ) {
			gateFieldName[0] = tolower( gateFieldName[0] );
			gateFieldName = gateFieldName + "Gate";
		}
		Field temp( gate, "gate" );
		if ( !chan->field( gateFieldName ).add( temp ) )
			error("Shell::setupAlphaFunc:: Failed to add message from channel to gate: ", gate->path() );
	}
	if ( !gate ) {
		error( "Failed to find gate: ", 
			string( argv[ 1 ] ) + "/" + argv[ 2 ] );
		return;
	}
	int xdivs = DEFAULT_XDIVS;
	double xmin = DEFAULT_XMIN;
	double xmax = DEFAULT_XMAX;

	for (int i = 13; i < argc - 1; i++ )
		if ( strncmp( argv[i], "-s", 2 ) == 0)
			xdivs = atoi( argv[ i + 1 ] );
	for (int i = 13; i < argc - 2; i++ ) {
		if ( strncmp( argv[i], "-r", 2 ) == 0) {
			xmin = atof( argv[ i + 1 ] );
			xmax = atof( argv[ i + 2 ] );
		}
	}
	Interpol A( xdivs, xmin, xmax );
	setupInterpol( A,  atof( argv[3] ), atof( argv[4] ), 
		atof( argv[5] ), atof( argv[6] ), atof( argv[7] ) ); 
	Interpol B( xdivs, xmin, xmax );
	setupInterpol( B,  atof( argv[8] ), atof( argv[9] ), 
		atof( argv[10] ), atof( argv[11] ), atof( argv[12] ) ); 
	if ( setupTau )
		tauTweakInterpol( A, B );
	else
		sumInterpol( A, B );
	
	Ftype1< Interpol >::set( gate, "A", A );
	Ftype1< Interpol >::set( gate, "B", B );
}

// Entries in tables may be loaded in a different manner than the
// ones used for calculations. This function switches them around.
// For reference: A is the alpha term, B = alpha + beta
// and: tau = 1/(alpha + beta), minfinity = alpha/( alpha + beta )
// if setupTau == 0: A and B are just the alpha and beta terms. This
// 	function sets B += A for all entries.
// if setupTau == 1: A and B are the tau and minfinity terms. This
// 	function sets A = minf/tau; B = 1 / tau
void Shell::tweakFunc( int argc, const char** argv, bool setupTau )
{
	if (argc < 3 ) {
		error( string("usage: ") + argv[ 0 ] +
		" channel-element gate");
		return;
	}
	Element *chan = findElement( argv[ 1 ] );
	if ( !chan ) {
		error( "tweakFunc: Failed to find channel-element: ", argv[1] );
		return;
	}
	Element *gate = chan->relativeFind( argv[ 2 ] );
	if ( !gate ) {
		error( "tweakFunc: Failed to find gate: ", 
			string( argv[ 1 ] ) + "/" + argv[ 2 ] );
		return;
	}
	Interpol A;
	Interpol B;
	Ftype1< Interpol >::get( gate, "A", A );
	Ftype1< Interpol >::get( gate, "B", B );

	if ( setupTau )
		tauTweakInterpol( A, B );
	else
		sumInterpol( A, B );

	Ftype1< Interpol >::set( gate, "A", A );
	Ftype1< Interpol >::set( gate, "B", B );
}
