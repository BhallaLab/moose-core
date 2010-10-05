/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "shell/Shell.h"
#include "element/Wildcard.h"
#include "element/Neutral.h"

#include "GLview.h"

#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <sstream>
#include <limits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include <math.h>
#include <string.h>
#include <sys/types.h>

#ifdef WIN32
#include <Winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#define errno WSAGetLastError()
#else
#include <errno.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#endif


#include "Constants.h"
#include "GLviewResetData.h"
#include "AckPickData.h"

const int GLview::MSGTYPE_HEADERLENGTH = 1;
const int GLview::MSGSIZE_HEADERLENGTH = 8;
const char GLview::SYNCMODE_ACKCHAR = '*';

const Cinfo* initGLviewCinfo()
{
	static Finfo* processShared[] = 
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			       RFCAST( &GLview::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			       RFCAST( &GLview::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
						 sizeof( processShared ) / sizeof( Finfo* ),
						 "shared message to receive Process messages from scheduler objects" );

	static Finfo* GLviewFinfos[] = 
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		new ValueFinfo( "host",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getClientHost ),
				RFCAST( &GLview::setClientHost )
				),
		new ValueFinfo( "port",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getClientPort ),
				RFCAST( &GLview::setClientPort )
				),
		new ValueFinfo( "vizpath",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getPath ),
				RFCAST( &GLview::setPath )
				),
		new ValueFinfo( "relpath",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getRelPath ),
				RFCAST( &GLview::setRelPath )
				),
		new ValueFinfo( "value1",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getValue1Field ),
				RFCAST( &GLview::setValue1Field )
				),
		new ValueFinfo( "value2",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getValue2Field ),
				RFCAST( &GLview::setValue2Field )
				),
		new ValueFinfo( "value3",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getValue3Field ),
				RFCAST( &GLview::setValue3Field )
				),
		new ValueFinfo( "value4",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getValue4Field ),
				RFCAST( &GLview::setValue4Field )
				),
		new ValueFinfo( "value5",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getValue5Field ),
				RFCAST( &GLview::setValue5Field )
				),
		new ValueFinfo( "value1min",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue1Min ),
				RFCAST( &GLview::setValue1Min )
				),
		new ValueFinfo( "value1max",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue1Max ),
				RFCAST( &GLview::setValue1Max )
				),
		new ValueFinfo( "value2min",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue2Min ),
				RFCAST( &GLview::setValue2Min )
				),
		new ValueFinfo( "value2max",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue2Max ),
				RFCAST( &GLview::setValue2Max )
				),
		new ValueFinfo( "value3min",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue3Min ),
				RFCAST( &GLview::setValue3Min )
				),
		new ValueFinfo( "value3max",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue3Max ),
				RFCAST( &GLview::setValue3Max )
				),
		new ValueFinfo( "value4min",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue4Min ),
				RFCAST( &GLview::setValue4Min )
				),
		new ValueFinfo( "value4max",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue4Max ),
				RFCAST( &GLview::setValue4Max )
				),
		new ValueFinfo( "value5min",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue5Min ),
				RFCAST( &GLview::setValue5Min )
				),
		new ValueFinfo( "value5max",
				ValueFtype1< double >::global(),
				GFCAST( &GLview::getValue5Max ),
				RFCAST( &GLview::setValue5Max )
				),
		new ValueFinfo( "bgcolor",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getBgColor ),
				RFCAST( &GLview::setBgColor )
				),
		new ValueFinfo( "sync",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getSyncMode ),
				RFCAST( &GLview::setSyncMode )
				),
		new ValueFinfo( "grid",
				ValueFtype1< string >::global(),
				GFCAST( &GLview::getGridMode ),
				RFCAST( &GLview::setGridMode )
				),
		new ValueFinfo( "color_val",
				ValueFtype1< unsigned int >::global(),
				GFCAST( &GLview::getColorVal ),
				RFCAST( &GLview::setColorVal )
				),
		new ValueFinfo( "morph_val",
				ValueFtype1< unsigned int >::global(),
				GFCAST( &GLview::getMorphVal ),
				RFCAST( &GLview::setMorphVal )
				),
		new ValueFinfo( "xoffset_val",
				ValueFtype1< unsigned int >::global(),
				GFCAST( &GLview::getXOffsetVal ),
				RFCAST( &GLview::setXOffsetVal )
				),
		new ValueFinfo( "yoffset_val",
				ValueFtype1< unsigned int >::global(),
				GFCAST( &GLview::getYOffsetVal ),
				RFCAST( &GLview::setYOffsetVal )
				),
		new ValueFinfo( "zoffset_val",
				ValueFtype1< unsigned int >::global(),
				GFCAST( &GLview::getZOffsetVal ),
				RFCAST( &GLview::setZOffsetVal )
				),
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		//		parser,
		process,
	};

	// Schedule molecules for the slower clock, stage 0.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static string doc[] = 
	{
		"Name", "GLview",
		"Author", "Karan Vasudeva, 2009, NCBS",
		"Description", "GLview: class to drive the spatial map widget",
	};

	static Cinfo glviewCinfo(
				 doc,
				 sizeof( doc ) / sizeof( string ),
				 initNeutralCinfo(),
				 GLviewFinfos,
				 sizeof( GLviewFinfos ) / sizeof( Finfo * ),
				 ValueFtype1< GLview >::global(),
				 schedInfo, 1
				 );

	return &glviewCinfo;
}

static const Cinfo* glviewCinfo = initGLviewCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

GLview::GLview()
	:
	sockFd_( -1 ),
	isConnectionUp_( false ),
	strClientHost_( "localhost" ),
	strClientPort_( "" ),
	syncMode_( false ),
	gridMode_( false ),
	bgcolorRed_( 0.0 ),
	bgcolorGreen_( 0.0 ),
	bgcolorBlue_( 0.0 ),
	strPath_( "" ),
	strRelPath_( "" ),
	color_val_( 0 ),
	morph_val_( 0 ),
	xoffset_val_( 0 ),
	yoffset_val_( 0 ),
	zoffset_val_( 0 ),
	x_( NULL ),
	y_( NULL ),
	z_( NULL )
{
	for ( unsigned int i = 0; i < 5; ++i )
	{
		values_[i] = NULL;
		value_min_[i] = VALUE_MIN_DEFAULT;
		value_max_[i] = VALUE_MAX_DEFAULT;
		strValueField_[i] = "";
	}

#ifdef WIN32
	if ( initWinsock() < 0 )
	{
		std::cerr << "Winsock could not be initialized. Cannot connect to client." << std::endl;
	}
#endif
}

GLview::~GLview()
{
	disconnect();
		
#ifdef WIN32
	WSACleanup();
#endif

	for ( unsigned int i = 0; i < 5; ++i )
	  free ( values_[i] );
	
	free ( x_ );
	free ( y_ );
	free ( z_ );

	if ( ! mapId2GLshapeData_.empty() )
	{
		std::map< unsigned int, GLshapeData* >::iterator id2glshapeIterator;			
		for ( id2glshapeIterator = mapId2GLshapeData_.begin();
		      id2glshapeIterator != mapId2GLshapeData_.end();
		      id2glshapeIterator++ )
		{
			free ( id2glshapeIterator->second );
		}
		
		mapId2GLshapeData_.clear();
	}
	
	vecElements_.clear();
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void GLview::setClientHost( const Conn* c, string strClientHost )
{
	static_cast< GLview * >( c->data() )->innerSetClientHost( strClientHost );
}

void GLview::innerSetClientHost( const string& strClientHost )
{
	strClientHost_ = strClientHost;
}

string GLview::getClientHost( Eref e )
{
	return static_cast< const GLview* >( e.data() )->strClientHost_;
}

void GLview::setClientPort( const Conn* c, string strClientPort )
{
	static_cast< GLview * >( c->data() )->innerSetClientPort( strClientPort );
}

void GLview::innerSetClientPort( const string& strClientPort )
{
	strClientPort_ = strClientPort;
}

string GLview::getClientPort( Eref e )
{
	return static_cast< const GLview* >( e.data() )->strClientPort_;
}

void GLview::setPath( const Conn* c, string strPath )
{
	static_cast< GLview * >( c->data() )->innerSetPath( strPath );
}

void GLview::innerSetPath( const string& strPath )
{
	strPath_ = strPath;
}

string GLview::getPath( Eref e )
{
	return static_cast< const GLview * >( e.data() )->strPath_;
}

void GLview::setRelPath( const Conn* c, string strRelPath )
{
	static_cast< GLview * >( c->data() )->innerSetRelPath( strRelPath );
}

void GLview::innerSetRelPath( const string& strRelPath )
{
	strRelPath_ = strRelPath;
}

string GLview::getRelPath( Eref e )
{
	return static_cast< GLview * >( e.data() )->strRelPath_;
}

void GLview::setValue1Field( const Conn* c, string strValue1Field )
{
	static_cast< GLview * >( c->data() )->innerSetValue1Field( strValue1Field );
}

void GLview::innerSetValue1Field( const string& strValue1Field )
{
	strValueField_[0] = strValue1Field;
}

string GLview::getValue1Field( Eref e )
{
	return static_cast< GLview * >( e.data() )->strValueField_[0];
}

void GLview::setValue2Field( const Conn* c, string strValue2Field )
{
	static_cast< GLview * >( c->data() )->innerSetValue2Field( strValue2Field );
}

void GLview::innerSetValue2Field( const string& strValue2Field )
{
	strValueField_[1] = strValue2Field;
}

string GLview::getValue2Field( Eref e )
{
	return static_cast< GLview * >( e.data() )->strValueField_[1];
}

void GLview::setValue3Field( const Conn* c, string strValue3Field )
{
	static_cast< GLview * >( c->data() )->innerSetValue3Field( strValue3Field );
}

void GLview::innerSetValue3Field( const string& strValue3Field )
{
	strValueField_[2] = strValue3Field;
}

string GLview::getValue3Field( Eref e )
{
	return static_cast< GLview * >( e.data() )->strValueField_[2];
}

void GLview::setValue4Field( const Conn* c, string strValue4Field )
{
	static_cast< GLview * >( c->data() )->innerSetValue4Field( strValue4Field );
}

void GLview::innerSetValue4Field( const string& strValue4Field )
{
	strValueField_[3] = strValue4Field;
}

string GLview::getValue4Field( Eref e )
{
	return static_cast< GLview * >( e.data() )->strValueField_[3];
}

void GLview::setValue5Field( const Conn* c, string strValue5Field )
{
	static_cast< GLview * >( c->data() )->innerSetValue5Field( strValue5Field );
}

void GLview::innerSetValue5Field( const string& strValue5Field )
{
	strValueField_[4] = strValue5Field;
}

string GLview::getValue5Field( Eref e )
{
	return static_cast< GLview * >( e.data() )->strValueField_[4];
}

void GLview::setBgColor( const Conn* c, string strBgColor )
{
	double red, green, blue;

	bool error = false;
	int bgcolor;
	std::istringstream intstream( strBgColor );
	if ( intstream >> bgcolor )
	{
		blue = ( bgcolor % 1000 ) / 255.;
		green = ( ( bgcolor/1000 ) % 1000 ) / 255.;
		red = ( ( bgcolor/1000000 ) % 1000 ) / 255.;

		if ( red > 1.0 || blue > 1.0 || green > 1.0 )
		{
			error = true;
		}
		else
		{
			static_cast< GLview * >( c->data() )->innerSetBgColor( red, green, blue );
		}
	}
	else
	{
		error = true;
	}	

	if ( error ) // report error; default is (0,0,0) (black)
	{
		std::cerr << "GLview error: the field 'bgcolor' is not in the expected format, defaulting to black" << std::endl;
	}
}

void GLview::innerSetBgColor( const double red, const double green, const double blue )
{
	bgcolorRed_ = red;
	bgcolorGreen_ = green;
	bgcolorBlue_ = blue;
}

string GLview::getBgColor( Eref e )
{
	double red = static_cast< const GLview* >( e.data() )->bgcolorRed_;
	double green = static_cast< const GLview* >( e.data() )->bgcolorGreen_;
	double blue = static_cast< const GLview* >( e.data() )->bgcolorBlue_;
	
	int bgcolor = (red * 255.) * 1000000 + (green * 255.) * 1000 + (blue * 255.);

	std::string s;
	std::stringstream out;
	out << bgcolor;
	return out.str();
}

void GLview::setSyncMode( const Conn* c, string syncMode )
{
	if ( syncMode == string( "on" ) )
		static_cast< GLview * >( c->data() )->innerSetSyncMode( true );
	else if ( syncMode == string( "off" ) )
		static_cast< GLview * >( c->data() )->innerSetSyncMode( false );
	else
		std::cerr << "GLview error: cannot set sync mode; argument must be either 'on' or 'off'." << std::endl;
}

void GLview::innerSetSyncMode( const bool syncMode )
{
	syncMode_ = syncMode;
}

string GLview::getSyncMode( Eref e )
{
	bool currentSyncMode = static_cast< const GLview* >( e.data() )->syncMode_;

	if ( currentSyncMode )
		return string( "on" );
	else
		return string( "off" );
}

void GLview::setGridMode( const Conn* c, string gridMode )
{
	if ( gridMode == string( "on" ) )
		static_cast< GLview * >( c->data() )->innerSetGridMode( true );
	else if ( gridMode == string( "off" ) )
		static_cast< GLview * >( c->data() )->innerSetGridMode( false );
	else
		std::cerr << "GLview error: cannot set grid mode; argument must be either 'on' or 'off'." << std::endl;
}

void GLview::innerSetGridMode( const bool gridMode )
{
	gridMode_ = gridMode;
}

string GLview::getGridMode( Eref e )
{
	bool currentGridMode = static_cast< const GLview* >( e.data() )->gridMode_;

	if ( currentGridMode )
		return string( "on" );
	else
		return string( "off" );
}

void GLview::setColorVal( const Conn* c, unsigned int colorVal )
{
	static_cast< GLview * >( c->data() )->innerSetColorVal( colorVal );
}

void GLview::innerSetColorVal( unsigned int colorVal )
{
	color_val_ = colorVal;
}

unsigned int GLview::getColorVal( Eref e )
{
	return static_cast< const GLview* >( e.data() )->color_val_;
}

void GLview::setMorphVal( const Conn* c, unsigned int morphVal )
{
	static_cast< GLview * >( c->data() )->innerSetMorphVal( morphVal );
}

void GLview::innerSetMorphVal( unsigned int morphVal )
{
	morph_val_ = morphVal;
}

unsigned int GLview::getMorphVal( Eref e )
{
	return static_cast< const GLview* >( e.data() )->morph_val_;
}

void GLview::setXOffsetVal( const Conn* c, unsigned int xoffsetVal )
{
	static_cast< GLview * >( c->data() )->innerSetXOffsetVal( xoffsetVal );
}

void GLview::innerSetXOffsetVal( unsigned int xoffsetVal )
{
	xoffset_val_ = xoffsetVal;
}

unsigned int GLview::getXOffsetVal( Eref e )
{
	return static_cast< const GLview* >( e.data() )->xoffset_val_;
}

void GLview::setYOffsetVal( const Conn* c, unsigned int yoffsetVal )
{
	static_cast< GLview * >( c->data() )->innerSetYOffsetVal( yoffsetVal );
}

void GLview::innerSetYOffsetVal( unsigned int yoffsetVal )
{
	yoffset_val_ = yoffsetVal;
}

unsigned int GLview::getYOffsetVal( Eref e )
{
	return static_cast< const GLview* >( e.data() )->yoffset_val_;
}

void GLview::setZOffsetVal( const Conn* c, unsigned int zoffsetVal )
{
	static_cast< GLview * >( c->data() )->innerSetZOffsetVal( zoffsetVal );
}

void GLview::innerSetZOffsetVal( unsigned int zoffsetVal )
{
	zoffset_val_ = zoffsetVal;
}

unsigned int GLview::getZOffsetVal( Eref e )
{
	return static_cast< const GLview* >( e.data() )->zoffset_val_;
}

void GLview::setValue1Min( const Conn* c, double value1Min )
{
	static_cast< GLview * >( c->data() )->innerSetValue1Min( value1Min );
}

void GLview::innerSetValue1Min( const double& value1Min )
{
	value_min_[0] = value1Min;
}

double GLview::getValue1Min( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_min_[0];
}

void GLview::setValue1Max( const Conn* c, double value1Max )
{
	static_cast< GLview * >( c->data() )->innerSetValue1Max( value1Max );
}

void GLview::innerSetValue1Max( const double& value1Max )
{
	value_max_[0] = value1Max;
}

double GLview::getValue1Max( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_max_[0];
}

void GLview::setValue2Min( const Conn* c, double value2Min )
{
	static_cast< GLview * >( c->data() )->innerSetValue2Min( value2Min );
}

void GLview::innerSetValue2Min( const double& value2Min )
{
	value_min_[1] = value2Min;
}

double GLview::getValue2Min( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_min_[1];
}

void GLview::setValue2Max( const Conn* c, double value2Max )
{
	static_cast< GLview * >( c->data() )->innerSetValue2Max( value2Max );
}

void GLview::innerSetValue2Max( const double& value2Max )
{
	value_max_[1] = value2Max;
}

double GLview::getValue2Max( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_max_[1];
}

void GLview::setValue3Min( const Conn* c, double value3Min )
{
	static_cast< GLview * >( c->data() )->innerSetValue3Min( value3Min );
}

void GLview::innerSetValue3Min( const double& value3Min )
{
	value_min_[2] = value3Min;
}

double GLview::getValue3Min( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_min_[2];
}

void GLview::setValue3Max( const Conn* c, double value3Max )
{
	static_cast< GLview * >( c->data() )->innerSetValue3Max( value3Max );
}

void GLview::innerSetValue3Max( const double& value3Max )
{
	value_max_[2] = value3Max;
}

double GLview::getValue3Max( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_max_[2];
}

void GLview::setValue4Min( const Conn* c, double value4Min )
{
	static_cast< GLview * >( c->data() )->innerSetValue4Min( value4Min );
}

void GLview::innerSetValue4Min( const double& value4Min )
{
	value_min_[3] = value4Min;
}

double GLview::getValue4Min( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_min_[3];
}

void GLview::setValue4Max( const Conn* c, double value4Max )
{
	static_cast< GLview * >( c->data() )->innerSetValue4Max( value4Max );
}

void GLview::innerSetValue4Max( const double& value4Max )
{
	value_max_[3] = value4Max;
}

double GLview::getValue4Max( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_max_[3];
}

void GLview::setValue5Min( const Conn* c, double value5Min )
{
	static_cast< GLview * >( c->data() )->innerSetValue5Min( value5Min );
}

void GLview::innerSetValue5Min( const double& value5Min )
{
	value_min_[4] = value5Min;
}

double GLview::getValue5Min( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_min_[4];
}

void GLview::setValue5Max( const Conn* c, double value5Max )
{
	static_cast< GLview * >( c->data() )->innerSetValue5Max( value5Max );
}

void GLview::innerSetValue5Max( const double& value5Max )
{
	value_max_[4] = value5Max;
}

double GLview::getValue5Max( Eref e )
{
	return static_cast< const GLview* >( e.data() )->value_max_[4];
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void GLview::reinitFunc( const Conn* c, ProcInfo info )
{
	static_cast< GLview * >( c->data() )->reinitFuncLocal( c );
}

void GLview::reinitFuncLocal( const Conn* c )
{
	// If this element has no children yet:
	Id id = c->target().id();
	Conn* i = id()->targets( "childSrc", 0 );
	if ( ! i->good() )
	{
		// create default shape templates
		Neutral::createArray( "GLshape", "shape", id, Id::scratchId(), 2 );

		// set default values for the new shapes
		vector< Eref > ret;
		children( id, ret, "GLshape" );

		::set< double >( ret[0], "color", 0.0 );
		::set< double >( ret[0], "len", 0.0 );
		::set< double >( ret[1], "color", 1.0 );
		::set< double >( ret[1], "len", 1.0 );
	}
	delete i;

	// determine child elements of the type GLshape as possible interpolation targets
	vecErefGLshapeChildren_.clear();
	children( id, vecErefGLshapeChildren_, "GLshape" );

	// set shapetype to that of the first interpolation target
	int shapetype;
	get< int >( vecErefGLshapeChildren_[0], "shapetype", shapetype );

	if ( ! strPath_.empty() )  
	{
		vecElements_.clear();
		wildcardFind( strPath_, vecElements_ );
		std::cout << "GLview: " << vecElements_.size() << " elements found." << std::endl; 

		// (re) allocate memory (because vecElements_.size() might have changed)
		if ( mapId2GLshapeData_.size() != 0 )
		{
			std::map< unsigned int, GLshapeData* >::iterator id2glshapeIterator;			
			for ( id2glshapeIterator = mapId2GLshapeData_.begin();
			      id2glshapeIterator != mapId2GLshapeData_.end();
			      id2glshapeIterator++ )
			{
				free ( id2glshapeIterator->second );
			}

			mapId2GLshapeData_.clear();
		}
		for ( unsigned int i = 0; i < vecElements_.size(); ++i )
		{
			unsigned int id = vecElements_[i].id();
			GLshapeData* temp = (GLshapeData *) malloc( sizeof( GLshapeData ) );
			if ( temp == NULL )
			{
				std::cerr << "GLview: could not allocate memory!" << std::endl;
				return;
			}
			else 
			{
				mapId2GLshapeData_[id] = temp;
			}
		}

		double maxsize = populateXYZ();

		GLviewResetData resetData;
		resetData.bgcolorRed = bgcolorRed_;
		resetData.bgcolorGreen = bgcolorGreen_;
		resetData.bgcolorBlue = bgcolorBlue_;
		resetData.strPathName = strPath_;
		resetData.maxsize = maxsize;
		
		for ( unsigned int i = 0; i < vecElements_.size(); ++i )
		{
			GLviewShapeResetData shape;

			shape.id = vecElements_[i].id();
			shape.strPathName = vecElements_[i].path();
			shape.x = x_[i];
			shape.y = y_[i];
			shape.z = z_[i];
			shape.shapetype = shapetype;

			resetData.vecShapes.push_back( shape );
		}

		if ( strClientPort_.empty() )
			std::cerr << "GLview error: Client port not specified." << std::endl;
		else if ( strClientHost_.empty() )
			std::cerr << "GLview error: Client hostname not specified." << std::endl;
		else
			transmit( resetData, RESET );
	}
}

void GLview::processFunc( const Conn* c, ProcInfo info )
{
	static_cast< GLview * >( c->data() )->processFuncLocal( c->target(), info );
}

void GLview::processFuncLocal( Eref e, ProcInfo info )
{
	if ( vecErefGLshapeChildren_.size() < 2 )
	{
		std::cerr << "GLview error: should have at least two child elements of type GLshape" << std::endl;
		return;
	}

	// set parameters to default values
	for ( unsigned int i = 0; i < vecElements_.size(); ++i )
	{
		unsigned int id = vecElements_[i].id();
		mapId2GLshapeData_[id]->color = -1.; // -1 signifies no change in this variable
		mapId2GLshapeData_[id]->xoffset = 0;
		mapId2GLshapeData_[id]->yoffset = 0.;
		mapId2GLshapeData_[id]->zoffset = 0.;
		mapId2GLshapeData_[id]->len = -1.; // -1 signifies no change in this variable
	}

	// refresh values_[][]
	for ( unsigned int i = 0; i < 5; ++i )
	{
		if ( ! strValueField_[i].empty() )
		{
			if ( populateValues( i+1, &values_[i], strValueField_[i] ) < 0 )
			{
				return;
			}
		}			
	}

	// obtain parameter values by linear interpolation between
	// values of respective interpolation templates
	if ( color_val_ > 0 && color_val_ <= 5 )
	{
		for ( unsigned int i = 0; i < vecElements_.size(); ++i)
		{
			unsigned int id = vecElements_[i].id();
			double value = values_[color_val_-1][i];

			// determine interpolation targets
			unsigned int iLow, iHigh;
			chooseInterpolationPair( vecErefGLshapeChildren_.size(), value,
						 value_min_[color_val_-1], value_max_[color_val_-1],
						 iLow, iHigh);
			// obtain parameter value by linear interpolation and set
			double attr_low;
			get< double >( vecErefGLshapeChildren_[iLow], "color", attr_low );
			double attr_high;
			get< double >( vecErefGLshapeChildren_[iHigh], "color", attr_high );

			interpolate( value_min_[color_val_-1], attr_low,
				     value_max_[color_val_-1], attr_high,
				     value, mapId2GLshapeData_[id]->color );
		}
	}

	if ( morph_val_ > 0 && morph_val_ <= 5 )
	{
		for ( unsigned int i = 0; i < vecElements_.size(); ++i)
		{
			unsigned int id = vecElements_[i].id();
			double value = values_[morph_val_-1][i];

			// determine interpolation targets
			unsigned int iLow, iHigh;
			chooseInterpolationPair( vecErefGLshapeChildren_.size(), value,
						 value_min_[morph_val_-1], value_max_[morph_val_-1],
						 iLow, iHigh);
			// obtain parameter value by linear interpolation and set
			double attr_low;
			get< double >( vecErefGLshapeChildren_[iLow], "len", attr_low );
			double attr_high;
			get< double >( vecErefGLshapeChildren_[iHigh], "len", attr_high );

			interpolate( value_min_[morph_val_-1], attr_low,
				     value_max_[morph_val_-1], attr_high,
				     value, mapId2GLshapeData_[id]->len );
		}
	}

	if ( xoffset_val_ > 0 && xoffset_val_ <= 5 )
	{
		for ( unsigned int i = 0; i < vecElements_.size(); ++i)
		{
			unsigned int id = vecElements_[i].id();
			double value = values_[xoffset_val_-1][i];

			// determine interpolation targets
			unsigned int iLow, iHigh;
			chooseInterpolationPair( vecErefGLshapeChildren_.size(), value,
						 value_min_[xoffset_val_-1], value_max_[xoffset_val_-1],
						 iLow, iHigh);
			// obtain parameter value by linear interpolation and set
			double attr_low;
			get< double >( vecErefGLshapeChildren_[iLow], "xoffset", attr_low );
			double attr_high;
			get< double >( vecErefGLshapeChildren_[iHigh], "xoffset", attr_high );

			interpolate( value_min_[xoffset_val_-1], attr_low,
				     value_max_[xoffset_val_-1], attr_high,
				     value, mapId2GLshapeData_[id]->xoffset );
		}
	}

	if ( yoffset_val_ > 0 && yoffset_val_ <= 5 )
	{
		for ( unsigned int i = 0; i < vecElements_.size(); ++i)
		{
			unsigned int id = vecElements_[i].id();
			double value = values_[yoffset_val_-1][i];

			// determine interpolation targets
			unsigned int iLow, iHigh;
			chooseInterpolationPair( vecErefGLshapeChildren_.size(), value,
						 value_min_[yoffset_val_-1], value_max_[yoffset_val_-1],
						 iLow, iHigh);
			// obtain parameter value by linear interpolation and set
			double attr_low;
			get< double >( vecErefGLshapeChildren_[iLow], "yoffset", attr_low );
			double attr_high;
			get< double >( vecErefGLshapeChildren_[iHigh], "yoffset", attr_high );

			interpolate( value_min_[yoffset_val_-1], attr_low,
				     value_max_[yoffset_val_-1], attr_high,
				     value, mapId2GLshapeData_[id]->yoffset );
		}
	}

	if ( zoffset_val_ > 0 && zoffset_val_ <= 5 )
	{
		for ( unsigned int i = 0; i < vecElements_.size(); ++i)
		{
			unsigned int id = vecElements_[i].id();
			double value = values_[zoffset_val_-1][i];

			// determine interpolation targets
			unsigned int iLow, iHigh;
			chooseInterpolationPair( vecErefGLshapeChildren_.size(), value,
						 value_min_[zoffset_val_-1], value_max_[zoffset_val_-1],
						 iLow, iHigh);
			// obtain parameter value by linear interpolation and set
			double attr_low;
			get< double >( vecErefGLshapeChildren_[iLow], "zoffset", attr_low );
			double attr_high;
			get< double >( vecErefGLshapeChildren_[iHigh], "zoffset", attr_high );

			interpolate( value_min_[zoffset_val_-1], attr_low,
				     value_max_[zoffset_val_-1], attr_high,
				     value, mapId2GLshapeData_[id]->zoffset );
		}
	}

	if ( syncMode_ )
	{
		transmit( mapId2GLshapeData_, PROCESS_COLORS_SYNC );
		receiveAck();
		// The client will wait for the display to be updated before
		// sending this ack in response to a PROCESS_COLORS_SYNC message.
	}
	else
	{
		transmit( mapId2GLshapeData_, PROCESS_COLORS );
		receiveAck();
	}
}

///////////////////////////////////////////////////
// private function definitions
///////////////////////////////////////////////////

int GLview::populateValues( int valueNum, double ** pValues, const string& strValueField )
{
	int status = 0;

	if ( *pValues == NULL )
		*pValues = ( double * ) malloc( sizeof( double ) * vecElements_.size() );
	
	if ( *pValues == NULL ) // if it's still NULL
	{
		std::cerr << "GLview error: could not allocate memory to set field values" << std::endl;
		status = -1;
	}
	else
	{
		double * values = *pValues;

		for ( unsigned int i = 0; i < vecElements_.size(); ++i)
		{
			Id id = vecElements_[i];
			std::string path;

			if ( ! strRelPath_.empty() ) 
			{
				path = vecElements_[i].path();
				path.push_back('/');
				path.append(strRelPath_);
				id = Id::Id( path, "/" );
			}
			
			if ( id.eref() == NULL )
			{
				std::cerr << "GLview error: could not find vizpath: " << path << "; error in relpath? " << std::endl;
				status = -2;
				break;
			}
			else if ( id.eref().e->findFinfo( strValueField ) )
			{
				get< double >( id.eref(), strValueField, values[i] );
			}
			else
			{
				std::cerr << "GLview error: for value" << valueNum << ", unable to find a field called '" << strValueField << "' in " << id.path() << std::endl;
				status = -3;
				break;
			}
			/* else
			   {
			   std::cout << "field1 value " << strValueField << " for element " << i << " is " << values[i] << std::endl;
			   } */
		}
	}

	if ( status == -2 || status == -3 )
	{
		free( *pValues );
		*pValues = NULL;
	}

	return status;
}

int GLview::getXYZ( Id id, double& xout, double& yout, double& zout, double &maxsize )
{
	if ( id() == Element::root() )
		return -1;

	if ( ! id.eref().e->findFinfo( "x" ) ||
	     ! id.eref().e->findFinfo( "y" ) ||
	     ! id.eref().e->findFinfo( "z" ) ||
	     ! id.eref().e->findFinfo( "x0" ) ||
	     ! id.eref().e->findFinfo( "y0" ) ||
	     ! id.eref().e->findFinfo( "z0" ) ||
	     ! id.eref().e->findFinfo( "length") )
	{
		Id parent = Shell::parent( id );
		if ( parent == Id::badId() )
			return -1;
		else
			return getXYZ( parent, xout, yout, zout, maxsize ); // recurses
	}
	
	// success
	double x, y, z, x0, y0, z0, length;
	get< double >( id.eref(), "x", x );
	get< double >( id.eref(), "y", y );
	get< double >( id.eref(), "z", z );
	get< double >( id.eref(), "x0", x0 );
	get< double >( id.eref(), "y0", y0 );
	get< double >( id.eref(), "z0", z0 );
	get< double >( id.eref(), "length", length );
	std::string name = id.eref().name();
	
	if ( length < SIZE_EPSILON ||
	     name.compare("soma") == 0 )
	{
		xout = x;
		yout = y;
		zout = z;
	}
	else
	{
		xout = ( x0 + x ) / 2;
		yout = ( y0 + y ) / 2;
		zout = ( z0 + z ) / 2;
	}
	
	// determining maxsize_
	double maxsize_ = 0, temp;
	if ( id.eref().e->findFinfo( "diameter") &&
	     get< double >( id.eref(), "diameter", temp ) &&
	     temp > maxsize_)
	{
		maxsize_ = temp;
	}
	if ( id.eref().e->findFinfo( "length") &&
	     get< double >( id.eref(), "length", temp ) &&
	     temp > maxsize_)
	{
		maxsize_ = temp;
	}
	maxsize = maxsize_;

	return 0;
}

double GLview::populateXYZ()
{
	if ( x_ == NULL )
		x_ = ( double * ) malloc( sizeof( double ) * vecElements_.size() );
	if ( y_ == NULL )
		y_ = ( double * ) malloc( sizeof( double ) * vecElements_.size() );
	if ( z_ == NULL )
		z_ = ( double * ) malloc( sizeof( double ) * vecElements_.size() );

	double x, y, z, size, maxsize = 0;
	vector< unsigned int > unassignedShapes;
	double bbx = 0;
	double bby = 0;
	double bbz = 0;

	// If the field 'grid' is 'off', there will be two steps to
	// determine collision-free x,y,z co-ordinates (to 6 decimal
	// places). This procedure also determines the maximum length
	// in any dimension of shapes with given geometries so that
	// the remaining shapes (with no geometrical basis) can be
	// assigned sizes on the same scale of size; this is based on
	// the assumption that most geometrical shapes in the
	// simulation will be laid out to be non-overlapping and
	// therefore the maximum length provides an approximation of
	// the typical distance between elements.
	//
	// If the field 'grid' is 'on', maxsize will be set to 1 and
	// all shapes will be forced into a grid layout.

	if ( gridMode_ )
	{
		for ( unsigned int i = 0; i < vecElements_.size(); ++i )
		{
			unassignedShapes.push_back( i );
		}
	}
	else
	{
		// 1. We get x,y,z from elements or their first
		// non-root ancestor that have valid x,y,z values. We
		// check these into a map, intending to separate
		// shapes corresponding to elements specified with
		// duplicate x,y,z co-ordinates.

		map< string, unsigned int > mapXYZ;
		for ( unsigned int i = 0; i < vecElements_.size(); ++i )
		{
			if ( getXYZ( vecElements_[i], x, y, z, size ) == 0 )
			{
				string key = boxXYZ( x, y, z );
				if ( mapXYZ.count( key ) == 0 )
					mapXYZ[ key ] = 1;
				else
					mapXYZ[ key ] += 1;

				if ( size > maxsize )
					maxsize = size;
			}
		}

		// 2. We finalize non-duplicate x,y,z co-ordinates and place
		// the rest on a list that will be automatically assigned sane
		// co-ordinates just outside the bounding box of the first
		// group. We also determine a corner of this bounding box to
		// act as the starting location of the second group.

		for ( unsigned int i = 0; i < vecElements_.size(); ++i )
		{
			if ( getXYZ( vecElements_[i], x, y, z, size ) == 0 )
			{
				string key = boxXYZ( x, y, z );
				if ( mapXYZ[key] > 1 ) // collision
				{
					unassignedShapes.push_back( i );
				}
				else
				{
					x_[i] = x;
					y_[i] = y;
					z_[i] = z;
				
					if ( bbx < x )
						bbx = x + maxsize;
					if ( bby < y )
						bby = y + maxsize;
					if ( bbz < z )
						bbz = z + maxsize;
				}
			}
			else
			{
				unassignedShapes.push_back( i );
			}
		}
	}

	// We take the starting location calculated in the last
	// step and assign co-ordinates to all shapes on the
	// (collision/unassigned) list as described in the last
	// step. These co-ordinates will be assigned to lay out all
	// elements in a planar grid, approximating a square in shape.

	int n = (int)( sqrt( (float) unassignedShapes.size() ) );

	// Finally, if maxsize is still zero, i.e., no non-root
	// ancestors with valid geometries were found, we must still
	// set the size to an arbitrary non-zero value, so we use 1.
	//
	// This will always apply if the field 'grid' is 'on'.

	if ( maxsize < FP_EPSILON )
	{
		maxsize = 1;
	}

	for ( unsigned int j = 0; j < unassignedShapes.size(); ++j )
	{
		unsigned int i = unassignedShapes[j];
		
		x_[i] = bbx + (j % n) * maxsize;
		y_[i] = bby + (j / n) * maxsize;
		z_[i] = bbz;
	}

	return maxsize;
} 

string GLview::boxXYZ( const double& x, const double& y, const double& z )
{
	string key( inttostring( int( x * 1e8 ) ) );
	key.append( inttostring( int( y * 1e8 ) ) );
	key.append( inttostring( int( z * 1e8 ) ) );

	return key;
}

string GLview::inttostring( int i )
{
	std::string s;
	std::stringstream out;
	out << i;
	return out.str();
}

int GLview::children( Id object, vector< Eref >& ret, const string& type )
{
	unsigned int oldSize = ret.size();
	
	Eref found;
	Conn* i = object()->targets( "childSrc", 0 );
	for ( ; i->good(); i->increment() ) {
		found = i->target();
		if ( ! isType( found->id(), type ) )
			continue;
		
		ret.push_back( found );		
	}
	delete i;
	
	return ret.size() - oldSize;
}

bool GLview::isType( Id object, const string& type )
{
	return object()->cinfo()->isA( Cinfo::find( type ) );
}

void GLview::chooseInterpolationPair( const int& numTargets, const double& val,
				      const double& val_min, const double& val_max,
				      unsigned int& iLow, unsigned int& iHigh )
{
	if ( numTargets == 2 )
	{
		iLow = 0;
		iHigh = 1;
	}
	else
	{
		double step = 1. / (numTargets - 1);
		int low, high;

		low = static_cast< int >(floor((val-val_min) / ((val_max-val_min) * step) ));
		if ( low >= numTargets - 1 )
			low = numTargets - 2;
		else if ( low < 0 )
			low = 0;

		high = low + 1;

		iLow = low;
		iHigh = high;
	}
}

void GLview::interpolate( const double& val_min, const double& attr_min,
			  const double& val_max, const double& attr_max,
			  const double& val, double& attr )
{
	attr = attr_min +
		((val - val_min) * (attr_max - attr_min)) / ( val_max - val_min );
}

///////////////////////////////////////////////////
// networking helper function definitions
///////////////////////////////////////////////////

void* GLview::getInAddress( struct sockaddr *sa )
{
	if ( sa->sa_family == AF_INET ) {
		return &( ( ( struct sockaddr_in* )sa )->sin_addr );
	}

	return &( ( ( struct sockaddr_in6* )sa )->sin6_addr );
}

int GLview::getSocket( const char* hostname, const char* service )
{
	struct addrinfo hints, *servinfo, *p;
	int rv;
	char s[INET6_ADDRSTRLEN];

#ifdef WIN32
	unsigned int socket_error = INVALID_SOCKET;
	unsigned int connect_error = SOCKET_ERROR;
#else
	int socket_error = -1;
	int connect_error = -1;
#endif
	
	memset( &hints, 0, sizeof hints );
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;

	if ( ( rv = getaddrinfo( hostname, service, &hints, &servinfo ) ) != 0 ) {
		std::cerr << "GLview error: getaddrinfo: " << gai_strerror( rv ) << std::endl;
		return -1;
	}
	// loop through all the results and connect to the first we can
	for ( p = servinfo; p != NULL; p = p->ai_next ) {
		if ( ( sockFd_ = socket( p->ai_family, p->ai_socktype,
				     p->ai_protocol ) ) == socket_error ) {
		    //std::cerr << "GLview error: socket" << std::endl;
			continue;
		}
		
		if ( connect( sockFd_, p->ai_addr, p->ai_addrlen ) == connect_error ) {
#ifdef WIN32
			closesocket( sockFd_ );
#else
			close( sockFd_ );
#endif
			//std::cerr << "GLview error: connect" << std::endl;
			continue;
		}
		
		break;
	}

	if ( p == NULL ) {
		std::cerr << "GLview error: failed to connect" << std::endl;
		return -1;
	}
	
	/*inet_ntop( p->ai_family, getInAddress( ( struct sockaddr * )p->ai_addr ),
		   s, sizeof s );*/
	// std::cout << "Connecting to " << s << std::endl;
	
	freeaddrinfo( servinfo );
	
	isConnectionUp_ = true;
	return sockFd_;
}

int GLview::sendAll( int socket, char* buf, unsigned int* len )
{
	unsigned int total = 0;        // how many bytes we've sent
	int bytesleft = *len; // how many we have left to send
	int n = 0;

#ifdef WIN32
	unsigned int send_error = SOCKET_ERROR;
#else
	int send_error = -1;
#endif

	while ( total < *len )
	{
		n = send( socket, buf+total, bytesleft, 0 );
		if ( n == send_error )
		{
			std::cerr << "GLview error: send error; errno: " << errno << " " << strerror( errno ) << std::endl;    
			break;
		}
		total += n;
		bytesleft -= n;
	}

	*len = total; // return number actually sent here

	return n == -1 ? -1 : 0; // return -1 on failure, 0 on success
}

int GLview::recvAll( int socket, char* buf, unsigned int* len)
{
	unsigned int total = 0;        // how many bytes we've received
	int bytesleft = *len; // how many we have left to receive
	int n = 0;

#ifdef WIN32
	unsigned int recv_error = SOCKET_ERROR;
#else
	int recv_error = -1;
#endif
	
	while ( total < *len )
	{
		n = recv( socket, buf+total, bytesleft, 0 );
		if ( n == recv_error )
		{
			std::cerr << "GLview error: recv error; errno: " << errno << " " << strerror( errno ) << std::endl;
			break;
		}
		total += n;
		bytesleft -= n;
	}
	
	*len = total; /// return number actually received here
	
	return n == -1 ? -1 : 0; // return -1 on failure, 0 on success
}

int GLview::receiveAck()
{
	if ( ! isConnectionUp_ )
	{
		std::cerr << "Could not receive ACK because the connection is down." << std::endl;
		return -1;
	}

	unsigned int numBytes, inboundDataSize;
	char header[MSGSIZE_HEADERLENGTH + 1];
	char *buf;

	numBytes = sizeof( AckPickData ) + 1;
	buf = ( char * ) malloc( numBytes );

	numBytes = MSGSIZE_HEADERLENGTH + 1;
	if ( recvAll( sockFd_, header, &numBytes) == -1 ||
	     numBytes < MSGSIZE_HEADERLENGTH + 1 )
	{
		std::cerr << "GLview error: could not receive Ack header!" << std::endl;
		isConnectionUp_ = false;
		return -1;
	}
	else
	{
		std::istringstream ackHeaderStream( std::string( header,
								 MSGSIZE_HEADERLENGTH ) );
		ackHeaderStream >> std::hex >> inboundDataSize;
	}

	numBytes = inboundDataSize + 1;
	buf = ( char * ) malloc( numBytes * sizeof( char ) );

	if ( recvAll( sockFd_, buf, &numBytes ) == -1 ||
	     numBytes < inboundDataSize + 1 )
	{
		std::cerr << "GLview error: could not receive Ack!" << std::endl;
		isConnectionUp_ = false;
		return -2;
	}
	else
	{
		std::istringstream archiveStream( std::string( buf, inboundDataSize ) );

		// starting new scope so that the archive's stream's destructor is called after the archive's
		{
			boost::archive::text_iarchive archive( archiveStream );
		
			AckPickData ackPickData;		
			archive >> ackPickData;

			if ( ackPickData.wasSomethingPicked )
			{
				handlePick( ackPickData.idPicked );
			}
		}
	}
	free(buf);
	return 1;
}

void GLview::handlePick( unsigned int idPicked )
{
	std::cout << "GLview: Compartment with id " << idPicked << " was picked!" << std::endl;
}

void GLview::disconnect()
{
	if ( ! isConnectionUp_ )
	{
		sockFd_ = getSocket( strClientHost_.c_str(), strClientPort_.c_str() );
		if ( sockFd_ == -1 ) 
		{
			std::cerr << "GLview error: couldn't connect to client!" << std::endl;
			return;
		}
	}

	std::ostringstream headerStream;
	headerStream << std::setw( MSGSIZE_HEADERLENGTH ) << 0;
	headerStream << std::setw( MSGTYPE_HEADERLENGTH ) << DISCONNECT;

	unsigned int headerLen = headerStream.str().size() + 1;
	char* headerData = (char *) malloc( headerLen * sizeof( char ) );
	strcpy( headerData, headerStream.str().c_str() );


	if ( sendAll( sockFd_, headerData, &headerLen ) == -1 ||
	     headerLen < headerStream.str().size() + 1 )
	{
		std::cerr << "GLview error: couldn't transmit DISCONNECT message to client!" << std::endl;
	}

	free( headerData );
#ifdef WIN32
	closesocket( sockFd_ );
#else
	close( sockFd_ );
#endif
}

template< class T >
void GLview::transmit( T& data, MsgType messageType )
{
	if ( strClientHost_.empty() || strClientPort_.empty() ) // these should have been set.
		return;

	if ( ! isConnectionUp_ )
	{
		sockFd_ = getSocket( strClientHost_.c_str(), strClientPort_.c_str() );
		if ( sockFd_ == -1 ) 
		{
			std::cerr << "GLview error: Couldn't connect to client!" << std::endl;
			return;
		}
	}

	std::ostringstream archiveStream;

	// starting new scope so that the archive's stream's destructor is called after the archive's
	{
		boost::archive::text_oarchive archive( archiveStream );

		archive << data;

		std::ostringstream headerStream;
		headerStream << std::setw( MSGSIZE_HEADERLENGTH )
			     << std::hex << archiveStream.str().size();

		headerStream << std::setw( MSGTYPE_HEADERLENGTH )
			     << messageType;

		unsigned int headerLen = headerStream.str().size() + 1;
		char* headerData = ( char * ) malloc( headerLen * sizeof( char ) );
		strcpy( headerData, headerStream.str().c_str() );
	
		if ( sendAll( sockFd_, headerData, &headerLen ) == -1 ||
		     headerLen < headerStream.str().size() + 1 )
		{
			std::cerr << "GLview error: couldn't transmit header to client!" << std::endl;

			isConnectionUp_ = false;
#ifdef WIN32
			closesocket( sockFd_ );
#else
			close( sockFd_ );
#endif
		}
		else
		{
			unsigned int archiveLen = archiveStream.str().size() + 1;
			char* archiveData = ( char * ) malloc( archiveLen * sizeof( char ) );
			strcpy( archiveData, archiveStream.str().c_str() );
				
			if ( sendAll( sockFd_, archiveData, &archiveLen ) == -1 ||
			     archiveLen < archiveStream.str().size() + 1 )
			{
				std::cerr << "GLview error: couldn't transmit data to client!" << std::endl;	
			}
			free( archiveData );
		}
		free( headerData );
	}
}

#ifdef WIN32
int GLview::initWinsock( void )
{
	WORD wVersionRequested;
	WSADATA wsaData;
	int err;

	wVersionRequested = MAKEWORD(2, 2);

	err = WSAStartup(wVersionRequested, &wsaData);
	if (err != 0) 
	{
		std::cerr << "WSAStartup failed with error: ." << err << std::endl;
		return -1;
	}

	if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
	{
		std::cerr << "Could not find a usable version of Winsock.dll." << std::endl;
		WSACleanup();
		return -1;
	}

	return 0;
}
#endif
