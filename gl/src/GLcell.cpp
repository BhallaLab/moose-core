/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

////////////////////////////////////////////////////////////////////////////////
// The socket code is mostly taken from Beej's Guide to Network Programming   //
// at http://beej.us/guide/bgnet/. The original code is in the public domain. //
////////////////////////////////////////////////////////////////////////////////

#include "moose.h"
#include "shell/Shell.h"
#include "GLcell.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <string>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include <string.h>

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

#include "AckPickData.h"
#include "ParticleData.h"
#include "SmoldynShapeData.h"
#include "GLcellProcData.h"
#include "Constants.h"

#include "GLCompartmentCylinderData.h"
#include "GLCompartmentDiskData.h"
#include "GLCompartmentHemiData.h"
#include "GLCompartmentRectData.h"
#include "GLCompartmentSphereData.h"
#include "GLCompartmentTriData.h"

const int GLcell::MSGTYPE_HEADERLENGTH = 1;
const int GLcell::MSGSIZE_HEADERLENGTH = 8;
const char GLcell::SYNCMODE_ACKCHAR = '*';

const Cinfo* initGLcellCinfo()
{
	static Finfo* processShared[] = 
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &GLcell::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &GLcell::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ),
		"shared message to receive Process messages from scheduler objects" );

	static Finfo* GLcellFinfos[] = 
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "vizpath",
				ValueFtype1< string >::global(),
				GFCAST( &GLcell::getPath ),
				RFCAST( &GLcell::setPath )
				),
		new ValueFinfo( "host",
				ValueFtype1< string >::global(),
				GFCAST( &GLcell::getClientHost ),
				RFCAST( &GLcell::setClientHost )
				),
		new ValueFinfo( "port",
				ValueFtype1< string >::global(),
				GFCAST( &GLcell::getClientPort ),
				RFCAST( &GLcell::setClientPort )
				),
		new ValueFinfo( "attribute",
				ValueFtype1< string >::global(),
				GFCAST( &GLcell::getAttributeName ),
				RFCAST( &GLcell::setAttributeName )
				),
		new ValueFinfo( "threshold",
				ValueFtype1< double >::global(),
				GFCAST( &GLcell::getChangeThreshold ),
				RFCAST( &GLcell::setChangeThreshold )
				),
		new ValueFinfo( "vscale",
				ValueFtype1< double >::global(),
				GFCAST( &GLcell::getVScale ),
				RFCAST( &GLcell::setVScale )
				),
		new ValueFinfo( "sync",
				ValueFtype1< string >::global(),
				GFCAST( &GLcell::getSyncMode ),
				RFCAST( &GLcell::setSyncMode )
				),
		new ValueFinfo( "bgcolor",
				ValueFtype1< string >::global(),
				GFCAST( &GLcell::getBgColor ),
				RFCAST( &GLcell::setBgColor )
				),
		new ValueFinfo( "highvalue",
				ValueFtype1< double >::global(),
				GFCAST( &GLcell::getHighValue ),
				RFCAST( &GLcell::setHighValue )
				),
		new ValueFinfo( "lowvalue",
				ValueFtype1< double >::global(),
				GFCAST( &GLcell::getLowValue ),
				RFCAST( &GLcell::setLowValue )
				),
	///////////////////////////////////////////////////////
	// Message destination definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "particleData",
			       Ftype1< vector< ParticleData > >::global(),
			       RFCAST( &GLcell::setParticleData )
			       ),
		new DestFinfo( "smoldynShapeData",
			       Ftype1< vector< SmoldynShapeData > >::global(),
			       RFCAST( &GLcell::setSmoldynShapeData )
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
		"Name", "GLcell",
		"Author", "Karan Vasudeva, 2009, NCBS",
		"Description", "GLcell: class to drive the 3D cell visualization widget",
	};

	static Cinfo glcellCinfo(
				 doc,
				 sizeof( doc ) / sizeof( string ),
				 initNeutralCinfo(),
				 GLcellFinfos,
				 sizeof( GLcellFinfos ) / sizeof( Finfo * ),
				 ValueFtype1< GLcell >::global(),
				 schedInfo, 1
				 );

	return &glcellCinfo;
}

static const Cinfo* glcellCinfo = initGLcellCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

GLcell::GLcell()
	:
	strPath_( "" ),
	strClientHost_( "localhost" ),
	strClientPort_( "" ),
	isConnectionUp_( false ),
	strAttributeName_( "Vm" ),
	sockFd_( -1 ),
	changeThreshold_( 1 ),
	vScale_( 1.0 ),
	syncMode_( false ),
	bgcolorRed_( 0.0 ),
	bgcolorGreen_( 0.0 ),
	bgcolorBlue_( 0.0 ),
	highValue_( 0.05 ),
	lowValue_( -0.1 ),
	testTicker_( 0 )
{
#ifdef WIN32
	if ( initWinsock() < 0 )
	{
		std::cerr << "Winsock could not be initialized. Cannot connect to client." << std::endl;
	}
#endif
}

GLcell::~GLcell()
{
	disconnect();

#ifdef WIN32
	WSACleanup();
#endif
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void GLcell::setPath( const Conn* c, string strPath )
{
	static_cast< GLcell * >( c->data() )->innerSetPath( strPath );
}

void GLcell::innerSetPath( const string& strPath )
{
	strPath_ = strPath;
}

string GLcell::getPath( Eref e )
{
	return static_cast< const GLcell* >( e.data() )->strPath_;
}

void GLcell::setClientHost( const Conn* c, string strClientHost )
{
	static_cast< GLcell * >( c->data() )->innerSetClientHost( strClientHost );
}

void GLcell::innerSetClientHost( const string& strClientHost )
{
	strClientHost_ = strClientHost;
}

string GLcell::getClientHost( Eref e )
{
	return static_cast< const GLcell* >( e.data() )->strClientHost_;
}

void GLcell::setClientPort( const Conn* c, string strClientPort )
{
	static_cast< GLcell * >( c->data() )->innerSetClientPort( strClientPort );
}

void GLcell::innerSetClientPort( const string& strClientPort )
{
	strClientPort_ = strClientPort;
}

string GLcell::getClientPort( Eref e )
{
	return static_cast< const GLcell* >( e.data() )->strClientPort_;
}

void GLcell::setAttributeName( const Conn* c, string strAttributeName )
{
	static_cast< GLcell * >( c->data() )->innerSetAttributeName( strAttributeName );
}

void GLcell::innerSetAttributeName( const string& strAttributeName )
{
	strAttributeName_ = strAttributeName;
}

string GLcell::getAttributeName( Eref e )
{
	return static_cast< const GLcell* >( e.data() )->strAttributeName_;
}

void GLcell::setChangeThreshold( const Conn* c, double changeThreshold )
{
	static_cast< GLcell * >( c->data() )->innerSetChangeThreshold( changeThreshold );
}

void GLcell::innerSetChangeThreshold( const double changeThreshold )
{
	changeThreshold_ = changeThreshold;
}

double GLcell::getChangeThreshold( Eref e )
{
	return static_cast< const GLcell* >( e.data() )->changeThreshold_;
}

void GLcell::setVScale( const Conn* c, double vScale )
{
	static_cast< GLcell * >( c->data() )->innerSetVScale( vScale );
}

void GLcell::innerSetVScale( const double vScale )
{
	vScale_ = vScale;
}

double GLcell::getVScale( Eref e )
{
	return static_cast< const GLcell* >( e.data() )->vScale_;
}

void GLcell::setSyncMode( const Conn* c, string syncMode )
{
	if ( syncMode == string( "on" ) )
		static_cast< GLcell * >( c->data() )->innerSetSyncMode( true );
	else if ( syncMode == string( "off" ) )
		static_cast< GLcell * >( c->data() )->innerSetSyncMode( false );
	else
		std::cerr << "GLcell error: annot set sync mode; argument must be either 'on' or 'off'." << std::endl;
}

void GLcell::innerSetSyncMode( const bool syncMode )
{
	syncMode_ = syncMode;
}

string GLcell::getSyncMode( Eref e )
{
	bool currentSyncMode = static_cast< const GLcell* >( e.data() )->syncMode_;

	if ( currentSyncMode )
		return string( "on" );
	else
		return string( "off" );
}

void GLcell::setBgColor( const Conn* c, string strBgColor )
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
			static_cast< GLcell * >( c->data() )->innerSetBgColor( red, green, blue );
		}
	}
	else
	{
		error = true;
	}	

	if ( error ) // report error; default is (0,0,0) (black)
	{
		std::cerr << "GLcell error: the field 'bgcolor' is not in the expected format, defaulting to black" << std::endl;
	}

}

void GLcell::innerSetBgColor( const double red, const double green, const double blue )
{
	bgcolorRed_ = red;
	bgcolorGreen_ = green;
	bgcolorBlue_ = blue;
}

string GLcell::getBgColor( Eref e )
{
	double red = static_cast< const GLcell* >( e.data() )->bgcolorRed_;
	double green = static_cast< const GLcell* >( e.data() )->bgcolorGreen_;
	double blue = static_cast< const GLcell* >( e.data() )->bgcolorBlue_;
	
	int bgcolor = (red * 255.) * 1000000 + (green * 255.) * 1000 + (blue * 255.);

	std::string s;
	std::stringstream out;
	out << bgcolor;
	return out.str();
}

void GLcell::setHighValue( const Conn* c, double highValue )
{
	static_cast< GLcell * >( c->data() )->innerSetHighValue( highValue );
}

void GLcell::innerSetHighValue( const double highValue )
{
	if ( highValue_ <= lowValue_ )
	{
		std::cerr << "GLcell warning: highvalue must be set to be greather than 'lowvalue'." << std::endl;
	}

	highValue_ = highValue;
}

double GLcell::getHighValue( Eref e )
{
	return static_cast< const GLcell* >( e.data() )->highValue_;
}

void GLcell::setLowValue( const Conn* c, double lowValue )
{	
	static_cast< GLcell * >( c->data() )->innerSetLowValue( lowValue );
}

void GLcell::innerSetLowValue( const double lowValue )
{
	if ( highValue_ <= lowValue_ )
	{
		std::cerr << "GLcell warning: highvalue must be set to be greather than 'lowvalue'." << std::endl;
	}

	lowValue_ = lowValue;
}

double GLcell::getLowValue( Eref e )
{
	return static_cast< const GLcell* >( e.data() )->lowValue_;
}	

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void GLcell::reinitFunc( const Conn* c, ProcInfo info )
{
	static_cast< GLcell * >( c->data() )->reinitFuncLocal( c );
}

void GLcell::reinitFuncLocal( const Conn* c )
{
	GLcellResetData geometryData;
	double diameter, length, x0, y0, z0, x, y, z;

	vecParticleData_.clear();
	vecSmoldynShapeData_.clear();

	geometryData.strPathName = " ";
	geometryData.bgcolorRed = bgcolorRed_;
	geometryData.bgcolorGreen = bgcolorGreen_;
	geometryData.bgcolorBlue = bgcolorBlue_;

	/// Reload model geometry.
	// strPath_ should have been set.
	if ( ! strPath_.empty() )
	{
		// vecRenderList_ holds the flattened tree of elements to render.
		vecRenderList_.clear();

		geometryData.strPathName = strPath_;
		geometryData.vScale = vScale_;
		geometryData.vecRenderListCompartmentData.clear();

		// Start populating vecRenderList_ with the node in strPath_ 
		// and its children, recursively.
		add2RenderList( Shell::path2eid( strPath_, "/", 1 ) );

		for ( unsigned int i = 0; i < vecRenderList_.size(); ++i )
		{

			if ( ( vecRenderList_[i]()->cinfo()->isA( Cinfo::find( "Compartment" ) ) 
			       || vecRenderList_[i]()->cinfo()->isA( Cinfo::find( "SymCompartment" ) ) )
				&& get< double >( vecRenderList_[i].eref(), "diameter", diameter )
				&& get< double >( vecRenderList_[i].eref(), "length", length )
				&& get< double >( vecRenderList_[i].eref(), "x0", x0 )
				&& get< double >( vecRenderList_[i].eref(), "y0", y0 )
				&& get< double >( vecRenderList_[i].eref(), "z0", z0 )
				&& get< double >( vecRenderList_[i].eref(), "x", x )
				&& get< double >( vecRenderList_[i].eref(), "y", y )
				&& get< double >( vecRenderList_[i].eref(), "z", z ) )
			{
				GLcellProcData compartmentData;
				
				compartmentData.id = vecRenderList_[i].id();
				compartmentData.strName = vecRenderList_[i].eref().name();
				compartmentData.strPathName = vecRenderList_[i].path();

				std::vector< unsigned int > vecNeighbourIds;			     
				findNeighbours( vecRenderList_[i], vecNeighbourIds );
				compartmentData.vecNeighbourIds = vecNeighbourIds;
				
				compartmentData.diameter = diameter;
				compartmentData.length = length;
				compartmentData.x0 = x0;
				compartmentData.y0 = y0;
				compartmentData.z0 = z0;
				compartmentData.x = x;
				compartmentData.y = y;
				compartmentData.z = z;
				
				geometryData.vecRenderListCompartmentData.push_back( compartmentData );
			}
		}
	}

	if ( strClientPort_.empty() )
		std::cerr << "GLcell error: Client port not specified." << std::endl;
	else if ( strClientHost_.empty() )
		std::cerr << "GLcell error: Client hostname not specified." << std::endl;
	else
		transmit( geometryData, RESET );


	// testInsertVecSmoldynShapeData();
}


void GLcell::processFunc( const Conn* c, ProcInfo info )
{
	static_cast< GLcell * >( c->data() )->processFuncLocal( c->target(), info );
}

void GLcell::processFuncLocal( Eref e, ProcInfo info )
{
	unsigned int id;
	double attrValue;
	std::map< unsigned int, double> mapColors;
       
	if ( ! vecRenderList_.empty() )
	{
		renderMapAttrsTransmitted_.clear();
		
		for ( unsigned int i = 0; i < vecRenderList_.size(); ++i )
		{
			if ( ( vecRenderList_[i]()->cinfo()->isA( Cinfo::find( "Compartment" ) ) 
			       || vecRenderList_[i]()->cinfo()->isA( Cinfo::find( "SymCompartment" ) ) )
			     && get< double >( vecRenderList_[i].eref(), strAttributeName_.c_str(), attrValue ) )
			{
				id = vecRenderList_[i].id();

				if ( ( renderMapAttrsLastTransmitted_.empty() )  ||
				     // on the first PROCESS after a RESET

				     syncMode_ ||
				     // or we're in sync mode

				     ( fabs( attrValue - renderMapAttrsLastTransmitted_[id] )
				       > changeThreshold_/100 * ( highValue_ - lowValue_ ) ) )
				     // or the current change differs significantly from
				     // that last transmitted for this compartment
				{
					renderMapAttrsLastTransmitted_[id] = renderMapAttrsTransmitted_[id] = attrValue;
				}
			}
		}

		mapColors = mapAttrs2Colors( renderMapAttrsTransmitted_ );

		if ( syncMode_ )
		{
			transmit( mapColors, PROCESS_COLORS_SYNC );
			receiveAck();
			// The client will wait for the display to be updated before
			// sending this ack in response to a PROCESS_COLORS_SYNC message.
		}
		else
		{
			transmit( mapColors, PROCESS_COLORS );
			receiveAck();
		}
	}

	// testInsertVecSmoldynShapeData();
	// testInsertVecParticleData();

	if ( vecSmoldynShapeData_.size() > 0 )
	{
		transmit( vecSmoldynShapeData_, PROCESS_SMOLDYN_SHAPES );
		receiveAck();
		
		vecSmoldynShapeData_.clear();
	}

	if ( vecParticleData_.size() > 0 )
	{
		if ( syncMode_ )
		{
			transmit( vecParticleData_, PROCESS_PARTICLES_SYNC );
			receiveAck();
			// The client will wait for the display to be updated before
			// sending this ack in response to a PROCESS_PARTICLES_SYNC message.
		}
		else
		{
			transmit( vecParticleData_, PROCESS_PARTICLES );
			receiveAck();
		}

		vecParticleData_.clear();
	}
}

void GLcell::setParticleData( const Conn* c, vector< ParticleData > vecParticleData )
{
	static_cast< GLcell * >( c->data() )->innerSetParticleData( vecParticleData );
}

void GLcell::innerSetParticleData( const vector< ParticleData > vecParticleData )
{
	if ( vecParticleData_.size() > 0 )
	{
		vecParticleData_.clear();
	}

	vecParticleData_ = vecParticleData;
}

void GLcell::setSmoldynShapeData( const Conn* c, vector< SmoldynShapeData > vecSmoldynShapeData )
{
	static_cast< GLcell * >( c->data() )->innerSetSmoldynShapeData( vecSmoldynShapeData );
}

void GLcell::innerSetSmoldynShapeData( const vector< SmoldynShapeData > vecSmoldynShapeData )
{
	if ( vecSmoldynShapeData_.size() > 0 )
	{
		vecSmoldynShapeData_.clear();
	}

	vecSmoldynShapeData_ = vecSmoldynShapeData;
}

///////////////////////////////////////////////////
// private function definitions
///////////////////////////////////////////////////

std::map< unsigned int, double> GLcell::mapAttrs2Colors( std::map< unsigned int, double > renderMapAttrs )
{
	std::map< unsigned int, double> mapColors;
	std::map< unsigned int, double>::iterator renderMapAttrsIterator;

	for ( renderMapAttrsIterator = renderMapAttrs.begin();
	      renderMapAttrsIterator != renderMapAttrs.end();
	      renderMapAttrsIterator++ )
	{
		unsigned int id = renderMapAttrsIterator->first;
		double attr = renderMapAttrsIterator->second;

		if ( attr > highValue_ )
		{
			mapColors[id] = 1;
		}
		else if ( attr < lowValue_ )
		{
			mapColors[id] = 0;
		}
		else
		{
			mapColors[id] = ( attr - lowValue_ ) / ( highValue_ - lowValue_ );
		}
	}
	
	return mapColors;
}

void GLcell::findNeighbours( Id id, vector< unsigned int >& vecResult )
{
	// result is appended to vecResult
	
	findNeighboursOfType( id, "axial", "Compartment", vecResult );
	findNeighboursOfType( id, "raxial", "Compartment", vecResult );
	findNeighboursOfType( id, "axial", "SymCompartment", vecResult );
	findNeighboursOfType( id, "raxial", "SymCompartment", vecResult );
	findNeighboursOfType( id, "raxial1", "SymCompartment", vecResult );
	findNeighboursOfType( id, "raxial2", "SymCompartment", vecResult );
}

void GLcell::findNeighboursOfType( Id id, const string& messageType, const string& targetType, std::vector< unsigned int >& vecResult )
{
	// This function is derived largely from BioScan::targets()

	// result is appended to vecResult

	Id found;

	if ( messageType == "" )
	{
		std::cerr << "findNeighboursOfType() called with blank messageType" << std::endl;
		return;
	}

	Conn* i = id()->targets( messageType, 0 );
	for ( ; i->good(); i->increment() )
	{
		found = i->target()->id();

		if ( targetType != "" &&
		     found()->cinfo()->isA( Cinfo::find( targetType ) ) )
			vecResult.push_back( found.id() );
	}
	delete i;
}

void GLcell::add2RenderList( Id id )
{
	vector< Id > children;
	Id found;

	// Save this node on vecRenderList_, the flattened tree of elements to render.
	vecRenderList_.push_back( id );
	
	// Determine this node's (immediate) children by tracing outgoing "childSrc" messages.
	Conn* i = id()->targets( "childSrc", 0 );
	for ( ; i->good(); i->increment() )
	{
		found = i->target()->id();
		// Although we only display Compartment and SymCompartment types later, we don't
		// filter by those types here because our targets might be found to be children
		// of non-Compartments and non-SymCompartments.

		children.push_back( found );
	}
	delete i;

	// If there are any children, call add2RenderList on each of them.
	for ( unsigned int j = 0; j < children.size(); ++j )
	{
		add2RenderList( children[j] );
	}
}

///////////////////////////////////////////////////
// networking helper function definitions
///////////////////////////////////////////////////

void* GLcell::getInAddress( struct sockaddr *sa )
{
	if ( sa->sa_family == AF_INET ) {
		return &( ( ( struct sockaddr_in* )sa )->sin_addr );
	}

	return &( ( ( struct sockaddr_in6* )sa )->sin6_addr );
}

int GLcell::getSocket( const char* hostname, const char* service )
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
		std::cerr << "GLcell error: getaddrinfo: " << gai_strerror( rv ) << std::endl;
		return -1;
	}
	// loop through all the results and connect to the first we can
	for ( p = servinfo; p != NULL; p = p->ai_next ) {
		if ( ( sockFd_ = socket( p->ai_family, p->ai_socktype,
				     p->ai_protocol ) ) == socket_error ) {
		    //std::cerr << "GLcell error: socket" << std::endl;
			continue;
		}
		
		if ( connect( sockFd_, p->ai_addr, p->ai_addrlen ) == connect_error ) {
#ifdef WIN32
			closesocket( sockFd_ );
#else
			close( sockFd_ );
#endif
			//std::cerr << "GLcell error: connect" << std::endl;
			continue;
		}
		
		break;
	}

	if ( p == NULL ) {
		std::cerr << "GLcell error: failed to connect" << std::endl;
		return -1;
	}
	
	/*inet_ntop( p->ai_family, getInAddress( ( struct sockaddr * )p->ai_addr ),
		   s, sizeof s );
	 std::cout << "Connecting to " << s << std::endl; */
	
	freeaddrinfo( servinfo );
	
	isConnectionUp_ = true;
	return sockFd_;
}

int GLcell::sendAll( int socket, char* buf, unsigned int* len )
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
			std::cerr << "GLcell error: send error; errno: " << errno << " " << strerror( errno ) << std::endl;    
			break;
		}
		total += n;
		bytesleft -= n;
	}

	*len = total; // return number actually sent here

	return n == -1 ? -1 : 0; // return -1 on failure, 0 on success
}

int GLcell::recvAll( int socket, char *buf, unsigned int *len )
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
			std::cerr << "GLcell error: recv error; errno: " << errno << " " << strerror( errno ) << std::endl;
			break;
		}
		total += n;
		bytesleft -= n;
	}
	
	*len = total; /// return number actually received here
	
	return n == -1 ? -1 : 0; // return -1 on failure, 0 on success
}

int GLcell::receiveAck()
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
		std::cerr << "GLcell error: could not receive Ack header!" << std::endl;
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
		std::cerr << "GLcell error: could not receive Ack!" << std::endl;
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

void GLcell::handlePick( unsigned int idPicked )
{
	std::cout << "Compartment with id " << idPicked << " was picked!" << std::endl;
}

template< class T >
void GLcell::transmit( T& data, MsgType messageType)
{
	if ( strClientHost_.empty() || strClientPort_.empty() ) // these should have been set.
		return;

	if ( ! isConnectionUp_ )
	{
		sockFd_ = getSocket( strClientHost_.c_str(), strClientPort_.c_str() );
		if ( sockFd_ == -1 ) 
		{
			std::cerr << "GLcell error: Couldn't connect to client!" << std::endl;
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
			std::cerr << "GLcell error: couldn't transmit header to client!" << std::endl;

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
				std::cerr << "GLcell error: couldn't transmit data to client!" << std::endl;	
			}
			free( archiveData );
		}
		free( headerData );
	}
}

void GLcell::disconnect()
{
	if ( ! isConnectionUp_ )
	{
		sockFd_ = getSocket( strClientHost_.c_str(), strClientPort_.c_str() );
		if ( sockFd_ == -1 ) 
		{
			std::cerr << "GLcell error: couldn't connect to client!" << std::endl;
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

void GLcell::testParticle1( )
{
	ParticleData p;
	p.color[0] = 1.0;
	p.color[1] = 0.0;
	p.color[2] = 0.0;
	p.diameter = 0;
	for ( unsigned int i = 0; i < 100; i++ )
	{
		p.vecCoords.push_back( 1e-6 * (i*10 + 100*sin(testTicker_)) );
		p.vecCoords.push_back( 1e-6 * (i*10 + 100*cos(testTicker_++)) );
		p.vecCoords.push_back( 1e-6 * (i*10 + 10) );
	}

	ParticleData p1;
	p1.color[0] = 0.0;
	p1.color[1] = 0.0;
	p1.color[2] = 1.0;
	p1.diameter = 1e-6 * 10;
#ifdef WIN32
	unsigned int j = 0;
#else
	unsigned int j = (unsigned int)(random() % 10);
#endif
	for ( unsigned int i = 0; i < 10+j; i++ )
	{
		p1.vecCoords.push_back( 1e-6 * (i*5 + 50*cos(testTicker_)) );
		p1.vecCoords.push_back( 1e-6 * (i*5 + 50*sin(testTicker_++)) );
		p1.vecCoords.push_back( 1e-6 * (i*5 + 5) );
	}

	vecParticleData_.push_back( p );
	vecParticleData_.push_back( p1 );
}

void GLcell::testParticle2( )
{
	ParticleData p;
	p.color[0] = 1.0;
	p.color[1] = 0.0;
	p.color[2] = 0.0;
	p.diameter = 1.0;
	for ( unsigned int i = 0; i < 100; i++ )
	{
		testTicker_++;
		p.vecCoords.push_back( 0.0 );
		p.vecCoords.push_back( 0.0 );
		p.vecCoords.push_back( 10.0 );
	}

	vecParticleData_.push_back( p );
}

void GLcell::testInsertVecParticleData( )
{
	testParticle1( );
	// testParticle2( );
}

/*
 * Demonstrates different Smoldyn shapes.
 */
void GLcell::testShape1( )
{
	SmoldynShapeData s1, s2, s3, s4, s5, s6;

	s1.color[0] = 0.6; s1.color[1] = 0.7; s1.color[2] = 0.8; s1.color[3] = 1.0;
	s2.color[0] = 0.6; s2.color[1] = 0.7; s2.color[2] = 0.8; s2.color[3] = 1.0;
	s3.color[0] = 0.6; s3.color[1] = 0.7; s3.color[2] = 0.8; s3.color[3] = 1.0;
	s4.color[0] = 0.6; s4.color[1] = 0.7; s4.color[2] = 0.8; s4.color[3] = 0.1;
	s5.color[0] = 0.6; s5.color[1] = 0.7; s5.color[2] = 0.8; s5.color[3] = 0.1;
	s6.color[0] = 0.6; s6.color[1] = 0.7; s6.color[2] = 0.8; s6.color[3] = 0.1;

	GLCompartmentCylinderData d1;
	d1.endPoint1[0] = 2; d1.endPoint1[1] = 2; d1.endPoint1[2] = 2;
	d1.endPoint2[0] = 6; d1.endPoint2[1] = 2; d1.endPoint2[2] = 2;
	d1.radius = 0.5;
	s1.data = d1;
	s1.name = "Cylinder";
	
	GLCompartmentSphereData d2;
	d2.centre[0] = 2; d2.centre[1] = 4; d2.centre[2] = 2;
	d2.radius = 1;
	s2.data = d2;
	s2.name = "Sphere";

	GLCompartmentDiskData d3;
	d3.centre[0] = 2; d3.centre[1] = 6; d3.centre[2] = 2;
	d3.orientation[0] = 0; d3.orientation[1] = 0; d3.orientation[2] = 1;
	d3.radius = 1;
	s3.data = d3;
	// s3.name = "Disk"; // Note that no name is provided

	GLCompartmentHemiData d4;
	d4.centre[0] = 2; d4.centre[1] = 8; d4.centre[2] = 2;
	d4.orientation[0] = 0; d4.orientation[1] = 0; d4.orientation[2] = 1;
	d4.radius = 1;
	s4.data = d4;
	// s4.name = "Hemi";
	
	GLCompartmentTriData d5;
	d5.corner1[0] = 2; d5.corner1[1] = 10; d5.corner1[2] = 2;
	d5.corner2[0] = 0; d5.corner2[1] = 12; d5.corner2[2] = 2;
	d5.corner3[0] = 4; d5.corner3[1] = 14; d5.corner3[2] = 2;
	s5.data = d5;
	s5.name = "Tri";

	GLCompartmentRectData d6;
	d6.corner1[0] = 2; d6.corner1[1] = 12; d6.corner1[2] = 2;
	d6.corner2[0] = 2; d6.corner2[1] = 14; d6.corner2[2] = 2;
	d6.corner3[0] = 4; d6.corner3[1] = 14; d6.corner3[2] = 2;
	d6.corner4[0] = 4; d6.corner4[1] = 12; d6.corner4[2] = 2;
	s6.data = d6;
	s6.name = "Rect";

	vecSmoldynShapeData_.push_back( s1 );
	vecSmoldynShapeData_.push_back( s2 );
	vecSmoldynShapeData_.push_back( s3 );
	vecSmoldynShapeData_.push_back( s4 );
	vecSmoldynShapeData_.push_back( s5 );
	vecSmoldynShapeData_.push_back( s6 );
}

/*
 * Draws a capsule using hemispheres and a cylinder.
 */
void GLcell::testShape2( )
{
	SmoldynShapeData s1, s2, s3;
	++testTicker_;

	s1.color[0] = 0.6; s1.color[1] = 0.7; s1.color[2] = 0.8; s1.color[3] = 0.3;
	s2.color[0] = 0.6; s2.color[1] = 0.7; s2.color[2] = 0.8; s2.color[3] = 0.3;
	s3.color[0] = 0.6; s3.color[1] = 0.7; s3.color[2] = 0.8; s3.color[3] = 0.3;
	
	GLCompartmentCylinderData d1;
	d1.endPoint1[0] = 0; d1.endPoint1[1] = 0; d1.endPoint1[2] = 0;
	d1.endPoint2[0] = testTicker_+10; d1.endPoint2[1] = 0; d1.endPoint2[2] = 0;
	d1.radius = 4;
	s1.data = d1;
	s1.name = "Cylinder";
	
	GLCompartmentHemiData d2;
	d2.centre[0] = 0; d2.centre[1] = 0; d2.centre[2] = 0;
	d2.orientation[0] = -1; d2.orientation[1] = 0; d2.orientation[2] = 0;
	d2.radius = 4;
	s2.data = d2;
	// s2.name = "Hemi";
	
	GLCompartmentHemiData d3;
	d3.centre[0] = testTicker_+10; d3.centre[1] = 0; d3.centre[2] = 0;
	d3.orientation[0] = 1; d3.orientation[1] = 0; d3.orientation[2] = 0;
	d3.radius = 4;
	s3.data = d3;
	// s3.name = "Hemi";
	
	vecSmoldynShapeData_.push_back( s1 );
	vecSmoldynShapeData_.push_back( s2 );
	vecSmoldynShapeData_.push_back( s3 );
}

void GLcell::testInsertVecSmoldynShapeData( )
{
	// testShape1( );
	testShape2( );
}

#ifdef WIN32
int GLcell::initWinsock( void )
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

