/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

////////////////////////////////////////////////////////////////////////////////
// The socket code is mostly taken from Beej's Guide to Network Programming   //
// at http://beej.us/guide/bgnet/. The original code is in the public domain. //
////////////////////////////////////////////////////////////////////////////////

#ifdef WIN32
#include <Winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#include "win32getopt.h"
#define getopt wgetopt
#define optarg woptarg
#define optopt woptopt
#define errno WSAGetLastError()
#else
#include <errno.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#endif

#include <ctype.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/filesystem.hpp>

#include <osg/Vec3d>
#include <osg/ref_ptr>
#include <osg/Notify>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osg/Quat>
#include <osg/Projection>
#include <osg/MatrixTransform>
#include <osg/Transform>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osg/CullSettings>
#include <osg/Point>

#include "GLcellProcData.h"
#include "AckPickData.h"
#include "GLcellResetData.h"
#include "GLviewResetData.h"
#include "GLshapeData.h"
#include "GLviewShape.h"

#include "TextBox.h"
#include "GLCompartmentCylinder.h"
#include "GLCompartmentSphere.h"
#include "GLCompartmentTri.h"
#include "GLCompartmentRect.h"
#include "GLCompartmentDisk.h"
#include "GLCompartmentHemi.h"
#include "GLCompartment.h"

#include "GLclient.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool KeystrokeHandler::handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa, osg::Object*, osg::NodeVisitor* )
{
	osgViewer::Viewer *viewer = dynamic_cast< osgViewer::Viewer* >( &aa );
	if ( ! viewer )
		return false;

	switch( ea.getEventType() )
	{
	case ( osgGA::GUIEventAdapter::PUSH ) :
	case ( osgGA::GUIEventAdapter::MOVE ) :
		x_ = ea.getX(); // record mouse location
		y_ = ea.getY();
		return false;
	case ( osgGA::GUIEventAdapter::RELEASE ) :
		// if the mouse hasn't moved since the last PUSH or MOVE,
		// perform a pick. otherwise let TrackBallManipulator handle this.
		if ( x_ == ea.getX() && y_ == ea.getY() )
		{
			if ( pick( ea.getXnormalized(), ea.getYnormalized(), viewer ) )
				return true;
		}
		return false;
	case ( osgGA::GUIEventAdapter::KEYDOWN ) :
		if ( ea.getKey() == 'c' || ea.getKey() == 'C' )
		{
			screenCaptureHandler_->captureNextFrame( *viewer_ );

			std::cout << "Saving screenshot. " << std::endl;
			return true;
		}
		else if ( ea.getKey() == 'm' || ea.getKey() == 'M' )
		{
			if ( isSavingMovie_ == true )
			{
				isSavingMovie_ = false;
				std::cout << "Stopping movie recording. " << std::endl;
			}
			else
			{			
				isSavingMovie_ = true;
				std::cout << "Starting movie recording... " << std::endl;
			}			
			return true;
		}
		else if ( ea.getKey() == 'p' || ea.getKey() == 'P' )
		{
			switchProjection( viewer );
		}
		else
			return false;
	default:
		break;
	}
	return false;
}

void KeystrokeHandler::switchProjection( osgViewer::Viewer* viewer )
{
	static double oleft, oright, obottom, otop, oznear, ozfar;
	static double fleft, fright, fbottom, ftop, fznear, fzfar;

	if ( isCurrentProjectionOrtho_ )
	{
		viewer->getCamera()->getProjectionMatrixAsOrtho( oleft, oright,
								 obottom, otop,
								 oznear, ozfar );
		viewer->getCamera()->setProjectionMatrixAsFrustum( fleft, fright,
								   fbottom, ftop,
								   fznear, fzfar );

		isCurrentProjectionOrtho_ = false;
	}
	else // always true on the first call to this function
	{
		viewer->getCamera()->getProjectionMatrixAsFrustum( fleft, fright,
								   fbottom, ftop,
								   fznear, fzfar );

		oleft = fleft;
		oright = fright;
		obottom = fbottom;
		otop = ftop;
		oznear = fznear;
		ozfar = fzfar;
		
		viewer->getCamera()->setProjectionMatrixAsOrtho( oleft, oright,
								 obottom, otop,
								 oznear, ozfar );

		isCurrentProjectionOrtho_ = true;
	}
}

bool KeystrokeHandler::pick( const double x, const double y, osgViewer::Viewer* viewer )
{
	if ( ! viewer->getSceneData() )
		return false;

	if ( textParentTop_ != NULL )
	{
		textParentTop_->setText( "" );
	}

	double w = .05;
	double h = .05;

	osgUtil::PolytopeIntersector* picker = new osgUtil::PolytopeIntersector( osgUtil::Intersector::PROJECTION, 
										 x-w, y-h, x+w, y+h );
	osgUtil::IntersectionVisitor iv( picker );
	viewer->getCamera()->accept( iv );

	if ( picker->containsIntersections() )
	{
		const osg::NodePath& nodePath = picker->getFirstIntersection().nodePath;
		
		osg::Geode* geode = dynamic_cast< osg::Geode* >( nodePath[nodePath.size()-1] );

		if ( mapGeode2NameId_.count( geode ) == 1 )
		{
			std::stringstream pickedTextStream;
			pickedTextStream << *(mapGeode2NameId_[geode]->second) << " ("
					 << mapGeode2NameId_[geode]->first << ") was picked.";
			std::cout << pickedTextStream.str() << std::endl;
			textParentTop_->setText( pickedTextStream.str() );

			{
				boost::mutex::scoped_lock lock( mutexPickingDataUpdated_ );
				pickedId_ = mapGeode2NameId_[geode]->first;
			}
			isPickingDataUpdated_ = true;
		}
		return true;	
	}
	
	return false;
}

// get sockaddr, IPv4 or IPv6:
void* getInAddr( struct sockaddr* sa )
{
	if ( sa->sa_family == AF_INET ) {
		return &( ( ( struct sockaddr_in* )sa )->sin_addr );
	}

	return &( ( ( struct sockaddr_in6* )sa )->sin6_addr );
}

int sendAll( int socket, char* buf, unsigned int* len )
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
			std::cerr << "send error; errno: " << errno << " " << strerror( errno ) << std::endl;    
			break;
		}
		total += n;
		bytesleft -= n;
	}

	*len = total; // return number actually sent here

	return n == -1 ? -1 : 0; // return -1 on failure, 0 on success
}

int recvAll( int socket, char *buf, unsigned int *len )
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
			std::cerr << "recv error; errno: " << errno << " " << strerror( errno ) << std::endl;
			break;
		}
		total += n;
		bytesleft -= n;
	}
	
	*len = total; /// return number actually received here
	
	return n == -1 ? -1 : 0; // return -1 on failure, 0 on success
}

void networkLoop( void )
{
	int newFd;

	while ( true )
	{
		if ( (newFd = acceptNewConnection( port_ )) != -1 )
		{
			receiveData( newFd );
		}		
		else
		{
			std::cerr << "Error in network loop... exiting." << std::endl;
			exit(1);
		}
	}
}

#ifdef WIN32
int initWinsock( void )
{
	WORD wVersionRequested;
	WSADATA wsaData;
	int err;

	wVersionRequested = MAKEWORD(2, 2);

	err = WSAStartup(wVersionRequested, &wsaData);
	if (err != 0) 
	{
		std::cerr << "WSAStartup failed with error: . Exiting." << err << std::endl;
		return -1;
	}

	if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
	{
		std::cerr << "Could not find a usable version of Winsock.dll. Exiting." << std::endl;
		WSACleanup();
		return -1;
	}

	return 0;
}
#endif

int acceptNewConnection( char * port )
{
	int sockFd, newFd;  // listen on sock_fd, new connection on new_fd
	struct addrinfo hints, *servinfo, *p;
	struct sockaddr_storage theirAddr; // connector's address information
	socklen_t sinSize;
	int yes=1;
	char s[INET6_ADDRSTRLEN];
	int rv;
	
#ifdef WIN32
	unsigned int socket_error = INVALID_SOCKET;
	unsigned int setsockopt_error = SOCKET_ERROR;
	unsigned int bind_error = SOCKET_ERROR;
	unsigned int listen_error = SOCKET_ERROR;
	unsigned int accept_error = INVALID_SOCKET;

	if ( initWinsock() < 0 )
	{
			exit(1);
	}
#else
	int socket_error = -1;
	int setsockopt_error = -1;
	int bind_error = -1;
	int listen_error = -1;
	int accept_error = -1;
#endif

	memset( &hints, 0, sizeof( hints ) );
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE; // use my IP
	
	if ( ( rv = getaddrinfo( NULL, port, &hints, &servinfo ) ) != 0 ) {
		std::cerr << "getaddrinfo: " << gai_strerror( rv ) << std::endl;
		return -1;
	}
	
	// loop through all the results and bind to the first we can
	for( p = servinfo; p != NULL; p = p->ai_next ) {
		if ( ( sockFd = socket( p->ai_family, p->ai_socktype, p->ai_protocol ) ) == socket_error ) {
			std::cerr <<  "GLclient error: socket" << std::endl;
			continue;
		}
		
#ifdef WIN32
		if ( setsockopt( sockFd, SOL_SOCKET, SO_REUSEADDR, (const char*) &yes, sizeof( int ) ) == setsockopt_error ) {
#else
		if ( setsockopt( sockFd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof( int ) ) == setsockopt_error ) {
#endif
			std::cerr << "GLclient error: setsockopt" << std::endl;
			return -1;
		}
		
		if ( bind( sockFd, p->ai_addr, p->ai_addrlen ) == bind_error ) {
#ifdef WIN32
			closesocket( sockFd );
#else
			close( sockFd );
#endif
			std::cerr << "GLclient error: bind" << std::endl;
			continue;
		}
		
		break;
	}
	
	if ( p == NULL )  {
		std::cerr << "GLclient error: failed to bind" << std::endl;
		return -1;
	}
  
	freeaddrinfo( servinfo ); // all done with this structure
  
	if ( listen( sockFd, BACKLOG ) == listen_error ) {
		std::cerr << "GLclient error: listen" << std::endl;
		return -1;
	}

	std::cout << "GLclient: waiting for connections..." << std::endl;

	sinSize = sizeof( theirAddr );
	newFd = accept( sockFd, ( struct sockaddr * ) &theirAddr, &sinSize );
	if ( newFd == accept_error ) {
		std::cerr << "GLclient error: accept" << std::endl;
		return -1;
	}
		
	/*inet_ntop( theirAddr.ss_family, getInAddr( ( struct sockaddr * ) &theirAddr ), s, sizeof( s ) );

	std::cout << "GLclient: connected to " << s << std::endl;*/

#ifdef WIN32
	closesocket( sockFd );
#else
	close( sockFd );
#endif
	return newFd;
}

void receiveData( int newFd )
{
	unsigned int numBytes, inboundDataSize;
	char header[MSGSIZE_HEADERLENGTH + MSGTYPE_HEADERLENGTH + 1];
	int messageType;
	char *buf;

	while ( true )
	{
		numBytes = MSGSIZE_HEADERLENGTH + MSGTYPE_HEADERLENGTH + 1;
		if ( recvAll( newFd, header, &numBytes ) == -1 ||
		     numBytes < MSGSIZE_HEADERLENGTH + MSGTYPE_HEADERLENGTH + 1 ) 
		{
			std::cerr << "GLclient error:  could not receive message header!" << std::endl;
			break;
		}
		else
		{
			std::istringstream msgsizeHeaderstream( std::string( header, 
									     MSGSIZE_HEADERLENGTH ) );
			msgsizeHeaderstream >> std::hex >> inboundDataSize;
			
			std::istringstream msgtypeHeaderstream( std::string( header,
									     MSGSIZE_HEADERLENGTH,
									     MSGTYPE_HEADERLENGTH ) );
			msgtypeHeaderstream >> messageType;

			if ( messageType == DISCONNECT )
			{
				std::cout << "GLclient: MOOSE element disconnected normally." << std::endl;
				break;
			}
		}
		
		numBytes = inboundDataSize + 1;
		buf = ( char * ) malloc( numBytes * sizeof( char ) );
		
		if ( recvAll( newFd, buf, &numBytes ) == -1 ||
		     numBytes < inboundDataSize + 1 )
		{
			std::cerr << "GLclient error: incomplete data received!" << std::endl;
			std::cerr << "numBytes: " << numBytes << " inboundDataSize: " << inboundDataSize << std::endl;
			break;
		}
		else
		{
			std::istringstream archive_stream_i( std::string( buf, inboundDataSize ) );
			// starting new scope so that the archive's stream's destructor is called after the archive's
			{
				boost::archive::text_iarchive archive_i( archive_stream_i );
			
				if ( mode_ == GLCELL )
				{
					if ( messageType == RESET) 
					{
						GLcellResetData geometryData;
						archive_i >> geometryData;

						initializeRoot( geometryData.strPathName );
						updateGeometryGLcell( geometryData );

						isGeometryDirty_ = true;
					}
					else if ( messageType == PROCESS_SMOLDYN_SHAPES )
					{
						std::vector< SmoldynShapeData > vecSmoldynShapeData;
						archive_i >> vecSmoldynShapeData;
						
						initializeRoot( std::string( "Smoldyn model" ) );
						updateSmoldynGeometry( vecSmoldynShapeData );

						isSmoldynShapesDirty_ = true;

						sendAck( newFd, messageType );
					}
					else if ( messageType == PROCESS_COLORS || messageType == PROCESS_COLORS_SYNC )
					{
						{  // additional scope to wrap scoped_lock
							boost::mutex::scoped_lock lock( mutexColorSetSaved_ );
					
							renderMapColors_.clear();
							archive_i >> renderMapColors_;
						}
				
						// wait for display to update if in sync mode
						if ( messageType == PROCESS_COLORS )
						{
							if ( isColorSetDirty_ == true ) // We meant to set colorset dirty but
								// it is already dirty which means the rendering thread has not picked it up yet.
							{
								std::cerr << "skipping colors update in frame..." << std::endl;
							}
	
							isColorSetDirty_ = true;
						}
						else // messageType == PROCESS_COLORS_SYNC
						{
							isColorSetDirty_ = true;
				
							// wait for the updated color set to render to display
							{
								boost::mutex::scoped_lock lock( mutexColorSetUpdated_ );
								while ( isColorSetDirty_ )
								{
									condColorSetUpdated_.wait( lock );
								}
							}
						}
						sendAck( newFd, messageType );
					}
					else if ( messageType == PROCESS_PARTICLES || messageType == PROCESS_PARTICLES_SYNC )
					{
						{ // additional scope to wrap scoped_lock
							boost::mutex::scoped_lock lock( mutexParticlesSaved_ );
							
							archive_i >> vecParticleData_;
						}

						// wait for display to update if in sync mode
						if ( messageType == PROCESS_PARTICLES )
						{
							if ( isParticlesDirty_ == true ) // We meant to set particles dirty but
								// it is already dirty which means the rendering thread has not picked it up yet.
							{
								std::cerr << "skipping particles update in frame..." << std::endl;
							}
	
							isParticlesDirty_ = true;
						}
						else // messageType == PROCESS_PARTICLES_SYNC
						{
							isParticlesDirty_ = true;
				
							// wait for the updated particles to render to display
							{
								boost::mutex::scoped_lock lock( mutexParticlesUpdated_ );
								while ( isParticlesDirty_ )
								{
									condParticlesUpdated_.wait( lock );
								}
							}
						}
						sendAck( newFd, messageType );
					}
				}
				else if ( mode_ == GLVIEW )
				{
					if ( messageType == RESET) 
					{
						GLviewResetData data;
						archive_i >> data;

						initializeRoot( data.strPathName );
						updateGeometryGLview( data );

						isGeometryDirty_ = true;
					}
					else if ( messageType == PROCESS_COLORS || messageType == PROCESS_COLORS_SYNC )
					{
						{  // additional scope to wrap scoped_lock
							boost::mutex::scoped_lock lock( mutexColorSetSaved_ );
							
							if ( ! mapId2GLshapeData_.empty() )
							{
								std::map< unsigned int, GLshapeData* >::iterator id2glshapeIterator;
								for ( id2glshapeIterator = mapId2GLshapeData_.begin();
								      id2glshapeIterator != mapId2GLshapeData_.end();
								      id2glshapeIterator++ )
								{
									delete id2glshapeIterator->second;
								}

								mapId2GLshapeData_.clear();
							}
							archive_i >> mapId2GLshapeData_;
						}
				
						// wait for display to update if in sync mode
						if ( messageType == PROCESS_COLORS )
						{
							if ( isColorSetDirty_ == true ) // We meant to set data set dirty but
								// it is already dirty which means the rendering thread has not picked it up yet.
							{
								std::cerr << "skipping frame..." << std::endl;
							}
	
							isColorSetDirty_ = true;
						}
						else // messageType == PROCESS_COLORS_SYNC
						{
							isColorSetDirty_ = true;
				
							// wait for the updated data set to render to display
							{
								boost::mutex::scoped_lock lock( mutexColorSetUpdated_ );
								while ( isColorSetDirty_ )
								{
									condColorSetUpdated_.wait( lock );
								}
							}
						}		
						sendAck( newFd, messageType );
					}	
				}
			}
		}

		free( buf );
	}

#ifdef WIN32
	closesocket( newFd );
#else
	close( newFd );
#endif
}

void sendAck( int socket, int msgType )
{
	// send back AckPickData structure
	std::ostringstream archiveStream;

	// starting new scope so that the archive's stream's destructor is called after the archive's
	{
		boost::archive::text_oarchive archive( archiveStream );
				
		if ( isPickingDataUpdated_ )
		{
			AckPickData newPick;
			newPick.msgType = msgType;
			newPick.wasSomethingPicked = true;
			{
				boost::mutex::scoped_lock lock( mutexPickingDataUpdated_ );
				newPick.idPicked = pickedId_;
			}
			archive << newPick;
			isPickingDataUpdated_ = false;
		}
		else
		{
			AckPickData noPicks;
			noPicks.msgType = msgType;
			noPicks.wasSomethingPicked = false;
			noPicks.idPicked = 0;

			archive << noPicks;
		}

		std::ostringstream headerStream;
		headerStream << std::setw( MSGSIZE_HEADERLENGTH )
			     << std::hex << archiveStream.str().size();

		unsigned int headerLen = headerStream.str().size() + 1;
		char *headerData = ( char * ) malloc( headerLen * sizeof( char ) );
		strcpy( headerData, headerStream.str().c_str() );

		if ( sendAll( socket, headerData, &headerLen ) == -1 ||
		     headerLen < headerStream.str().size() + 1 )
		{
			std::cerr << "GLclient error: couldn't transmit Ack header to GLcell!" << std::endl;
#ifdef WIN32
			closesocket( socket );
#else
			close( socket );
#endif
		}
		else
		{		
			unsigned int archiveLen = archiveStream.str().size() + 1;
			char* archiveData = ( char * ) malloc( archiveLen * sizeof( char ) );
			strcpy( archiveData, archiveStream.str().c_str() );

			if ( sendAll( socket, archiveData, &archiveLen ) == -1 ||
			     archiveLen < archiveStream.str().size() + 1 )
			{
				std::cerr << "GLclient error: couldn't transmit Ack to GLcell!" << std::endl;
			}

			free( archiveData );
		}
		free( headerData );
	}
}

void initializeRoot( const std::string& strPathName )
{
	root_ = new osg::Group; // root_ is an osg::ref_ptr
	root_->setDataVariance( osg::Object::DYNAMIC );

	if ( textParentBottom_ != NULL)
		delete textParentBottom_;
	textParentBottom_ = new TextBox();
	textParentBottom_->setPosition( osg::Vec3d( 10, 10, 0 ) );
	textParentBottom_->setText( strPathName );
	root_->addChild( textParentBottom_->getGroup() );

	if ( textParentTop_ != NULL)
		delete textParentTop_;
	textParentTop_ = new TextBox();
	textParentTop_->setPosition( osg::Vec3d( 10, WINDOW_HEIGHT - 20, 0 ) );
	textParentTop_->setText( "" );
	root_->addChild( textParentTop_->getGroup() );
}

void updateGeometryGLcell( const GLcellResetData& geometryData )
{	
	double vScale = geometryData.vScale;
	bgcolor_ = osg::Vec4( geometryData.bgcolorRed, geometryData.bgcolorGreen, geometryData.bgcolorBlue, 1.0 ); 
	const std::vector< GLcellProcData >& compartments = geometryData.vecRenderListCompartmentData;
	
	if ( ! mapId2GLCompartment_.empty() )
	{
		for ( std::map< unsigned int, GLCompartment* >::iterator iterator = mapId2GLCompartment_.begin();
		      iterator != mapId2GLCompartment_.end();
		      iterator++ )
		{
			delete iterator->second;
		}
		mapId2GLCompartment_.clear();

		for ( std::map< osg::Geode*, std::pair< unsigned int, std::string* >* >::iterator iterator = mapGeode2NameId_.begin();
		      iterator != mapGeode2NameId_.end();
		      iterator++ )
		{
			delete iterator->second->second;
			delete iterator->second;
		}
		mapGeode2NameId_.clear();
	}

	// First pass: create the basic hollow cylinders with no end-caps
	for ( unsigned int i = 0; i < compartments.size(); ++i )
	{
		const std::string& strName = compartments[i].strName;
		const unsigned int& id = compartments[i].id;
		const std::string& strPathName = compartments[i].strPathName;
		const double& diameter = compartments[i].diameter;
		const double& length = compartments[i].length;
		const double& x0 = compartments[i].x0;
		const double& y0 = compartments[i].y0;
		const double& z0 = compartments[i].z0;
		const double& x = compartments[i].x;
		const double& y = compartments[i].y;
		const double& z = compartments[i].z;
		
		GLCompartment* compartment;

		if ( length < SIZE_EPSILON ||
		     strName.compare("soma") == 0 ) 
			// the compartment is spherical
		{ 
			compartment = new GLCompartmentSphere( osg::Vec3f( x, y, z ),
							       diameter/2,
							       incrementAngle_ );
		}
		else // the compartment is cylindrical
		{ 
			compartment = new GLCompartmentCylinder( osg::Vec3( x0, y0, z0 ),
								 osg::Vec3( x, y, z ),
								 vScale * diameter/2,
								 incrementAngle_ );
		}

		mapId2GLCompartment_[id] = compartment;

		osg::Geometry* geometry = compartment->getGeometry();	
		osg::Geode* geode = new osg::Geode;
		geode->addDrawable( geometry );
		root_->addChild( geode );
			
		mapGeode2NameId_[geode] = new std::pair< unsigned int, std::string* >( id, new std::string( strPathName ) );
	}

	// Second pass: for cylinders only, find neighbours and create interpolated joints
	for ( unsigned int i = 0; i < compartments.size(); ++i )
	{
		const unsigned int& id = compartments[i].id;
		const std::vector< unsigned int > vecNeighbourIds = compartments[i].vecNeighbourIds;

		for ( unsigned int j = 0; j < vecNeighbourIds.size(); ++j )
		{
			if ( mapId2GLCompartment_[id]->getCompartmentType() == COMP_CYLINDER &&
			     mapId2GLCompartment_[vecNeighbourIds[j]]->getCompartmentType() == COMP_CYLINDER )
			{
				dynamic_cast< GLCompartmentCylinder* >( mapId2GLCompartment_[id] )->
					addHalfJointToNeighbour( dynamic_cast< GLCompartmentCylinder* >( mapId2GLCompartment_[vecNeighbourIds[j]] ) );
			}
		} 
	 }

	// Third pass: for cylinders only, form hemispherical end-caps on any joints not yet attached to neighbours.
	for ( unsigned int i = 0; i < compartments.size(); ++i )
	{
		const unsigned int& id = compartments[i].id;

		if ( mapId2GLCompartment_[id]->getCompartmentType() == COMP_CYLINDER)
		{
			dynamic_cast< GLCompartmentCylinder* >( mapId2GLCompartment_[id] )->closeOpenEnds();
		}
	}
}

class updateSmoldynGeometryVisitor : public boost::static_visitor< GLCompartment* >
{
public:
	GLCompartment* operator()( const GLCompartmentCylinderData& cylinderData ) const
	{
		return new GLCompartmentCylinder( cylinderData, incrementAngle_ );
	}

	GLCompartment* operator()( const GLCompartmentDiskData& diskData ) const
	{
		return new GLCompartmentDisk( diskData, incrementAngle_ );
	}

	GLCompartment* operator()( const GLCompartmentHemiData& hemiData ) const
	{
		return new GLCompartmentHemi( hemiData, incrementAngle_ );
	}

	GLCompartment* operator()( const GLCompartmentRectData& rectData ) const
	{
		return new GLCompartmentRect( rectData );
	}

	GLCompartment* operator()( const GLCompartmentSphereData& sphereData ) const
	{
		return new GLCompartmentSphere( sphereData, incrementAngle_ );
	}

	GLCompartment* operator()( const GLCompartmentTriData& triData ) const
	{
		return new GLCompartmentTri( triData );
	}
};

void updateSmoldynGeometry( const std::vector< SmoldynShapeData >& vecSmoldynShapeData )
{
	if ( ! mapGeode2NameId_.empty() )
	{
		for ( unsigned int i = 0; i < vecSmoldynCompartments_.size(); ++i )
		{
			delete vecSmoldynCompartments_[i];
		}
		vecSmoldynCompartments_.clear();

		for ( std::map< osg::Geode*, std::pair< unsigned int, std::string* >* >::iterator iterator = mapGeode2NameId_.begin();
		      iterator != mapGeode2NameId_.end();
		      iterator++ )
		{
			delete iterator->second->second;
			delete iterator->second;
		}
		mapGeode2NameId_.clear();
	}

	for ( unsigned int i = 0; i < vecSmoldynShapeData.size(); ++i )
	{
		GLCompartment* compartment = boost::apply_visitor( updateSmoldynGeometryVisitor(), vecSmoldynShapeData[i].data );
		vecSmoldynCompartments_.push_back( compartment );

		osg::Geometry* geometry = compartment->getGeometry();	
		osg::Geode* geode = new osg::Geode;
		geode->addDrawable( geometry );
		geode->getOrCreateStateSet()->setMode( GL_BLEND,
						       osg::StateAttribute::ON );
		root_->addChild( geode );
			
		if ( ! vecSmoldynShapeData[i].name.empty() )
			mapGeode2NameId_[geode] = new std::pair< unsigned int, std::string* >( 0, new std::string( vecSmoldynShapeData[i].name ) );		

		compartment->setColor( osg::Vec4( vecSmoldynShapeData[i].color[0],
						  vecSmoldynShapeData[i].color[1],
						  vecSmoldynShapeData[i].color[2],
						  vecSmoldynShapeData[i].color[3] ) );
	}
}

void updateGeometryGLview( const GLviewResetData& data )
{
	bgcolor_ = osg::Vec4( data.bgcolorRed, data.bgcolorGreen, data.bgcolorBlue, 1.0 );
	maxsizeGLviewShape_ = data.maxsize;
	const std::vector< GLviewShapeResetData >& vecShapes = data.vecShapes;

	if ( ! mapId2GLviewShape_.empty() )
	{
		std::map< unsigned int, GLviewShape* >::iterator id2glcompIterator;
		for ( id2glcompIterator = mapId2GLviewShape_.begin();
		      id2glcompIterator != mapId2GLviewShape_.end();
		      id2glcompIterator++ )
		{
			delete id2glcompIterator->second;
		}

		mapId2GLviewShape_.clear();

		for ( std::map< osg::Geode*, std::pair< unsigned int, std::string* >* >::iterator iterator = mapGeode2NameId_.begin();
		      iterator != mapGeode2NameId_.end();
		      iterator++ )
		{
			delete iterator->second->second;
			delete iterator->second;
		}
		mapGeode2NameId_.clear();
	}

	for ( unsigned int i = 0; i < vecShapes.size(); ++i )
	{
		const unsigned int& id = vecShapes[i].id;
		const std::string& strPathName = vecShapes[i].strPathName;
		const double& x = vecShapes[i].x;
		const double& y = vecShapes[i].y;
		const double& z = vecShapes[i].z;
		const int& shapetype = vecShapes[i].shapetype;

		GLviewShape * shape = new GLviewShape( id, strPathName,
						       x, y, z,
						       0.5 * maxsizeGLviewShape_, shapetype );
		mapId2GLviewShape_[id] = shape;

		root_->addChild( shape->getGeode() );
		mapGeode2NameId_[ shape->getGeode() ] = new std::pair< unsigned int, std::string* >( id, new std::string( strPathName ) );
	}	
}

void draw()
{
	viewer_ = new osgViewer::Viewer;
	viewer_->setThreadingModel( osgViewer::Viewer::SingleThreaded );

	viewer_->setSceneData( new osg::Geode );
	
	osg::ref_ptr< osg::GraphicsContext::Traits > traits = new osg::GraphicsContext::Traits;
	traits->x = WINDOW_OFFSET_X; // window x offset in window manager
	traits->y = WINDOW_OFFSET_Y; // likewise, y offset ...
	traits->width = WINDOW_WIDTH;
	traits->height = WINDOW_HEIGHT;
	traits->windowDecoration = true;
	traits->doubleBuffer = true;
	traits->sharedContext = 0;

	osg::ref_ptr< osg::GraphicsContext > gc = osg::GraphicsContext::createGraphicsContext( traits.get() );
	
	viewer_->getCamera()->setGraphicsContext( gc.get() );
	viewer_->getCamera()->setViewport( new osg::Viewport( 0, 0, traits->width, traits->height ) );
		
	GLenum buffer = traits->doubleBuffer ? GL_BACK : GL_FRONT;
	viewer_->getCamera()->setDrawBuffer( buffer );
	viewer_->getCamera()->setReadBuffer( buffer );

	viewer_->getCamera()->setCullingMode(viewer_->getCamera()->getCullingMode() &
					     ~osg::CullSettings::SMALL_FEATURE_CULLING); // so that point particles are not culled away

	viewer_->realize();
	viewer_->setCameraManipulator( new osgGA::TrackballManipulator );
	viewer_->addEventHandler( new osgViewer::StatsHandler );
	viewer_->addEventHandler( new KeystrokeHandler );

	screenCaptureHandler_ = new osgViewer::ScreenCaptureHandler( new osgViewer::ScreenCaptureHandler::WriteToFile( getSaveFilename(), "png", osgViewer::ScreenCaptureHandler::WriteToFile::SEQUENTIAL_NUMBER ) );
	viewer_->addEventHandler( screenCaptureHandler_ );

	while ( !viewer_->done() )
	{
		if ( isGeometryDirty_ )
		{
			isGeometryDirty_ = false;

			viewer_->getCamera()->setClearColor( bgcolor_ );			
			viewer_->setSceneData( root_.get() );
		}

		if ( isSmoldynShapesDirty_ )
		{
			isSmoldynShapesDirty_ = false;

			viewer_->setSceneData( root_.get() );
		}

		if ( isColorSetDirty_ ) 
		{
			boost::mutex::scoped_lock lock( mutexColorSetSaved_ );
			
			if ( mode_ == GLCELL )
			{
				std::map< unsigned int, double >::iterator renderMapColorsIterator;
				for ( renderMapColorsIterator = renderMapColors_.begin();
				      renderMapColorsIterator != renderMapColors_.end();
				      renderMapColorsIterator++ )
				{
					unsigned int id = renderMapColorsIterator->first;
					double color = renderMapColorsIterator->second;

					GLCompartment* glcompartment = mapId2GLCompartment_[id];

					int ix;				
					if ( color <= (0 + FP_EPSILON) ) // color <= 0
					{
						ix = 0;
					}
					else if ( color >= (1 - FP_EPSILON) ) // color >= 1
					{
						ix = vecColormap_.size()-1;
					}
					else
					{
						ix = static_cast< int >( floor( color * vecColormap_.size() ) );
					}
					double red = vecColormap_[ ix ][ 0 ];
					double green = vecColormap_[ ix ][ 1 ];
					double blue = vecColormap_[ ix ][ 2 ];
					glcompartment->setColor( osg::Vec4( red, green, blue, 1.0f ) );
				}
			}
			else if ( mode_ == GLVIEW )
			{
				std::map< unsigned int, GLshapeData* >::iterator id2glshapeIterator;
				for ( id2glshapeIterator = mapId2GLshapeData_.begin();
				      id2glshapeIterator != mapId2GLshapeData_.end();
				      id2glshapeIterator++ )
				{
					unsigned int id = id2glshapeIterator->first;
					GLshapeData* newGLshape = id2glshapeIterator->second;
					GLviewShape* glViewShape = mapId2GLviewShape_[id];

					if ( newGLshape->len > (-1 + FP_EPSILON) ) // newGLshape->len > -1
					{
						glViewShape->resize( newGLshape->len * maxsizeGLviewShape_ );
					}

					if ( newGLshape->color > (-1 + FP_EPSILON) ) // newGLshape->color > -1
					{
						int ix;			
						if ( newGLshape->color <= (0 + FP_EPSILON) ) // newGLshape->color <= 0
						{
							ix = 0;
						}
						else if ( newGLshape->color >= (1 - FP_EPSILON) ) // newGLshape->color >= 1
						{
							ix = vecColormap_.size()-1;
						}
						else
						{
							ix = static_cast< int >( floor( newGLshape->color * vecColormap_.size() ) );
						}
						double red = vecColormap_[ ix ][ 0 ];
						double green = vecColormap_[ ix ][ 1 ];
						double blue = vecColormap_[ ix ][ 2 ];
						glViewShape->setColor( osg::Vec4( red, green, blue, 1.0f ) );
					}

					if ( fabs( newGLshape->xoffset - 0 ) > FP_EPSILON ||
					     fabs( newGLshape->yoffset - 0 ) > FP_EPSILON ||
					     fabs( newGLshape->zoffset - 0 ) > FP_EPSILON )
					{
						glViewShape->move( newGLshape->xoffset,
								   newGLshape->yoffset,
								   newGLshape->zoffset );
					}
				}
			}

			{
				boost::mutex::scoped_lock lock2( mutexColorSetUpdated_ );
				isColorSetDirty_ = false;				
			}
			condColorSetUpdated_.notify_one(); // no-op except when responding to PROCESS_COLORS_SYNC
		}

		if ( isParticlesDirty_ )
		{
			boost::mutex::scoped_lock lock( mutexParticlesSaved_ );
			
			for ( unsigned int i = 0; i < vecParticleGeodes_.size(); ++i )
			{
				root_->removeChild( vecParticleGeodes_[i] );
			}

			for ( unsigned int i = 0; i < vecParticleData_.size(); ++i )
			{
				ParticleData* particleData = &vecParticleData_[i];

				if ( particleData->diameter <= 0 + FP_EPSILON )
				{
					osg::Geode* geode = new osg::Geode;
					osg::Geometry* geometry = new osg::Geometry;

					osg::ref_ptr< osg::Vec3Array > vertices = new osg::Vec3Array;
					for ( unsigned int j = 0; j < particleData->vecCoords.size(); j += 3 )
					{
						vertices->push_back( osg::Vec3( particleData->vecCoords[j],
										particleData->vecCoords[j+1],
										particleData->vecCoords[j+2] ) );
					}
					geometry->setVertexArray( vertices.get() );
				
					osg::ref_ptr< osg::Vec4Array > colors = new osg::Vec4Array;
					colors->push_back( osg::Vec4( particleData->color[0],
								      particleData->color[1],
								      particleData->color[2],
								      1.0f ) );
					geometry->setColorArray( colors.get() );
					geometry->setColorBinding( osg::Geometry::BIND_OVERALL );
				
					osg::DrawArrays* drawArrays = new osg::DrawArrays( osg::PrimitiveSet::POINTS,
											   0,
											   vertices->size() );
					geometry->addPrimitiveSet( drawArrays );
					geode->getOrCreateStateSet()->setMode( GL_LIGHTING,
									       osg::StateAttribute::OFF );
				
					osg::ref_ptr< osg::Point > point = new osg::Point;
					point->setSize( POINT_PARTICLE_DIAMETER );
					geode->getOrCreateStateSet()->setAttribute( point.get(),
										    osg::StateAttribute::ON );
					geode->addDrawable( geometry );

					root_->addChild( geode );
					vecParticleGeodes_.push_back( geode );
				}
				else
				{
					for ( unsigned int j = 0; j < particleData->vecCoords.size(); j += 3 )
					{
						GLCompartmentSphere* sphere = new GLCompartmentSphere( osg::Vec3f( particleData->vecCoords[j],
														   particleData->vecCoords[j+1],
														   particleData->vecCoords[j+2] ),
												       particleData->diameter/2,
												       incrementAngle_ );
						sphere->setColor( osg::Vec4( particleData->color[0],
									     particleData->color[1],
									     particleData->color[2],
									     1.0f ) );
						osg::Geometry* sphereGeom = sphere->getGeometry();
						
						osg::Geode* geode = new osg::Geode;
						geode->addDrawable( sphereGeom );
						
						root_->addChild( geode );
						vecParticleGeodes_.push_back( geode );
					}
				}
			}
			vecParticleData_.clear();
			
			{
				boost::mutex::scoped_lock lock2( mutexParticlesUpdated_ );
				isParticlesDirty_ = false;				
			}
			condParticlesUpdated_.notify_one(); // no-op except when responding to PROCESS_PARTICLES_SYNC
				
		}

		if ( isSavingMovie_ )
			screenCaptureHandler_->captureNextFrame( *viewer_ );

		viewer_->frame();
	}
}

std::string getSaveFilename( void )
{
	std::stringstream filename;
	filename << "Screenshot_";

	/*char strTime[26];
	time_t t;
	time( &t );
	strcpy( strTime, ctime( &t ) );
	strTime[24] = '\0'; // strip '\n'

	filename << strTime;*/

	filename << static_cast<long>( time( NULL ) );
	
	boost::filesystem::path fullPath( saveDirectory_ / filename.str() );
	return fullPath.string();
}

int main( int argc, char* argv[] )
{
	int c;
	std::string strHelp = "Usage: glclient\n"
		"\t-p <number>: port number\n"
		"\t-c <string>: filename of colormap file\n"
		"\t-m <string>: 'c' or 'v', for connection with MOOSE element of type GLcell or GLview respectively\n"
		"\t[-d <string>: pathname in which to save screenshots and sequential image files (default is ./)]\n"
		"\t[-a <number>: required to be between 1 and 60 degrees; this value represents angular increments in drawing the sides of curved bodies; smaller numbers give smoother bodies (default is 10)]\n";
	
	bool isValid;
	double value;

	// Check command line arguments.
	while ( ( c = getopt( argc, argv, "hp:c:m:d:a:" ) ) != -1 )
		switch( c )
		{
		case 'h':
			std::cout << strHelp << std::endl;
			return 1;
		case 'p':
			port_ = optarg;
			break;
		case 'c':
			fileColormap_ = optarg;
			break;
		case 'm':
			if ( optarg[0] == 'c' )
			{
				mode_ = GLCELL;
			}
			else if ( optarg[0] == 'v' )
			{
				mode_ = GLVIEW;
			}
			else
			{
				std::cerr << "Unknown mode specifier argument to -m!" << std::endl;
				return -1;
			}
			break;
		case 'd':
			saveDirectory_ = optarg;
			isValid = boost::filesystem::is_directory(saveDirectory_);
			if ( !isValid )
			{
				printf( "Argument to option -d must be a valid directory name.\n" );
				return 1;
			}
			break;
		case 'a':
			value = strtod( optarg, NULL );
			if ( value < 1 )
				incrementAngle_ = 1;
			else if ( value > 60 )
				incrementAngle_ = 60;
			else
				incrementAngle_ = value;
			break;
		case '?':
			if ( optopt == 'm' || optopt == 'p' || optopt == 'c' || optopt == 'u' || optopt == 'l' || optopt == 'd' )
				printf( "Option -%c requires an argument.\n", optopt );
			else
				printf( "Unknown option -%c.\n", optopt );
			return 1;
		default:
			return 1;
		}
	
	if ( port_ == NULL || fileColormap_ == NULL || mode_ == NONE )
	{
		std::cerr << "-p, -c and -m are required.\n\n";
		std::cerr << strHelp << std::endl;
		return 1;
	}

	// parse colormap file
	std::ifstream fColormap;
	std::string lineToSplit;
	char* pEnd;

	fColormap.open( fileColormap_ );
	if ( ! fColormap.is_open() )
	{
		std::cerr << "Couldn't open colormap file: " << fileColormap_ << "!" << std::endl;
		return 2;
	}
	else
	{
		while ( ! fColormap.eof() )
		{
			getline( fColormap, lineToSplit );
			if ( lineToSplit.length() > 0 )  // not a blank line (typically the last line)
			{
				osg::Vec3d color;
				color[0] = strtod( lineToSplit.c_str(), &pEnd ) / 65535.;
				color[1] = strtod( pEnd, &pEnd ) / 65535.;
				color[2] = strtod( pEnd, NULL ) / 65535.;
				
				vecColormap_.push_back( color );
			}
		}
	}
	fColormap.close();
	
	// launch network thread and run the GUI in the main loop
	boost::thread threadProcess( networkLoop );
	draw();

#ifdef WIN32
	WSACleanup();
#endif

	return 0;
}
