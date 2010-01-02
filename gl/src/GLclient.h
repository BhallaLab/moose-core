/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "Constants.h"
#include "AckPickData.h"
#include "ParticleData.h"
#include "SmoldynShapeData.h"
#include "TextBox.h"

class KeystrokeHandler : public osgGA::GUIEventHandler
{
 public:
 KeystrokeHandler() :
	isCurrentProjectionOrtho_(false)
	{}
	
	virtual bool handle( const osgGA::GUIEventAdapter&, osgGA::GUIActionAdapter&, osg::Object*, osg::NodeVisitor* );

 private:
	~KeystrokeHandler() {}
	bool pick( const double x, const double y, osgViewer::Viewer* viewer );
	void switchProjection( osgViewer::Viewer* viewer );

	double x_;
	double y_;

	double isCurrentProjectionOrtho_;
};

void networkLoop( void );
int acceptNewConnection( char * port );
void receiveData( int newFd );

void* getInAddr( struct sockaddr* sa );
int sendAll( int socket, char* buf, int* len );
int recvAll( int socket, char* buf, int* len);
void sendAck( int socket, int msgType );
void initializeRoot( const std::string& pathName );
void updateGeometryGLcell( const GLcellResetData& geometry );
void updateGeometryGLview( const GLviewResetData& data );
void updateSmoldynGeometry( const std::vector< SmoldynShapeData >& vecSmoldynShapeData );

std::string getSaveFilename( void );

#ifdef WIN32
int initWinsock( void );
#endif

osg::ref_ptr< osg::Group > root_;

std::vector< ParticleData > vecParticleData_;
std::vector< osg::Geode* > vecParticleGeodes_;

TextBox* textParentTop_ = NULL;
TextBox* textParentBottom_ = NULL;

// GLcell mode: 
// this is used to call the function setColor() and also acts as general storage for GLCompartment
std::map< unsigned int, GLCompartment* > mapId2GLCompartment_;

// Attribute values mapped to colors, data received in PROCESS step:
std::map< unsigned int, double > renderMapColors_;

// GLview mode:
// used to resize, move and recolor displayed shapes
std::map< unsigned int, GLviewShape* > mapId2GLviewShape_;
// data received in PROCESS step
std::map< unsigned int, GLshapeData* > mapId2GLshapeData_;

// both modes:
std::map< osg::Geode*, std::pair< unsigned int, std::string* >* > mapGeode2NameId_; 
// this is used to obtain the id of a compartment or shape that the user has picked with the mouse
double maxsizeGLviewShape_;

volatile bool isGeometryDirty_ = false;
volatile bool isColorSetDirty_ = false;

volatile bool isParticlesDirty_ = false;
boost::mutex mutexParticlesSaved_;
boost::mutex mutexParticlesUpdated_;
boost::condition condParticlesUpdated_;

volatile bool isSmoldynShapesDirty_ = false;

std::vector< GLCompartment* > vecSmoldynCompartments_;

volatile bool isPickingDataUpdated_ = false;
boost::mutex mutexPickingDataUpdated_;
unsigned int pickedId_;

char * port_ = NULL;
char * fileColormap_ = NULL;
double incrementAngle_ = DEFAULT_INCREMENT_ANGLE;

const int MSGTYPE_HEADERLENGTH = 1;
const int MSGSIZE_HEADERLENGTH = 8;
const int BACKLOG = 10; // how many pending connections will be queued

enum MODETYPE
{
	NONE,
	GLCELL,
	GLVIEW
};
int mode_ = NONE;

bool isSavingMovie_ = false;
boost::filesystem::path saveDirectory_(".");

osg::Vec4 bgcolor_ = osg::Vec4( 0.0, 0.0, 0.0, 1.0 );

// Used for both GLcell and GLview
boost::mutex mutexColorSetSaved_;
boost::mutex mutexColorSetUpdated_;
boost::condition condColorSetUpdated_;

std::vector< osg::Vec3d > vecColormap_;

osgViewer::ScreenCaptureHandler* screenCaptureHandler_;
osgViewer::Viewer* viewer_;

