/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "GLcellProcData.h"
#include "GLcellResetData.h"
#include "ParticleData.h"
#include "SmoldynShapeData.h"
#include "Constants.h"

class GLcell
{
 public:
	GLcell();
	~GLcell();

	static void process( const Conn* c, ProcInfo p );
	
	static void processFunc( const Conn* c, ProcInfo p );
        void processFuncLocal( Eref e, ProcInfo info );
	
	static void reinitFunc( const Conn* c, ProcInfo info );
	void reinitFuncLocal( const Conn* c );

	static void setPath( const Conn* c, string strPath );
	void innerSetPath( const string& strPath );
	static string getPath( Eref e );

	static void setClientHost( const Conn* c, string strClientHost );
	void innerSetClientHost( const string& strClientHost );
	static string getClientHost( Eref e );

	static void setClientPort( const Conn* c, string strClientPort );
	void innerSetClientPort( const string& strClientPort );
	static string getClientPort( Eref e );
	
	static void setAttributeName( const Conn* c, string strAttributeName );
	void innerSetAttributeName( const string& strAttributeName );
	static string getAttributeName( Eref e );

	static void setChangeThreshold( const Conn* c, double changeThreshold );
	void innerSetChangeThreshold( const double changeThreshold );
	static double getChangeThreshold( Eref e );

	static void setVScale( const Conn* c, double vScale );
	void innerSetVScale( const double vScale );
	static double getVScale( Eref e );

	static void setSyncMode( const Conn* c, string syncMode );
	void innerSetSyncMode( const bool syncMode );
	static string getSyncMode( Eref e );

	static void setBgColor( const Conn* c, string strBgColor );
	void innerSetBgColor( const double red, const double green, const double blue );
	static string getBgColor( Eref e );

	static void setHighValue( const Conn* c, double highValue );
	void innerSetHighValue( const double highValue );
	static double getHighValue( Eref e );

	static void setLowValue( const Conn* c, double lowValue );
	void innerSetLowValue( const double lowValue );
	static double getLowValue( Eref e );

	static void setParticleData( const Conn* c, vector< ParticleData > vecParticleData );
	void innerSetParticleData( const vector< ParticleData > vecParticleData );

	static void setSmoldynShapeData( const Conn* c, vector< SmoldynShapeData > vecSmoldynShapeData );
	void innerSetSmoldynShapeData( const vector< SmoldynShapeData > vecSmoldynShapeData );

	static const int MSGTYPE_HEADERLENGTH;
	static const int MSGSIZE_HEADERLENGTH;
	static const char SYNCMODE_ACKCHAR;

 private:

	string strPath_;
	string strClientHost_;
	string strClientPort_;
	bool isConnectionUp_;
	string strAttributeName_;
	int sockFd_;
	double changeThreshold_; // any change in attribute below this value is not updated visually (if not in sync mode)
	double vScale_; // factor by which the diameter of cylindrical compartments will be scaled up (only in visual appearance, not numerically)
	bool syncMode_;
	double bgcolorRed_;
	double bgcolorGreen_;
	double bgcolorBlue_;
	double highValue_;
	double lowValue_;

	double testTicker_; // used by testInsertVecParticleData()

	vector< Id > vecRenderList_;
	vector< ParticleData > vecParticleData_;
	vector< SmoldynShapeData > vecSmoldynShapeData_;

	map< unsigned int, double > renderMapAttrsLastTransmitted_;
	map< unsigned int, double > renderMapAttrsTransmitted_;

	map< unsigned int, double> mapAttrs2Colors( map< unsigned int, double > renderMapAttrs );

	void add2RenderList( Id id );
	void findNeighbours( Id id, std::vector< unsigned int>& vecResult );
	void findNeighboursOfType( Id id, const string& messageType, const string& targetType, std::vector< unsigned int >& vecResult );

	/// networking helper functions
	void* getInAddress( struct sockaddr *sa );
	int getSocket( const char* hostname, const char* service );
	int sendAll( int socket, char* buf, unsigned int* len );
	int recvAll( int socket, char* buf, unsigned int* len);
	int receiveAck();
	void handlePick( unsigned int idPicked );
	void disconnect();
	template< class T >
	  void transmit( T& data, MsgType messageType );
	
	// testing functions
	void testInsertVecSmoldynShapeData( void );
	void testShape1( void );
	void testShape2( void );
	void testInsertVecParticleData( void );
	void testParticle1( void );
	void testParticle2( void );
  

#ifdef WIN32
	int initWinsock( void );
#endif

};

