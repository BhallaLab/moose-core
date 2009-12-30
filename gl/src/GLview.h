/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "GLshapeData.h"
#include "Constants.h"

class GLview
{
 public:
	GLview();
	~GLview();

	static void process( const Conn* c, ProcInfo info );

	static void processFunc( const Conn* c, ProcInfo info );
	void processFuncLocal( Eref e, ProcInfo info );

	static void reinitFunc( const Conn* c, ProcInfo info );
	void reinitFuncLocal( const Conn* c );

	static void setClientHost( const Conn* c, string strClientHost );
	void innerSetClientHost( const string& strClientHost );
	static string getClientHost( Eref e );

	static void setClientPort( const Conn* c, string strClientPort );
	void innerSetClientPort( const string& strClientPort );
	static string getClientPort( Eref e );

	static void setPath( const Conn* c, string strPath );
	void innerSetPath( const string& strPath );
	static string getPath( Eref e );
	
	static void setRelPath( const Conn* c, string strRelPath );
	void innerSetRelPath( const string& strRelPath );
	static string getRelPath( Eref e );

	static void setValue1Field( const Conn* c, string strValue1Field );
	void innerSetValue1Field( const string& strValue1Field );
	static string getValue1Field( Eref e );

	static void setValue2Field( const Conn* c, string strValue2Field );
	void innerSetValue2Field( const string& strValue2Field );
	static string getValue2Field( Eref e );
	
	static void setValue3Field( const Conn* c, string strValue3Field );
	void innerSetValue3Field( const string& strValue3Field );
	static string getValue3Field( Eref e );
	
	static void setValue4Field( const Conn* c, string strValue4Field );
	void innerSetValue4Field( const string& strValue4Field );
	static string getValue4Field( Eref e );
	
	static void setValue5Field( const Conn* c, string strValue5Field );
	void innerSetValue5Field( const string& strValue5Field );
	static string getValue5Field( Eref e );

	static void setBgColor( const Conn* c, string strBgColor );
	void innerSetBgColor( const double red, const double green, const double blue );
	static string getBgColor( Eref e );

	static void setSyncMode( const Conn* c, string syncMode );
	void innerSetSyncMode( const bool syncMode );
	static string getSyncMode( Eref e );

	static void setGridMode( const Conn* c, string gridMode );
	void innerSetGridMode( const bool gridMode );
	static string getGridMode( Eref e );

	static void setColorVal( const Conn* c, unsigned int colorVal );
	void innerSetColorVal( unsigned int colorVal );
	static unsigned int getColorVal( Eref e );

	static void setMorphVal( const Conn* c, unsigned int morphVal );
	void innerSetMorphVal( unsigned int morphVal );
	static unsigned int getMorphVal( Eref e );

	static void setXOffsetVal( const Conn* c, unsigned int xoffsetVal );
	void innerSetXOffsetVal( unsigned int xoffsetVal );
	static unsigned int getXOffsetVal( Eref e );

	static void setYOffsetVal( const Conn* c, unsigned int yoffsetVal );
	void innerSetYOffsetVal( unsigned int yoffsetVal );
	static unsigned int getYOffsetVal( Eref e );

	static void setZOffsetVal( const Conn* c, unsigned int zoffsetVal );
	void innerSetZOffsetVal( unsigned int zoffsetVal );
	static unsigned int getZOffsetVal( Eref e );
	
	static void setValueMin( const Conn* c, unsigned int index, double value );
	void innerSetValueMin( unsigned int index, double value );

	static void setValueMax( const Conn* c, unsigned int index, double value );
	void innerSetValueMax( unsigned int index, double value ); 

	static const int MSGTYPE_HEADERLENGTH;
     	static const int MSGSIZE_HEADERLENGTH;
	static const char SYNCMODE_ACKCHAR;

 private:
	int sockFd_;
	bool isConnectionUp_;
	string strClientHost_;
	string strClientPort_;
	bool syncMode_;
	bool gridMode_;
	double bgcolorRed_;
	double bgcolorGreen_;
	double bgcolorBlue_;

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

#ifdef WIN32
	int initWinsock( void );
#endif


	string strPath_;
	string strRelPath_;

	// the elements found on strPath_
	vector< Id > vecElements_;
	// child elements of the type GLshape used as interpolation targets
	vector< Eref > vecErefGLshapeChildren_;

	double* values_[5];
	double value_min_[5];
	double value_max_[5];
	string strValueField_[5];

	unsigned int color_val_;
	unsigned int morph_val_;
	unsigned int xoffset_val_;
	unsigned int yoffset_val_;
	unsigned int zoffset_val_;

	double* x_;
	double* y_;
	double* z_;

	std::map< unsigned int, GLshapeData* > mapId2GLshapeData_;

	int populateValues( int valueNum, double ** pValues, const string& strValueField );
	
	double populateXYZ();
	string boxXYZ( const double& x, const double& y, const double& z );
	string inttostring( int i );
	
	// helper functions taken (largely) from BioScan; they are
	// private there or I would use them directly
	int children( Id object, vector< Eref >& ret, const string& type );
	bool isType( Id object, const string& type );

	void chooseInterpolationPair( const int& numTargets, const double& val,
				      const double& val_min, const double& val_max,
				      unsigned int& iLow, unsigned int& iHigh );

	void interpolate( const double& val_min, const double& attr_min,
			  const double& val_max, const double& attr_max,
			  const double& val, double& attr );

	// gets x, y, z co-ordinates for the element represented by id, or if not found
	// such co-ordinates of its parent or its parent's parent and so on, unless root is reached
	int getXYZ( Id id, double& x, double& y, double& z, double &maxsize ); 
	
};
