/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef SHAPETYPE
#define SHAPETYPE

enum SHAPETYPE
{
	CUBE,
	SPHERE
};

#endif // SHAPETYPE

class GLshape
{
 public:
	GLshape();
	~GLshape();

	static void process( const Conn* c, ProcInfo info );
	
	static void processFunc( const Conn* c, ProcInfo info );
	void processFuncLocal( Eref e, ProcInfo info );

	static void reinitFunc( const Conn* c, ProcInfo info );
	void reinitFuncLocal( const Conn* c );

	static void setColor( const Conn* c, double color );
	void innerSetColor( double color );
	static double getColor( Eref e );

	static void setXOffset( const Conn* c, double xoffset );
	void innerSetXOffset( double xoffset );
	static double getXOffset( Eref e );

	static void setYOffset( const Conn* c, double yoffset );
	void innerSetYOffset( double yoffset );
	static double getYOffset( Eref e );

	static void setZOffset( const Conn* c, double zoffset );
	void innerSetZOffset( double zoffset );
	static double getZOffset( Eref e );

	static void setLen( const Conn* c, double len );
	void innerSetLen( double len );
	static double getLen( Eref e );

	static void setShapeType( const Conn* c, int shapetype );
	void innerSetShapeType( int shapetype );
	static int getShapeType( Eref e );

 private:
	double color_;
	double xoffset_;
	double yoffset_;
	double zoffset_;
	double len_;
	int shapetype_;
};
