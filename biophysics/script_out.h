/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SCRIPT_OUT_H
#define _SCRIPT_OUT_H

/**
 * The script_out class sets up an asymmetric compartment for
 * branched nerve calculations. Handles electronic structure and
 * also channels. This is not a particularly efficient way of doing
 * this, so we should use a solver for any substantial calculations.
 */
class script_out
{
	public:
			script_out();
			virtual ~script_out() {;}

			static void processFunc( const Conn* c, ProcInfo p );
			static void reinitFunc( const Conn* c, ProcInfo p );
			static void initFunc( const Conn* c, ProcInfo p );
			static void initReinitFunc( const Conn* c, ProcInfo p );
			static void setaction( const Conn* c, int action );
			static int getaction( Eref );
			static void setX(const Conn* c, double value);
			static void setY(const Conn* c, double value);
			static void setH(const Conn* c, double value);
			static void setW(const Conn* c, double value);
			static void setDX(const Conn* c, double value);
			static void setDY(const Conn* c, double value);
			static double getX( Eref e );
			static double getY( Eref e );
			static double getH( Eref e );
			static double getW( Eref e );
			static double getDX( Eref e );
			static double getDY( Eref e );

	protected:
			
	private:
		virtual void innerProcessFunc( Eref e, ProcInfo p );
		int action_;
		double bar_x_,bar_y_,bar_w_,bar_h_,bar_dx_,bar_dy_;
};


#endif // _SCRIPT_OUT_H
