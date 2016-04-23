/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ODE_SYSTEM_H
#define _ODE_SYSTEM_H

class BoostSys;

class OdeSystem {
    public:
        OdeSystem()
            : method( "rk54" ),
            initStepSize( 1 ),
            epsAbs( 1e-6 ),
            epsRel( 1e-6 )
    {;}

        string method;

#ifdef USE_BOOST
        BoostSys* boostSys;
#endif

        double initStepSize;

        double epsAbs; // Absolute error
        double epsRel; // Relative error
};

#endif // _ODE_SYSTEM_H
