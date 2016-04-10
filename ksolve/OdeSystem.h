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

#ifdef USE_BOOST

#include <boost/numeric/odeint.hpp>
typedef boost::numeric::ublas::vector< double > vector_type_;
typedef boost::numeric::ublas::matrix< double > matrix_type_;

/*
 * =====================================================================================
 *        Class:  BoostSys
 *  Description:  The ode system describing chemical kinetics in BOOST.
 * =====================================================================================
 */
class BoostSys
{
    public:
        BoostSys (){ };                             /* constructor */

        /*-----------------------------------------------------------------------------
         *  Following functiors implement equivalent of 'function' and
         *  `jacobian` of gsl_odeiv2_system. These wrappers are just to have
         *  consistency between calls to gsl or boost solver.
         *-----------------------------------------------------------------------------*/
        int (*rhs) (  double y
                , double& dydt
                ,  double t 
                //, void * params 
                );

        // Fixme: Change the types of argument.
        int (*jacobian) ( double t
                , const double y[]
                , double dfdt[]
                , void* params
                );

        size_t dimensions;                      /* dimensions of the system */

        /* Pointer to the arbitrary parameters of the system */
        void * params;

        boost::numeric::odeint::runge_kutta_dopri5< double, double ,double
            , double, boost::numeric::odeint::vector_space_algebra > stepper;

}; /* -----  end of class BoostSys  ----- */


#endif


class OdeSystem {
	public:
		OdeSystem()
				: method( "rk5" ),
					initStepSize( 1 ),
					epsAbs( 1e-6 ),
					epsRel( 1e-6 )
		{;}

		string method;
		// GSL stuff
#ifdef USE_GSL
		gsl_odeiv2_system gslSys;
		const gsl_odeiv2_step_type* gslStep;

#elif defined(USE_BOOST)
                BoostSys boostSys;
#endif
		double initStepSize;

		double epsAbs; // Absolute error
		double epsRel; // Relative error
};

#endif // _ODE_SYSTEM_H
