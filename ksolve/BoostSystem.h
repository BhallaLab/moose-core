#ifndef BOOSTSYSTEM_H
#define BOOSTSYSTEM_H

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
        BoostSys (); 

        /*-----------------------------------------------------------------------------
         *  Following functiors implement equivalent of 'function' and
         *  `jacobian` of gsl_odeiv2_system. These wrappers are just to have
         *  consistency between calls to gsl or boost solver.
         *-----------------------------------------------------------------------------*/
        int operator()( double y , double& dydt ,  double t );

        // Fixme: Change the types of argument.
        int (*jacobian) ( double t
                , const double y[]
                , double dfdt[]);

        size_t dimensions;                      /* dimensions of the system */

        /* Pointer to the arbitrary parameters of the system */
        void * params;

        boost::numeric::odeint::runge_kutta_dopri5< double, double ,double
            , double, boost::numeric::odeint::vector_space_algebra > stepper;
};

#endif // USE_BOOST

#endif /* end of include guard: BOOSTSYSTEM_H */

