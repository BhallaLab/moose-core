#ifndef BOOSTSYSTEM_H
#define BOOSTSYSTEM_H

#ifdef USE_BOOST

#include <vector>
#include <boost/numeric/odeint.hpp>


/*-----------------------------------------------------------------------------
 *  NOTICE:
 *  Before writing typedef for stepper read this 
 *  http://stackoverflow.com/questions/36564285/template-parameters-of-boostnumericodeintrunge-kutta-x-compatible-with-c/36564610?noredirect=1#comment60732822_36564610
 *-----------------------------------------------------------------------------*/
#ifndef USE_CUDA
typedef double value_type_;
typedef std::vector<value_type_> state_type_;
typedef boost::numeric::odeint::runge_kutta4< state_type_ > stepper_type_;
//typedef boost::numeric::odeint::runge_kutta_dopri5< state_type_ > stepper_type_;
//typedef boost::numeric::odeint::runge_kutta_cash_karp54< state_type_ > stepper_type_;
#else
typedef double value_type_;
typedef trust::device_vector< value_type_ > state_type_;
typedef boost::numeric::odeint::runge_kutta_dopri5< 
        state_type_
        , value_type_
        , state_type_
        , value_type_
        , boost::numeric::odeint::thrust_algebra
        , boost::numeric::odeint::thrust_operations
    >  stepper_type_;
#endif

/*
 * =====================================================================================
 *        Class:  BoostSys
 *  Description:  The ode system of ksolve. It uses boost library to solve it.
 *  It is intended to be gsl replacement.
 * =====================================================================================
 */
class BoostSys
{
    public:
        BoostSys (); 
        ~BoostSys();

        /* Pointer to the arbitrary parameters of the system */
        void * params;

        stepper_type_ stepper;
};

#endif // USE_BOOST

#endif /* end of include guard: BOOSTSYSTEM_H */

