#ifndef BOOSTSYSTEM_H
#define BOOSTSYSTEM_H

#ifdef USE_BOOST

#include <vector>
#include <boost/numeric/odeint.hpp>


#ifndef USE_CUDA
typedef double value_type_;
typedef std::vector<value_type_> state_type_;
typedef boost::numeric::odeint::runge_kutta_dopri5< state_type_ > stepper_type_;
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
 *  Description:  The ode system describing chemical kinetics in BOOST.
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

