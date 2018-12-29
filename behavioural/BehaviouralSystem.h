/***
 *    Description:  Behavioural system.
 *
 *        Created:  2018-12-29

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  MIT License
 */

#ifndef BEHAVIOURALSYSTEM_H
#define BEHAVIOURALSYSTEM_H

#include <memory>
#include "OdeSystem.h"

using namespace std;

namespace moose {

/*
 * =====================================================================================
 *        Class:  BehaviouralSystem
 *  Description:  
 * =====================================================================================
 */
class BehaviouralSystem 
{
    public:
        BehaviouralSystem ();                             /* constructor      */
        ~BehaviouralSystem ();                            /* destructor       */

        void buildSystem();

        OdeSystem* getODE() const;

    protected:

    private:
        OdeSystem* ode_;

}; /* -----  end of class BehaviouralSystem  ----- */


} // namespace moose.

#endif /* end of include guard: BEHAVIOURALSYSTEM_H */
