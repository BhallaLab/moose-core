/***
 *    Description:  ODE system.
 *
 *        Created:  2018-12-28

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  See LICENSE file.
 */

#ifndef ODESYSTEM_H
#define ODESYSTEM_H

#include <boost/numeric/odeint.hpp>
#include <vector>
#include <map>

using namespace std;

namespace moose {

class OdeSystem 
{
public:
    OdeSystem();
    ~OdeSystem();

    void setEquations(const vector<string>& eqs);
    vector<string> getEquations(void) const;
    void buildSystem(void* obj);
    bool isValid(void);

private:
    vector<double> dydt_;
    vector<double> res_;
    vector<string> eqs_;
    bool isValid_ = false;

    // Map lhs = rhs 
    map<string, string> odeMap_;
    map<string, string> eqsMap_;
    vector<string> symbols_;
};

} // namespace moose

#endif /* end of include guard: ODESYSTEM_H */
