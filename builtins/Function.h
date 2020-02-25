// Function.h ---
// Description: moose.Function class.
// Author: Subhasis Ray
// Maintainer: Dilawar Singh
// Version: See git logs.

#ifndef FUNCTIONH_
#define FUNCTIONH_
#include <memory>

class Variable;
class Eref;
class Cinfo;

#include "../builtins/MooseParser.h"

/**
   Expression parser and evaluator based on ExprTK. 
 */
// double *functionAddVar_(const char *name, void *data);

class Function 
{
public:
    static const int VARMAX;
    Function();
    Function(const Function& f);

    // Destructor.
    ~Function();

    // copy operator.
    Function& operator=(const Function& rhs);

    static const Cinfo * initCinfo();

    void setExpr(const Eref& e, const string expr);
    bool innerSetExpr(const Eref& e, const string expr);

    string getExpr(const Eref& e) const;

    // get a list of variable identifiers.
    // this is created by the parser
    vector<string> getVars() const;
    void setVarValues(vector<string> vars, vector<double> vals);

    // get/set the value of variable `name`
    void setVar(unsigned int index, double value);
    Variable * getVar(unsigned int ii);

    // get function eval result
    double getValue() const;
    double getRate() const;

    // get/set operation mode
    void setMode(unsigned int mode);
    unsigned int getMode() const;

    // set/get flag to use trigger mode.
    void setUseTrigger(bool useTrigger);
    bool getUseTrigger() const;

    // set/get flag to do function evaluation at reinit
    void setDoEvalAtReinit(bool doEvalAtReinit);
    bool getDoEvalAtReinit() const;

    void setNumVar(unsigned int num);
    unsigned int getNumVar() const;

    void setConst(string name, double value);
    double getConst(string name) const;

    void setIndependent(string index);
    string getIndependent() const;

    vector < double > getY() const;

    double getDerivative() const;

    void findXsYs( const string& expr, vector<string>& vars );

    unsigned int addVar();
    /* void dropVar(unsigned int msgLookup); */

    void process(const Eref& e, ProcPtr p);
    void reinit(const Eref& e, ProcPtr p);

    void addVariable(const string& name);

    void showError(moose::Parser::exception_type &e) const;


protected:

    bool valid_;
    unsigned int numVar_;
    double lastValue_;
    double value_;
    double rate_;
    unsigned int mode_;
    bool useTrigger_;
    bool doEvalAtReinit_;

    double t_;                             // local storage for current time
    string independent_;                   // To take derivative.

    // this stores variables received via incoming messages, identifiers of
    // the form x{i} are included in this
    vector<Variable*> xs_;

    // this stores variable values pulled by sending request. identifiers of
    // the form y{i} are included in this
    vector<double*> ys_;

    // Used by kinetic solvers when this is zombified.
    void* stoich_;

    // Parser which should never be copied. Multithreaded programs may behave
    // strangely if copy-constructor or operator()= is implemented.
    moose::MooseParser parser_;

};

#endif /* end of include guard: FUNCTIONH_ */
