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
    Function();
    ~Function();

    // Needs to copy the function.
    Function(const Function& rhs);
    Function& operator=(const Function& rhs);

    static const Cinfo * initCinfo();

    void innerSetExpr( const Eref& e, string expr);

    void setExpr( const Eref& e, string expr);
    string getExpr( const Eref& e ) const;

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

    double* addVariable(const char* name);


    void clearBuffer();
    void showError(moose::Parser::exception_type &e) const;


protected:
    // friend double * functionAddVar_(const char * name, void *data);
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
    vector<shared_ptr<Variable>> xs_;

    // this stores variable values pulled by sending request. identifiers of
    // the form y{i} are included in this
    vector<shared_ptr<double>> ys_;

    // parser. It is often copied.
    std::shared_ptr<moose::MooseParser> parser_;

    // Used by kinetic solvers when this is zombified.
    void* stoich_;

    // These variables may be redundant but used for interfacing with
    // MooseParser.
    map<string, double*> map_;
};

#endif /* end of include guard: FUNCTIONH_ */
