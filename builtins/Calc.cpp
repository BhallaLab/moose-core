// Calc.cpp --- 
// 
// Filename: Calc.cpp
// Description: Implementation of a wrapper around GNU libmatheval to calculate arbitrary functions.
// Author: Subhasis Ray
// Maintainer: Subhasis Ray
// Created: Sat May 25 16:35:17 2013 (+0530)
// Version: 
// Last-Updated: Tue May 28 12:38:18 2013 (+0530)
//           By: subha
//     Update #: 532
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change log:
// 
// 
// 
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:

#include "header.h"
#include "../utility/utility.h"
#include "Calc.h"

static SrcFinfo1<double> *valueOut()
{
    static SrcFinfo1<double> valueOut("valueOut",
                                      "Evaluated value of the function for the current variable values.");
    return &valueOut;
}

static SrcFinfo1< double > *derivativeOut()
{
    static SrcFinfo1< double > derivativeOut("derivativeOut",
                                             "Value of derivative of the function for the current variable values");
    return &derivativeOut;
}

const Cinfo * Calc::initCinfo()
{
    ////////////////////////////////////////////////////////////
    // Value fields    
    ////////////////////////////////////////////////////////////
    static  ReadOnlyValueFinfo< Calc, double > value("value",
                                                     "Result of the function evaluation with current variable values.",
                                                     &Calc::getValue);
    static ReadOnlyValueFinfo< Calc, double > derivative("derivative",
                                                         "Derivative of the function at given variable values.",
                                                         &Calc::getDerivative);
    static ValueFinfo< Calc, unsigned int > mode("mode",
                                                 "Mode of operation: \n"
                                                 " 1: only the function value will be calculated\n"
                                                 " 2: only the derivative will be calculated\n"
                                                 " 3: both function value and derivative at current variable values will be calculated.",
                                                 &Calc::setMode,
                                                 &Calc::getMode);
    static ValueFinfo< Calc, string > expr("expr",
                                           "Mathematical expression defining the function. The underlying parser\n"
                                           "is muParser. Hence the available functions and operators are (from\n"
                                           "muParser docs):\n"
                                           "\nFunctions\n"
                                           "Name        args    explanation\n"
                                           "sin         1       sine function\n"
                                           "cos         1       cosine function\n"
                                           "tan         1       tangens function\n"
                                           "asin        1       arcus sine function\n"
                                           "acos        1       arcus cosine function\n"
                                           "atan        1       arcus tangens function\n"
                                           "sinh        1       hyperbolic sine function\n"
                                           "cosh        1       hyperbolic cosine\n"
                                           "tanh        1       hyperbolic tangens function\n"
                                           "asinh       1       hyperbolic arcus sine function\n"
                                           "acosh       1       hyperbolic arcus tangens function\n"
                                           "atanh       1       hyperbolic arcur tangens function\n"
                                           "log2        1       logarithm to the base 2\n"
                                           "log10       1       logarithm to the base 10\n"
                                           "log         1       logarithm to the base 10\n"
                                           "ln  1       logarithm to base e (2.71828...)\n"
                                           "exp         1       e raised to the power of x\n"
                                           "sqrt        1       square root of a value\n"
                                           "sign        1       sign function -1 if x<0; 1 if x>0\n"
                                           "rint        1       round to nearest integer\n"
                                           "abs         1       absolute value\n"
                                           "min         var.    min of all arguments\n"
                                           "max         var.    max of all arguments\n"
                                           "sum         var.    sum of all arguments\n"
                                           "avg         var.    mean value of all arguments\n"
                                           "\nOperators\n"
                                           "Op  meaning         prioroty\n"
                                           "=   assignement     -1\n"
                                           "&&  logical and     1\n"
                                           "||  logical or      2\n"
                                           "<=  less or equal   4\n"
                                           ">=  greater or equal        4\n"
                                           "!=  not equal       4\n"
                                           "==  equal   4\n"
                                           ">   greater than    4\n"
                                           "<   less than       4\n"
                                           "+   addition        5\n"
                                           "-   subtraction     5\n"
                                           "*   multiplication  6\n"
                                           "/   division        6\n"
                                           "^   raise x to the power of y       7\n"
                                           "\n"
                                           "?:  if then else operator   C++ style syntax\n",
                                           &Calc::setExpr,
                                           &Calc::getExpr);
    static LookupValueFinfo < Calc, string, double > var("var",
                                                         "Lookup table for variable values.",
                                                         &Calc::setVar,
                                                         &Calc::getVar);
    static ReadOnlyValueFinfo< Calc, vector<string> > vars("vars",
                                                           "Variable names in the expression",
                                                           &Calc::getVars);
    static ValueFinfo< Calc, double > x("x",
                                        "Value for variable named x. This is a shorthand. If the\n"
                                        "expression does not have any variable named x, this the first variable\n"
                                        "in the sequence `vars`.",
                                        &Calc::setX,
                                        &Calc::getX);
    static ValueFinfo< Calc, double > y("y",
                                        "Value for variable named y. This is a utility for two/three\n"
                                        " variable functions where the y value comes from a source separate\n"
                                        " from that of x. This is a shorthand. If the\n"
                                        "expression does not have any variable named y, this the second\n"
                                        "variable in the sequence `vars`.",
                                        &Calc::setY,
                                        &Calc::getY);
    static ValueFinfo< Calc, double > z("z",
                                        "Value for variable named z. This is a utility for three\n"
                                        " variable functions where the z value comes from a source separate\n"
                                        " from that of x or z. This is a shorthand. If the expression does not\n"
                                        " have any variable named z, this the third variable in the sequence `vars`.",
                                        &Calc::setZ,
                                        &Calc::getZ);
    ////////////////////////////////////////////////////////////
    // DestFinfos
    ////////////////////////////////////////////////////////////
    static DestFinfo varIn("varIn",
                           "Handle value for specified variable coming from other objects",
                           new OpFunc2< Calc, string, double > (&Calc::setVar));
    static DestFinfo xIn("xIn",
                         "Handle value for variable named x. This is a shorthand. If the\n"
                         "expression does not have any variable named x, this the first variable\n"
                         "in the sequence `vars`.",
                         new OpFunc1< Calc, double > (&Calc::setX));
    static DestFinfo yIn("yIn",
                         "Handle value for variable named y. This is a utility for two/three\n"
                         " variable functions where the y value comes from a source separate\n"
                         " from that of x. This is a shorthand. If the\n"
                         "expression does not have any variable named y, this the second\n"
                         "variable in the sequence `vars`.",
                         new OpFunc1< Calc, double > (&Calc::setY));
    static DestFinfo zIn("zIn",
                         "Handle value for variable named z. This is a utility for three\n"
                         " variable functions where the z value comes from a source separate\n"
                         " from that of x or y. This is a shorthand. If the expression does not\n"
                         " have any variable named y, this the second variable in the sequence `vars`.",
                         new OpFunc1< Calc, double > (&Calc::setZ));
    static DestFinfo xyIn("xyIn",
                          "Handle value for variables x and y for two-variable function",
                          new OpFunc2< Calc, double, double > (&Calc::setXY));
    static DestFinfo xyzIn("xyzIn",
                           "Handle value for variables x, y and z for three-variable function",
                           new OpFunc3< Calc, double, double, double > (&Calc::setXYZ));

    static DestFinfo setVars("setVars",
                             "Utility function to assign the variable values of the function.\n"
                             "Takes a list of variable names and a list of corresponding values.",
                             new OpFunc2< Calc, vector< string >, vector< double > > (&Calc::setVarValues));
    
    // TODO - a way to allow connect a source to a specific variable without the source knowing the variable name
    // simple case of x, [y, [z]] variables

    ///////////////////////////////////////////////////////////////////
    // Shared messages
    ///////////////////////////////////////////////////////////////////
    static DestFinfo process( "process",
                              "Handles process call, updates internal time stamp.",
                              new ProcOpFunc< Calc >( &Calc::process ) );
    static DestFinfo reinit( "reinit",
                             "Handles reinit call.",
                             new ProcOpFunc< Calc >( &Calc::reinit ) );
    static Finfo* processShared[] =
            {
		&process, &reinit
            };
    
    static SharedFinfo proc( "proc",
                             "This is a shared message to receive Process messages "
                             "from the scheduler objects."
                             "The first entry in the shared msg is a MsgDest "
                             "for the Process operation. It has a single argument, "
                             "ProcInfo, which holds lots of information about current "
                             "time, thread, dt and so on. The second entry is a MsgDest "
                             "for the Reinit operation. It also uses ProcInfo. ",
                             processShared, sizeof( processShared ) / sizeof( Finfo* )
                             );

    static Finfo *calcFinfos[] =
            {
                &value,
                &derivative,
                &mode,
                &expr,
                &var,
                &vars,
                &x,
                &y,
                &z,
                &varIn,
                &xIn,
                &yIn,
                &zIn,
                &xyIn,
                &xyzIn,
                &proc,
                valueOut(),
                derivativeOut(),
            };
    
    static string doc[] =
            {
                "Name", "Calc",
                "Author", "Subhasis Ray",
                "Description",
                "Calc: general purpose function calculator using real numbers. It can\n"
                "parse mathematical expression defining a function and evaluate it\n"                
                "and/or its derivative for specified variable values.\n"
                "The variables can be input from other moose objects. In case of\n"
                "arbitrary variable names, the source message must have the variable\n"
                "name as the first argument. For most common cases, input messages to\n"
                "set x, y, z and xy, xyz are made available without such\n"
                "requirement. This class handles only real numbers\n"
                "(C-double). Predefined constants are: pi=3.141592...,\n"
                "e=2.718281... \n"
            };
    
    static Cinfo calcCinfo("Calc",
                            Neutral::initCinfo(),
                            calcFinfos,
                            sizeof(calcFinfos) / sizeof(Finfo*),
                            new Dinfo<Calc>(),
                            doc,
                            sizeof(doc)/sizeof(string));
    return &calcCinfo;
                                                    
}

static const Cinfo * calcCinfo = Calc::initCinfo();

const int Calc::VARMAX = 10;

Calc::Calc():_x(NULL), _y(NULL), _z(NULL), _mode(1), _valid(false)
{
    _varbuf.reserve(VARMAX+1);
    _parser.SetVarFactory(_addVar, this);
    // Adding pi and e, the defaults are `_pi` and `_e`
    _parser.DefineConst(_T("pi"), (mu::value_type)M_PI);
    _parser.DefineConst(_T("e"), (mu::value_type)M_E);
}

Calc::~Calc()
{
    _parser.ClearConst();
    _parser.ClearVar();
}
/**
   Call-back to add variables to parser automatically. 
 */
double * _addVar(const char *name, void *data)
{
    Calc* calc = reinterpret_cast< Calc * >(data);
    calc->_varbuf.push_back(0.0);
    if (calc->_varbuf.size() >= calc->_varbuf.capacity()){
        calc->_valid = false;
        throw mu::Parser::exception_type("Variable buffer overflow.");
    } 
    double *ret = &(*calc->_varbuf.rbegin());
    return ret;
}

void Calc::setExpr(string expr)
{
    _varbuf.clear();
    _x = NULL;
    _y = NULL;
    _z = NULL;
    try{
        _parser.SetExpr(expr);
        _valid = true;
    } catch (mu::Parser::exception_type &e) {
        cout << "Error setting expression" << "\n"
             << "Message:  " << e.GetMsg() << "\n"
             << "Formula:  " << e.GetExpr() << "\n"
             << "Token:    " << e.GetToken() << "\n"
             << "Position: " << e.GetPos() << "\n"
             << "Error code:     " << e.GetCode() << endl;
        _valid = false;
        _varbuf.clear();
        _parser.SetExpr("0.0");
        return;
    }
    mu::varmap_type vars;
    try{
        vars = _parser.GetUsedVar();
    } catch (mu::Parser::exception_type &e) {
        cout << "Message:  " << e.GetMsg() << "\n"
             << "Formula:  " << e.GetExpr() << "\n"
             << "Token:    " << e.GetToken() << "\n"
             << "Position: " << e.GetPos() << "\n"
             << "Error code:     " << e.GetCode() << endl;
        _valid = false;
        _varbuf.clear();
        _parser.SetExpr("0.0");
        return;
    }
    mu::varmap_type::iterator v = vars.find("x");
    if (v != vars.end()){
        _x = v->second;
    } else if (vars.size() >= 1){
        v = vars.begin();
        _x = v->second;
    }
    v = vars.find("y");
    if (v != vars.end()){
        _y = v->second;
    } else if (vars.size() >= 2){
        v = vars.begin();
        ++v;
        _y = v->second;
    }
    v = vars.find("z");
    if (v != vars.end()){
        _z = v->second;
    } else if (vars.size() >= 3){
        v = vars.begin();
        v++; v++;
        _z = v->second;
    }
}

string Calc::getExpr() const
{
    if (!_valid){
        cout << "Error: Calc::getExpr() - invalid parser state" << endl;
        return "";
    }
    return _parser.GetExpr();
}

/**
   Set value of variable `name`
*/
void Calc::setVar(string name, double value)
{
    if (!_valid){
        cout << "Error: Calc::setVar() - invalid parser state" << endl;
        return;
    }
    mu::varmap_type vars;
    try{
        vars = _parser.GetUsedVar();
        _valid = true;
    } catch (mu::Parser::exception_type &e) {
        _valid = false;
        cout << "Message:  " << e.GetMsg() << "\n";
        cout << "Formula:  " << e.GetExpr() << "\n";
        cout << "Token:    " << e.GetToken() << "\n";
        cout << "Position: " << e.GetPos() << "\n";
        cout << "Error code:     " << e.GetCode() << "\n";
        _parser.SetExpr("0.0");
        _varbuf.clear();
        return;
    }
    mu::varmap_type::iterator v = vars.find(name);
    if (v != vars.end()){
        *v->second = value;
    } else {
        cout << "Error: no such variable " << name << endl;
    }
}

/**
   Get value of variable `name`
*/
double Calc::getVar(string name) const
{
    if (!_valid){
        cout << "Error: Calc::getVar() - invalid parser state" << endl;
        return 0.0;
    }
    try{
        const mu::varmap_type &vars = _parser.GetUsedVar();
        mu::varmap_type::const_iterator v = vars.find(name);
        if (v != vars.end()){
            return *v->second;
        } else {
            cout << "Error: no such variable " << name << endl;
            return 0.0;
        }
    } catch (mu::Parser::exception_type &e) {
        _valid = false;
        cout << "Message:  " << e.GetMsg() << "\n"
             << "Formula:  " << e.GetExpr() << "\n"
             << "Token:    " << e.GetToken() << "\n"
             << "Position: " << e.GetPos() << "\n"
             << "Error code:     " << e.GetCode() << endl;
        _valid = false;
        return 0.0;
    }
}

void Calc::setX(double x) 
{
    if (_x != NULL){
        *_x = x;
    }
}

double Calc::getX() const
{
    if (_x != NULL){
        return *_x;
    }
    return 0.0;
}

void Calc::setY(double y) 
{
    if (_y != NULL){
        *_y = y;
    }
}

double Calc::getY() const
{
    if (_y != NULL){
        return *_y;
    }
    return 0.0;
}
void Calc::setZ(double z) 
{
    if (_z != NULL){
        *_z = z;
    }
}

double Calc::getZ() const
{
    if (_z != NULL){
        return *_z;
    }
    return 0.0;
}

void Calc::setXY(double x, double y) 
{
    if (_x != NULL){
        *_x = x;
    }
    if (_y != NULL){
        *_y = y;
    }
}

void Calc::setXYZ(double x, double y, double z) 
{
    if (_x != NULL){
        *_x = x;
    }
    if (_y != NULL){
        *_y = y;
    }
    if (_z != NULL){
        *_z = z;
    }
}                           

void Calc::setMode(unsigned int mode)
{
    _mode = mode;
}

unsigned int Calc::getMode() const
{
    return _mode;
}

double Calc::getValue() const
{
    double value = 0.0;
    if (!_valid){
        cout << "Error: Calc::getValue() - invalid state" << endl;        
        return value;
    }
    try{
        value = _parser.Eval();
    } catch (mu::Parser::exception_type &e){
        cout << "Error evaluating function\n"
             << "Message:  " << e.GetMsg() << "\n"
             << "Formula:  " << e.GetExpr() << "\n"
             << "Token:    " << e.GetToken() << "\n"
             << "Position: " << e.GetPos() << "\n"
             << "Error code:     " << e.GetCode() << endl;

    }
    return value;
}

double Calc::getDerivative() const
{
    double value = 0.0;    
    if (!_valid){
        cout << "Error: Calc::getDerivative() - invalid state" << endl;        
        return value;
    }
    if (_x != NULL){
        try{
            value = _parser.Diff(_x, *_x);
        } catch (mu::Parser::exception_type &e){
            _valid = false;
            cout << "Error evaluating derivative\n"
                 << "Message:  " << e.GetMsg() << "\n"
                 << "Formula:  " << e.GetExpr() << "\n"
                 << "Token:    " << e.GetToken() << "\n"
                 << "Position: " << e.GetPos() << "\n"
                 << "Error code:     " << e.GetCode() << endl;
        }
    }
    return value;
}


vector<string> Calc::getVars() const
{
    vector< string > ret;
    if (!_valid){
        cout << "Error: Calc::getVars() - invalid parser state" << endl;        
        return ret;
    }
    mu::varmap_type vars;
    try{
        vars = _parser.GetUsedVar();
    } catch (mu::Parser::exception_type &e){
        _valid = false;
        cout << "Error getting variables\n"
             << "Message:  " << e.GetMsg() << "\n"
             << "Formula:  " << e.GetExpr() << "\n"
             << "Token:    " << e.GetToken() << "\n"
             << "Position: " << e.GetPos() << "\n"
             << "Error code:     " << e.GetCode() << endl;
    }
    for (mu::varmap_type::iterator ii = vars.begin();
         ii != vars.end(); ++ii){
        ret.push_back(ii->first);
    }
    return ret;                   
}

void Calc::setVarValues(vector<string> vars, vector<double> vals)
{
    
    if (vars.size() > vals.size() || !_valid){
        return;
    }
    mu::varmap_type varmap = _parser.GetUsedVar();
    for (unsigned int ii = 0; ii < vars.size(); ++ii){
        mu::varmap_type::iterator v = varmap.find(vars[ii]);
        if ( v != varmap.end()){
            *v->second = vals[ii];
        }
    }
}

void Calc::process(const Eref &e, ProcPtr p)
{
    if (!_valid){
        return;
    }
    if (_mode & 1){
        valueOut()->send(e, p->threadIndexInGroup, getValue());
    }
    if (_mode & 2){
        derivativeOut()->send(e, p->threadIndexInGroup, getDerivative());
    }
}

void Calc::reinit(const Eref &e, ProcPtr p)
{
    if (!_valid){
        cout << "Error: Calc::reinit() - invalid parser state. Will do nothing." << endl;
        return;
    }
    if (trim(_parser.GetExpr(), " \t\n\r").length() == 0){
        cout << "Error: no expression set. Will do nothing." << endl;
        _parser.SetExpr("0.0");
        _valid = false;
    }
}
// 
// Calc.cpp ends here
