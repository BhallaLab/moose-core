// Function.cpp --- 
// 
// Filename: Function.cpp
// Description: Implementation of a wrapper around GNU libmatheval to calculate arbitrary functions.
// Author: Subhasis Ray
// Maintainer: Subhasis Ray
// Created: Sat May 25 16:35:17 2013 (+0530)
// Version: 
// Last-Updated: Tue Jun 11 16:49:01 2013 (+0530)
//           By: subha
//     Update #: 619
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
// modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:

#include "header.h"
#include "../utility/utility.h"
#include "Function.h"

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

const Cinfo * Function::initCinfo()
{
    ////////////////////////////////////////////////////////////
    // Value fields    
    ////////////////////////////////////////////////////////////
    static  ReadOnlyValueFinfo< Function, double > value(
        "value",
        "Result of the function evaluation with current variable values.",
        &Function::getValue);
    static ReadOnlyValueFinfo< Function, double > derivative(
        "derivative",
        "Derivative of the function at given variable values.",
        &Function::getDerivative);
    static ValueFinfo< Function, unsigned int > mode(
        "mode",
        "Mode of operation: \n"
        " 1: only the function value will be funculated\n"
        " 2: only the derivative will be funculated\n"
        " 3: both function value and derivative at current variable values will be funculated.",
        &Function::setMode,
        &Function::getMode);
    static ValueFinfo< Function, string > expr(
        "expr",
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
        &Function::setExpr,
        &Function::getExpr);
    static FieldElementFinfo< Function, Variable > inputs(
        "x",
        "Input variables to the function. These can be passed via messages.",
        Variable< double >::initCinfo(),
        &Function::getVar,
        &Function::setNumVar,
        &Function::getNumVar);
    static LookupValueFinfo < Function, string, double > constants(
        "c",
        "Constants used in the function. These must be assigned before"
        " specifying the function expression.",
        &Function::setConst,
        &Function::getConst);

    static ValueFinfo< string > independent(
        "independent",
        "Independent variable. Differentiation is done based on this. Defaults"
        " to the first assigned variable.",
        &Function::setIndependent,
        &Function::getIndependent);
    // static ReadOnlyLookupValueFinfo< Function, string, double > vars(
    //     "var",
    //     "Variable names in the expression",
    //     &Function::getVars);
    ///////////////////////////////////////////////////////////////////
    // Shared messages
    ///////////////////////////////////////////////////////////////////
    static DestFinfo process( "process",
                              "Handles process call, updates internal time stamp.",
                              new ProcOpFunc< Func >( &Func::process ) );
    static DestFinfo reinit( "reinit",
                             "Handles reinit call.",
                             new ProcOpFunc< Func >( &Func::reinit ) );
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

    static Finfo *functionFinfos[] =
            {
                &value,
                &derivative,
                &mode,
                &expr,
                &inputs,
                &constants,
                // &vars,
                &proc,
                valueOut(),
                derivativeOut(),
            };
    
    static string doc[] =
            {
                "Name", "Function",
                "Author", "Subhasis Ray",
                "Description",
                "Function: general purpose function calculator using real numbers. It can "
                "parse mathematical expression defining a function and evaluate it "                
                "and/or its derivative for specified variable values. "
                "The variables can be input from other moose objects. In case of "
                "arbitrary variable names, the source message must have the variable "
                "name as the first argument. For most common cases, input messages to "
                "set x, y, z and xy, xyz are made available without such "
                "requirement. This class handles only real numbers "
                "(C-double). Predefined constants are: pi=3.141592..., "
                "e=2.718281...  "
            };
    
    static Dinfo< Function > dinfo;
    static Cinfo functionCinfo("Function",
                               Neutral::initCinfo(),
                               functionFinfos,
                               sizeof(functionFinfos) / sizeof(Finfo*),
                               &dinfo,
                               doc,
                               sizeof(doc)/sizeof(string));
    return &functionCinfo;
                                                    
}

static const Cinfo * functionCinfo = Function::initCinfo();

const int Function::VARMAX = 10;

Function::Function(): _mode(1), _valid(false)
{
    _varbuf.reserve(VARMAX);
    _parser.SetVarFactory(_addVar, this);
    // Adding pi and e, the defaults are `_pi` and `_e`
    _parser.DefineConst(_T("pi"), (mu::value_type)M_PI);
    _parser.DefineConst(_T("e"), (mu::value_type)M_E);
}

Function::Function(const Function& rhs): _mode(rhs._mode)
{
    _parser.SetVarFactory(_addVar, this);
    // Adding pi and e, the defaults are `_pi` and `_e`
    _parser.DefineConst(_T("pi"), (mu::value_type)M_PI);
    _parser.DefineConst(_T("e"), (mu::value_type)M_E);
    // Copy the constants
    mu::Parser::valmap_type cmap = rhs._parser.GetConst();
    if (cmap.size()){
        mu::Parser::valmap_type::const_iterator item = cmap.begin();
        for (; item!=cmap.end(); ++item){
            _parser.DefineConst(item->first, item->second);
        }
    }
    setExpr(rhs.getExpr());
    // Copy the values from the var pointers in rhs
    for (map< string, double >::iterator it = rhs._varbuf.begin();
         it != rhs._varbuf.end(); ++it){
        *_varbuf[it->first] = *it->second;
    }
}

Function& Function::operator=(const Function rhs)
{
    _clearBuffer();
    _mode = rhs._mode;
    // Adding pi and e, the defaults are `_pi` and `_e`
    _parser.DefineConst(_T("pi"), (mu::value_type)M_PI);
    _parser.DefineConst(_T("e"), (mu::value_type)M_E);
    setExpr(rhs.getExpr());
    // Copy the constants
    mu::Parser::valmap_type cmap = rhs._parser.GetConst();
    if (cmap.size()){
        mu::Parser::valmap_type::const_iterator item = cmap.begin();
        for (; item!=cmap.end(); ++item){
            _parser.DefineConst(item->first, item->second);
        }
    }
    // Copy the values from the var pointers in rhs
    for (map< string, double >::iterator it = rhs._varbuf.begin();
         it != rhs._varbuf.end(); ++it){
        *_varbuf[it->first] = *it->second;
    }
    return *this;
}

Function::~Function()
{
    _clearBuffer();
}

void Function::_clearBuffer()
{
    _parser.ClearVar();
    for (map< string, double >::iterator it = _varbuf.begin();
         it != _varbuf.end(); ++it){
        delete it->second;
    }
    _varbuf.clear();
}

void Function::_showError(mu::Parser::exception_type &e) const
{
    cout << "Error occurred in parser.\n" 
         << "Message:  " << e.GetMsg() << "\n"
         << "Formula:  " << e.GetExpr() << "\n"
         << "Token:    " << e.GetToken() << "\n"
         << "Position: " << e.GetPos() << "\n"
         << "Error code:     " << e.GetCode() << endl;
}
/**
   Call-back to add variables to parser automatically. 
 */
double * _addVar(const char *name, void *data)
{
    Function* function = reinterpret_cast< Function * >(data);
    map< string, double *>::iterator target = _varbuf.find(string(name));
    if (target == _varbuf.end()){
        double *ret = new double;
        *ret = 0.0;
        _varbuf.insert(pair<string, double*>(string(name), ret));
    } else {
        ret = target->second;
    }
    return ret;
}

void Function::setExpr(string expr)
{
    _valid = false;
    mu::varmap_type vars;
    try{
        _parser.SetExpr(expr);
    } catch (mu::Parser::exception_type &e) {
        _showError(e);
        _clearBuffer();
        return;
    }
    _valid = true;
}

string Function::getExpr() const
{
    if (!_valid){
        cout << "Error: Function::getExpr() - invalid parser state" << endl;
        return "";
    }
    return _parser.GetExpr();
}

void Function::setMode(unsigned int mode)
{
    _mode = mode;
}

unsigned int Function::getMode() const
{
    return _mode;
}

double Function::getValue() const
{
    double value = 0.0;
    if (!_valid){
        cout << "Error: Function::getValue() - invalid state" << endl;        
        return value;
    }
    try{
        value = _parser.Eval();
    } catch (mu::Parser::exception_type &e){
        _showError(e);
    }
    return value;
}

double Function::getDerivative() const
{
    double value = 0.0;    
    if (!_valid){
        cout << "Error: Function::getDerivative() - invalid state" << endl;        
        return value;
    }
    map<string, double *>::iterator it = _varbuf.find(_independent);
    if ((it != map::end) && (it->second != NULL)){
        try{
            value = _parser.Diff(it->second, *(it->second));
        } catch (mu::Parser::exception_type &e){
            _showError(e);
        }
    }
    return value;
}

void setNumVar(sunigned int num)
{
}

unsigned int getNumVar() const
{    
}

vector<string> Function::getVars() const
{
    vector< string > ret;
    if (!_valid){
        cout << "Error: Function::getVars() - invalid parser state" << endl;        
        return ret;
    }
    mu::varmap_type vars;
    try{
        vars = _parser.GetVar();
        for (mu::varmap_type::iterator ii = vars.begin();
             ii != vars.end(); ++ii){
            ret.push_back(ii->first);
        }
    } catch (mu::Parser::exception_type &e){
        _showError(e);
    }
    return ret;
}

void Function::setVarValues(vector<string> vars, vector<double> vals)
{
    
    if (vars.size() > vals.size() || !_valid){
        return;
    }
    mu::varmap_type varmap = _parser.GetVar();
    for (unsigned int ii = 0; ii < vars.size(); ++ii){
        mu::varmap_type::iterator v = varmap.find(vars[ii]);
        if ( v != varmap.end()){
            *v->second = vals[ii];
        }
    }
}

void Function::setConst(string name, double value)
{
    _parser.DefineConst(name, value);
}

double Function::getConst(string name) const
{
    mu::Parser::valmap_type cmap = _parser.GetConst();
    if (cmap.size()){
        mu::Parser::valmap_type::const_iterator it = cmap.find(name);
        if (it != cmap.end()){
            return it->second;
        }
    }
    return 0;
}

void Function::process(const Eref &e, ProcPtr p)
{
    if (!_valid){
        return;
    }
    if (_mode & 1){
        valueOut()->send(e, getValue());
    }
    if (_mode & 2){
        derivativeOut()->send(e, getDerivative());
    }
}

void Function::reinit(const Eref &e, ProcPtr p)
{
    if (!_valid){
        cout << "Error: Function::reinit() - invalid parser state. Will do nothing." << endl;
        return;
    }
    if (trim(_parser.GetExpr(), " \t\n\r").length() == 0){
        cout << "Error: no expression set. Will do nothing." << endl;
        setExpr("0.0");
        _valid = false;
    }
}
// 
// Function.cpp ends here
