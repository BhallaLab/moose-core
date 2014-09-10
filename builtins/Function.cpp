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
#include "Variable.h"

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

static SrcFinfo1< vector < double > *> *requestOut()
{
    static SrcFinfo1< vector < double > * > requestOut(
        "requestOut",
        "Sends request for input variable from a field on target object");
    return &requestOut;
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

    static ValueFinfo< Function, unsigned int > numVars(
        "numVars",
        "Number of variables used by Function.",
        &Function::setNumVar,
        &Function::getNumVar);
    
    static FieldElementFinfo< Function, Variable > inputs(
        "x",
        "Input variables to the function. These can be passed via messages.",
        Variable::initCinfo(),
        &Function::getVar,
        &Function::setNumVar,
        &Function::getNumVar);

    static LookupValueFinfo < Function, string, double > constants(
        "c",
        "Constants used in the function. These must be assigned before"
        " specifying the function expression.",
        &Function::setConst,
        &Function::getConst);
    
    static ReadOnlyValueFinfo< Function, vector < double > > y(
        "y",
        "Variable values received from target fields by requestOut",
        &Function::getY);

    static ValueFinfo< Function, unsigned int > independent(
        "independent",
        "Index of independent variable. Differentiation is done based on this. Defaults"
        " to the first assigned variable.",
        &Function::setIndependent,
        &Function::getIndependent);

    ///////////////////////////////////////////////////////////////////
    // Shared messages
    ///////////////////////////////////////////////////////////////////
    static DestFinfo process( "process",
                              "Handles process call, updates internal time stamp.",
                              new ProcOpFunc< Function >( &Function::process ) );
    static DestFinfo reinit( "reinit",
                             "Handles reinit call.",
                             new ProcOpFunc< Function >( &Function::reinit ) );
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
                &independent,
                &proc,
                requestOut(),
                valueOut(),
                derivativeOut(),
            };
    
    static string doc[] =
            {
                "Name", "Function",
                "Author", "Subhasis Ray",
                "Description",
                "Function: general purpose function calculator using real numbers. It can"
                " parse mathematical expression defining a function and evaluate it"                
                " and/or its derivative for specified variable values."                
                " The variables can be input from other moose objects."
                " Such variables must be named `x{i}` in the expression and the source"
                " field is connected to Function.x[i]'s setVar destination field."
                " In case the input variable is not available as a source field, but is"
                " a value field, then the value can be requested by connecting the"
                " `requestOut` message to the `get{Field}` destination on the target"
                " object. Such variables must be specified in the expression as y{i}"
                " and connecting the messages should happen in the same order as the"
                " y indices." 
                " This class handles only real numbers (C-double). Predefined constants"
                " are: pi=3.141592..., e=2.718281..."
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

Function::Function(): _mode(1), _valid(false), _numVar(0)
{
    _parser.SetVarFactory(_functionAddVar, this);
    // Adding pi and e, the defaults are `_pi` and `_e`
    _parser.DefineConst(_T("pi"), (mu::value_type)M_PI);
    _parser.DefineConst(_T("e"), (mu::value_type)M_E);
}

Function::Function(const Function& rhs): _mode(rhs._mode), _numVar(rhs._numVar)
{
    _parser.SetVarFactory(_functionAddVar, this);
    // Adding pi and e, the defaults are `_pi` and `_e`
    _parser.DefineConst(_T("pi"), (mu::value_type)M_PI);
    _parser.DefineConst(_T("e"), (mu::value_type)M_E);
    // Copy the constants
    mu::valmap_type cmap = rhs._parser.GetConst();
    if (cmap.size()){
        mu::valmap_type::const_iterator item = cmap.begin();
        for (; item!=cmap.end(); ++item){
            _parser.DefineConst(item->first, item->second);
        }
    }
    setExpr(rhs.getExpr());
    // Copy the values from the var pointers in rhs
    assert(_varbuf.size() == rhs._varbuf.size());
    for (unsigned int ii = 0; ii < rhs._varbuf.size(); ++ii){
        _varbuf[ii]->value = rhs._varbuf[ii]->value;
    }
    assert(_pullbuf.size() == rhs._pullbuf.size());
    for (unsigned int ii = 0; ii < rhs._pullbuf.size(); ++ii){
        *_pullbuf[ii] = *(rhs._pullbuf[ii]);
    }
}

Function& Function::operator=(const Function rhs)
{
    _clearBuffer();
    _mode = rhs._mode;
    // Adding pi and e, the defaults are `_pi` and `_e`
    _parser.DefineConst(_T("pi"), (mu::value_type)M_PI);
    _parser.DefineConst(_T("e"), (mu::value_type)M_E);
    // Copy the constants
    mu::valmap_type cmap = rhs._parser.GetConst();
    if (cmap.size()){
        mu::valmap_type::const_iterator item = cmap.begin();
        for (; item!=cmap.end(); ++item){
            _parser.DefineConst(item->first, item->second);
        }
    }
    // Copy the values from the var pointers in rhs
    setExpr(rhs.getExpr());
    assert(_varbuf.size() == rhs._varbuf.size());
    for (unsigned int ii = 0; ii < rhs._varbuf.size(); ++ii){
        _varbuf[ii]->value = rhs._varbuf[ii]->value;
    }
    assert(_pullbuf.size() == rhs._pullbuf.size());
    for (unsigned int ii = 0; ii < rhs._pullbuf.size(); ++ii){
        *_pullbuf[ii] = *(rhs._pullbuf[ii]);
    }
    return *this;
}

Function::~Function()
{
 //    _clearBuffer();
 }

// do not know what to do about Variables that have already been
// connected by message.
void Function::_clearBuffer()
{
    _numVar = 0;
    _parser.ClearVar();
    for (unsigned int ii = 0; ii < _varbuf.size(); ++ii){
        delete _varbuf[ii]; 
    }
    _varbuf.clear();
    for (unsigned int ii = 0; ii < _pullbuf.size(); ++ii){
        delete _pullbuf[ii]; 
    }
    _pullbuf.clear();
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

   We use different storage for constants and variables. Variables are
   stored in a vector of Variable object pointers. They must have the
   name x{index} where index is any non-negative integer. Note that
   the vector is resized to whatever the maximum index is. It is up to
   the user to make sure that indices are sequential without any
   gap. In case there is a gap in indices, those entries will remain
   unused.

   If the name starts with anything other than `x`, then it is taken
   to be a named constant, which must be set before any expression or
   variables and error is thrown.
 */
static double * _functionAddVar(const char *name, void *data)
{
    Function* function = reinterpret_cast< Function * >(data);
    double * ret = NULL;
    string strname(name);
    // Names starting with x are variables, everything else is constant.
    if (strname[0] == 'x'){
        int index = atoi(strname.substr(1).c_str());
        if (index >= function->_varbuf.size()){
            function->_varbuf.resize(index+1);
            function->_varbuf[index] = new Variable();
        }
        ret = &(function->_varbuf[index]->value);        
    } else if (strname[0] == 'y'){
        int index = atoi(strname.substr(1).c_str());
        if (index >= function->_pullbuf.size()){
            function->_pullbuf.resize(index+1);
            function->_pullbuf[index] = new double();
        }
        ret = function->_pullbuf[index];        
    } else {
        cerr << "Got an undefined constant " << name << endl
             << "You must define the constants beforehand using LookupField c: c[name]"
                " = value"
             << endl;
        throw mu::ParserError("Undefined constant.");
    }
    return ret;
}

/**
   This function is for automatically adding variables when setVar
   messages are connected.

   There are two ways new variables can be created :

   (1) When the user sets the expression, muParser calls _functionAddVar to
   get the address of the storage for each variable name in the
   expression.

   (2) When the user connects a setVar message to a Variable. ??

   
 */
unsigned int Function::addVar()
{
    unsigned int newVarIndex = _numVar;
    ++_numVar;
    stringstream name;
    name << "x" << newVarIndex;
    _functionAddVar(name.str().c_str(), this);
    return newVarIndex;
}

void Function::dropVar(unsigned int msgLookup)
{
    // Don't know what this can possibly mean in the context of
    // evaluating a set expression.
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

void Function::setIndependent(unsigned int index)
{
    _independent = index;
}

unsigned int Function::getIndependent() const
{
    return _independent;
}

vector< double > Function::getY() const
{
    vector < double > ret(_pullbuf.size());
    for (unsigned int ii = 0; ii < ret.size(); ++ii){
        ret[ii] = *_pullbuf[ii];
    }
    return ret;
}

double Function::getDerivative() const
{
    double value = 0.0;    
    if (!_valid){
        cout << "Error: Function::getDerivative() - invalid state" << endl;        
        return value;
    }
    assert(_independent < _varbuf.size());
    try{
        value = _parser.Diff(&(_varbuf[_independent]->value),
                             _varbuf[_independent]->value);
    } catch (mu::Parser::exception_type &e){
            _showError(e);
    }    
    return value;
}

void Function::setNumVar(const unsigned int num)
{
    _clearBuffer();
    for (unsigned int ii = 0; ii < num; ++ii){
        stringstream name;
        name << "x" << ii;
        _functionAddVar(name.str().c_str(), this);
    }
}

unsigned int Function::getNumVar() const
{
    return _varbuf.size();
}

void Function::setVar(unsigned int index, double value)
{
    assert(index < _varbuf.size());
    _varbuf[index]->setValue(value);
}

Variable * Function::getVar(unsigned int ii)
{
    static Variable dummy;
    if ( ii < _varbuf.size()){
        return _varbuf[ii];
    }
    cout << "Warning: Function::getVar: index: "
         << ii << " is out of range: "
         << _varbuf.size() << endl;
    return &dummy;
}

void Function::setConst(string name, double value)
{
    _parser.DefineConst(name, value);
}

double Function::getConst(string name) const
{
    mu::valmap_type cmap = _parser.GetConst();
    if (cmap.size()){
        mu::valmap_type::const_iterator it = cmap.find(name);
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
    vector < double > databuf;
    requestOut()->send(e, &databuf);
    for (unsigned int ii = 0;
         (ii < databuf.size()) && (ii < _pullbuf.size());
         ++ii){
        *_pullbuf[ii] = databuf[ii];        
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
