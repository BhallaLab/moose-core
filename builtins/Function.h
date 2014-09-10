// Function.h --- 
// 
// Filename: Function.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Fri May 30 19:34:13 2014 (+0530)
// Version: 
// Last-Updated: 
//           By: 
//     Update #: 0
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// A new version of Func with FieldElements to collect data.
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

#ifndef _FUNCTION_H
#define _FUNCTION_H

#include "muParser.h"

/**
   Simple function parser and evaluator for MOOSE. This can take a mathematical
   expression in standard C form and a list of variables values and
   evaluate the results.
 */
static double *_functionAddVar(const char *name, void *data);

class Function
{
  public:
    static const int VARMAX;
    Function();
    Function(const Function& rhs);
    ~Function();
    void setExpr(string expr);
    string getExpr() const;
    
    
    // get a list of variable identifiers.
    // this is created by the parser
    vector<string> getVars() const;
    void setVarValues(vector< string > vars, vector < double > vals);

    
    // get/set the value of variable `name`
    void setVar(unsigned int index, double value);
    Variable * getVar(unsigned int ii);

    // get function eval result
    double getValue() const;

    // get/set operation mode
    void setMode(unsigned int mode);
    unsigned int getMode() const;

    void setNumVar(unsigned int num);
    unsigned int getNumVar() const;

    void setConst(string name, double value);
    double getConst(string name) const;

    void setIndependent(unsigned int index);
    unsigned int getIndependent() const;

    vector < double > getY() const;

    double getDerivative() const;

    Function& operator=(const Function rhs);

    unsigned int addVar();
    void dropVar(unsigned int msgLookup);

    void process(const Eref& e, ProcPtr p);
    void reinit(const Eref& e, ProcPtr p);

    static const Cinfo * initCinfo();

protected:
    friend double * _functionAddVar(const char * name, void *data);
    unsigned int _mode;
    mutable bool _valid;
     // this stores variables received via incoming messages, identifiers of the form x{i} are included in this
    vector<Variable *> _varbuf;
    unsigned int _numVar;
    // this stores variable values pulled by sending request. identifiers of the form y{i} are included in this
    vector< double * > _pullbuf;
    map< string, double *> _constbuf;  // for constants
    unsigned int _independent; // index of independent variable
    mu::Parser _parser;
    void _clearBuffer();
    void _showError(mu::Parser::exception_type &e) const;
};


#endif


// 
// Function.h ends here
