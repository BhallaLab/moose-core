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
double *_addVar(const char *name, void *data);

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
    void setVar(string name, double value);
    double getVar(string name) const;

    // get function eval result
    double getValue() const;

    // get/set operation mode
    void setMode(unsigned int mode);
    unsigned int getMode() const;

    void setX(double value);
    double getX() const;

    void setY(double value);
    double getY() const;

    void setZ(double value);
    double getZ() const;

    void setXY(double x, double y);
    void setXYZ(double x, double y, double z);

    double getDerivative() const;

    Function& operator=(const Func rhs);

    void process(const Eref& e, ProcPtr p);
    void reinit(const Eref& e, ProcPtr p);

    static const Cinfo * initCinfo();

protected:
    friend double * _addVar(const char * name, void *data);
    map< string, double *> _varbuf; // for variables
    map< string, double *> _constbuf;  // for constants
    string _independent; // name of independent variable
    mu::Parser _parser;
    double *_x, *_y, *_z;
    unsigned int _mode;
    mutable bool _valid;
    void _clearBuffer();
    void _showError(mu::Parser::exception_type &e) const;
};


#endif


// 
// Function.h ends here
