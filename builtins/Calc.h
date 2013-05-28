/* Calc.h --- 
 * 
 * Filename: Calc.h
 * Description: A simple calculator class for MOOSE. 
 * Author: Subhasis Ray
 * Maintainer: 
 * Created: Sat May 25 16:14:13 2013 (+0530)
 * Version: 
 * Last-Updated: Tue May 28 12:12:14 2013 (+0530)
 *           By: subha
 *     Update #: 111
 * URL: 
 * Keywords: 
 * Compatibility: 
 * 
 */

/* Commentary: 
 * 
 * 
 * 
 */

/* Change log:
 * 
 * 
 */

/* This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 3, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street, Fifth
 * Floor, Boston, MA 02110-1301, USA.
 */

/* Code: */

#ifndef _CALC_H
#define _CALC_H

#include "muParser.h"
/**
   Simple calculator for MOOSE. This can take a mathematical
   expression in standard C form and a list of variables values and
   evaluate the results.
 */
double *_addVar(const char *name, void *data);

class Calc
{
  public:
    static const int VARMAX;
    Calc();
    ~Calc();
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

    void process(const Eref& e, ProcPtr p);
    void reinit(const Eref& e, ProcPtr p);

    static const Cinfo * initCinfo();

protected:
    friend double * _addVar(const char * name, void *data);
    vector< double > _varbuf;
    mu::Parser _parser;
    double *_x, *_y, *_z;
    unsigned int _mode;
    mutable bool _valid;
};
#endif



/* Calc.h ends here */
