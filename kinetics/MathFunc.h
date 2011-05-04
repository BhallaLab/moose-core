/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
** Written by Raamesh Deshpande 2007
** Modified by Upi Bhalla 2010
**********************************************************************/
/*
#include <string>
#include <iostream>
#include <fstream>
// #include <ifstream>
#include <vector>
#include <map>
#include "assert.h"
#include "math.h"
#include <sstream>
*/
 
namespace MathFuncNames
{
	enum {NOTHING, FUNCTION, FUNCTION_CI, NUMBER, VARIABLE, EXPRESSION, EQ, 
		SIN, COS, TAN, ARCTAN, ARCSIN, ARCCOS, 
		POWER, SUM, TIMES, PLUS, MINUS, DIVIDE, SQRT, PRODUCT, APPLY, 
		CN, CI, CIF, CIV, CNI, DONTKNOW, APPLYOVER, CNOVER, CIOVER, 
		BVAR, BVAROVER, LOWLIMIT, UPLIMIT, LOWLIMITOVER, UPLIMITOVER, 
		VECTOR, SELECTOR, DONE, ERROR, MEAN, SDEV, VARIANCE, LPAREN, 
		RPAREN, MMLSTRING, BLANK, FNSTRING };
}

#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))

class MathFunc {
  public:

    MathFunc();
    void executeFunction();
    double getResult();
    void processFunc ( const Eref& e, ProcPtr info);
    void reinitFunc( const Eref& e, ProcPtr info );
//     void argFunc( double d );
    void arg1Func( double d );
    void arg2Func( double d );
    void arg3Func( double d );
    void arg4Func( double d );

    void setMathMl( string value );
    string getMathML() const;

    void setFunction( string fn );
    string getFunction() const ;

    double getR() const;

    void infixToPrefix();

	static const Cinfo* initCinfo();
  private:
    /*functions*/
    void evaluate(int pos, int arity);
    bool precedence(int op1, int op2);
    bool storeArgNames(string args);
    bool testStoreArgNames();
    void clear();
    void error(int lineno, string errormsg);
    void error(string errormsg);
    
    /*variables*/
    string mmlstring_;
    string fn_;
    vector <double> stack_;
    map <string, double *> symtable_;
    int expect_;
    vector <int> function_;
    vector <string> vname_;
    vector <double *> v_;
    string vector_name_;
    vector <double> v;
    double result_;
    map <int, int> precedence_;
    int status_; //MMLSTRING, FNSTRING, ERROR, BLANK
};

