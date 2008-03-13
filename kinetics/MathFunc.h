#include <string>
#include <iostream>
#include <fstream>
// #include <ifstream>
#include <vector>
#include <map>
#include "assert.h"
#include "math.h"
#include <sstream>
 
enum {NOTHING, FUNCTION, FUNCTION_CI, NUMBER, VARIABLE, EXPRESSION, EQ, SIN, COS, TAN, ARCTAN, ARCSIN, ARCCOS, POWER, SUM, TIMES, PLUS, MINUS, DIVIDE, SQRT, PRODUCT, APPLY, CN, CI, CIF, CIV, CNI, DONTKNOW, APPLYOVER, CNOVER, CIOVER, BVAR, BVAROVER, LOWLIMIT, UPLIMIT, LOWLIMITOVER, UPLIMITOVER, VECTOR, SELECTOR, DONE, ERROR, MEAN, SDEV, VARIANCE, LPAREN, RPAREN, MMLSTRING, BLANK, FNSTRING };

using namespace std;

#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))

class MathFunc{
  public:
    MathFunc();
    void executeFunction();
    double getResult();
    static void processFunc( const Conn* c, ProcInfo info );
    void process ( Eref e, ProcInfo info);
    static void reinitFunc( const Conn* c, ProcInfo info );
    static void argFunc(const Conn* c, double d);
    static void arg1Func(const Conn* c, double d);
    static void arg2Func(const Conn* c, double d);
    static void arg3Func(const Conn* c, double d);
    static void arg4Func(const Conn* c, double d);
    static void setMathMl( const Conn* c, string value );
    static string getMathML( Eref e );
    static void setFunction(const Conn* c, string fn);
    static string getFunction( Eref e );
    static double getR( Eref e );
    static void setR(const Conn* c, double ss);
    void innerSetMMLString(string value);
    void innerSetFunctionString(string value);
    void reinitFuncLocal( Eref e );
    void infixToPrefix();
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
    vector <double *> v;
    double result_;
    map <int, int> precedence_;
    int status_; //MMLSTRING, FNSTRING, ERROR, BLANK
};

