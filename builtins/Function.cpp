/***
 *    Description:  Wrapper around MooseParser.
 *         Author:  Subhasis Ray
 *     Maintainer:  Dilawar Singh <dilawars@ncbs.res.in>
 */

#include "../basecode/header.h"
#include "../basecode/global.h"
#include "../utility/utility.h"
#include "../utility/numutil.h"
#include "../utility/print_function.hpp"
#include "../builtins/MooseParser.h"
#include "Variable.h"

#include "Function.h"
#include "ElementValueFinfo.h"

static const double TriggerThreshold = 0.0;

static SrcFinfo1<double> *valueOut()
{
    static SrcFinfo1<double> valueOut("valueOut",
            "Evaluated value of the function for the current variable values."
            );
    return &valueOut;
}

static SrcFinfo1< double > *derivativeOut()
{
    static SrcFinfo1< double > derivativeOut("derivativeOut",
            "Value of derivative of the function for the current variable values"
            );
    return &derivativeOut;
}

static SrcFinfo1< double > *rateOut()
{
    static SrcFinfo1< double > rateOut("rateOut",
            "Value of time-derivative of the function for the current variable values"
            );
    return &rateOut;
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
    // Value fields
    static  ReadOnlyValueFinfo< Function, double > value(
        "value",
        "Result of the function evaluation with current variable values.",
        &Function::getValue
    );

    static ReadOnlyValueFinfo< Function, double > derivative(
        "derivative",
        "Derivative of the function at given variable values. This is calulated"
        " using 5-point stencil "
        " <http://en.wikipedia.org/wiki/Five-point_stencil> at current value of"
        " independent variable. Note that unlike hand-calculated derivatives,"
        " numerical derivatives are not exact.",
        &Function::getDerivative
    );

    static ReadOnlyValueFinfo< Function, double > rate(
        "rate",
        "Derivative of the function at given variable values. This is computed"
        " as the difference of the current and previous value of the function"
        " divided by the time step.",
        &Function::getRate
    );

    static ValueFinfo< Function, unsigned int > mode(
        "mode",
        "Mode of operation: \n"
        " 1: only the function value will be sent out.\n"
        " 2: only the derivative with respect to the independent variable will be sent out.\n"
        " 3: only rate (time derivative) will be sent out.\n"
        " anything else: all three, value, derivative and rate will be sent out.\n",
        &Function::setMode,
        &Function::getMode
    );

    static ValueFinfo< Function, bool > useTrigger(
        "useTrigger",
        "When *false*, disables event-driven calculation and turns on "
        "Process-driven calculations. \n"
        "When *true*, enables event-driven calculation and turns off "
        "Process-driven calculations. \n"
        "Defaults to *false*. \n",
        &Function::setUseTrigger,
        &Function::getUseTrigger
    );

    static ValueFinfo< Function, bool > doEvalAtReinit(
        "doEvalAtReinit",
        "When *false*, disables function evaluation at reinit, and "
        "just emits a value of zero to any message targets. \n"
        "When *true*, does a function evaluation at reinit and sends "
        "the computed value to any message targets. \n"
        "Defaults to *false*. \n",
        &Function::setDoEvalAtReinit,
        &Function::getDoEvalAtReinit
    );

    static ElementValueFinfo< Function, string > expr(
        "expr",
        "Mathematical expression defining the function. The underlying parser\n"
        "is muParser. In addition to the available functions and operators  from\n"
        "muParser, some more functions are added.\n"
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
        "ln  	     1       logarithm to base e (2.71828...)\n"
        "exp         1       e raised to the power of x\n"
        "sqrt        1       square root of a value\n"
        "sign        1       sign function -1 if x<0; 1 if x>0\n"
        "rint        1       round to nearest integer\n"
        "abs         1       absolute value\n"
        "min         var.    min of all arguments\n"
        "max         var.    max of all arguments\n"
        "sum         var.    sum of all arguments\n"
        "avg         var.    mean value of all arguments\n"
        "rand        1       rand(seed), random float between 0 and 1, \n"
        "                    if seed = -1, then a 'random' seed is created.\n"
        "rand2       3       rand(a, b, seed), random float between a and b, \n"
        "                    if seed = -1, a 'random' seed is created using either\n"
        "                    by random_device or by reading system clock\n"
        "\nOperators\n"
        "Op  meaning         		priority\n"
        "=   assignment     		-1\n"
        "&&  logical and     		1\n"
        "||  logical or      		2\n"
        "<=  less or equal   		4\n"
        ">=  greater or equal  		4\n"
        "!=  not equal         		4\n"
        "==  equal   			4\n"
        ">   greater than    		4\n"
        "<   less than       		4\n"
        "+   addition        		5\n"
        "-   subtraction     		5\n"
        "*   multiplication  		6\n"
        "/   division        		6\n"
        "^   raise x to the power of y  7\n"
        "%   floating point modulo      7\n"
        "\n"
        "?:  if then else operator   	C++ style syntax\n",
        &Function::setExpr,
        &Function::getExpr
    );

    static ValueFinfo< Function, unsigned int > numVars(
        "numVars",
        "Number of variables used by Function.",
        &Function::setNumVar,
        &Function::getNumVar
    );

    static FieldElementFinfo< Function, Variable > inputs(
        "x",
        "Input variables to the function. These can be passed via messages.",
        Variable::initCinfo(),
        &Function::getVar,
        &Function::setNumVar,
        &Function::getNumVar
    );

    static LookupValueFinfo < Function, string, double > constants(
        "c",
        "Constants used in the function. These must be assigned before"
        " specifying the function expression.",
        &Function::setConst,
        &Function::getConst
    );

    static ReadOnlyValueFinfo< Function, vector < double > > y(
        "y",
        "Variable values received from target fields by requestOut",
        &Function::getY
    );

    static ValueFinfo< Function, string > independent(
        "independent",
        "Index of independent variable. Differentiation is done based on this. Defaults"
        " to the first assigned variable.",
        &Function::setIndependent,
        &Function::getIndependent
    );

    ///////////////////////////////////////////////////////////////////
    // Shared messages
    ///////////////////////////////////////////////////////////////////
    static DestFinfo process( "process",
            "Handles process call, updates internal time stamp.",
            new ProcOpFunc< Function >( &Function::process )
            );

    static DestFinfo reinit( "reinit",
            "Handles reinit call.",
            new ProcOpFunc< Function >( &Function::reinit )
            );

    static Finfo* processShared[] = { &process, &reinit };

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
        &rate,
        &derivative,
        &mode,
        &useTrigger,
        &doEvalAtReinit,
        &expr,
        &numVars,
        &inputs,
        &constants,
        &independent,
        &proc,
        requestOut(),
        valueOut(),
        rateOut(),
        derivativeOut(),
    };

    static string doc[] =
    {
        "Name", "Function",
        "Author", "Subhasis Ray/Dilawar Singh",
        "Description",
        "General purpose function calculator using real numbers.\n"
        "It can parse mathematical expression defining a function and evaluate"
        " it and/or its derivative for specified variable values."
        "You can assign expressions of the form::\n"
        "\n"
        "f(c0, c1, ..., cM, x0, x1, ..., xN, y0,..., yP ) \n"
        "\n"
        " where `ci`'s are constants and `xi`'s and `yi`'s are variables."

        "The constants must be defined before setting the expression and"
        " variables are connected via messages. The constants can have any"
        " name, but the variable names must be of the form x{i} or y{i}"
        "  where i is increasing integer starting from 0.\n"
        " The variables can be input from other moose objects."
        " Such variables must be named `x{i}` in the expression and the source"
        " field is connected to Function.x[i]'s `input` destination field.\n"
        " In case the input variable is not available as a source field, but is"
        " a value field, then the value can be requested by connecting the"
        " `requestOut` message to the `get{Field}` destination on the target"
        " object. Such variables must be specified in the expression as y{i}"
        " and connecting the messages should happen in the same order as the"
        " y indices.\n"
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

Function::Function(): 
    valid_(false), numVar_(0), lastValue_(0.0)
    , value_(0.0), rate_(0.0), mode_(1)
    , useTrigger_(false), doEvalAtReinit_(false)
    , t_(0.0), independent_("x0")
    , parser_( unique_ptr<moose::MooseParser>(new moose::MooseParser()) )
    , stoich_(nullptr)
{
    // Parser gets it t variable from  here.
    parser_->DefineVar("t", t_);
}

Function::Function(const Function& rhs):
    valid_(rhs.valid_),
    numVar_(rhs.numVar_),
    lastValue_(rhs.lastValue_),
    value_(rhs.value_), rate_(rhs.rate_),
    mode_(rhs.mode_),
    useTrigger_( rhs.useTrigger_),
    t_(rhs.t_), independent_(rhs.independent_),
    parser_(std::move(rhs.parser_) ),
    stoich_(nullptr),
    map_(rhs.map_)
{
    xs_.clear();
    for (size_t i = 0; i < rhs.xs_.size(); i++)
        xs_.push_back(std::move(rhs.xs_[i]));

    ys_.clear();
    for (size_t i = 0; i < rhs.ys_.size(); i++)
        ys_.push_back(std::move(rhs.ys_[i]));
}

Function& Function::operator=(const Function& rhs)
{
    // protect from self-assignment.
    if( this == &rhs)
        return *this;

    valid_ = rhs.valid_;
    numVar_ = rhs.numVar_;
    lastValue_ = rhs.lastValue_;
    value_ = rhs.value_;
    mode_ = rhs.mode_;
    useTrigger_ = rhs.useTrigger_;
    rate_ = rhs.rate_;
    independent_ = rhs.independent_;

    parser_ = std::move(rhs.parser_);

    // Copy the constants
    moose::Parser::varmap_type cmap = rhs.parser_->GetConst();
    if (! cmap.empty())
        for (auto item = cmap.begin(); item != cmap.end(); ++item)
            parser_->DefineConst(item->first.c_str(), item->second);

    // Move unique_ptr 
    xs_.clear();
    for (unsigned int ii = 0; ii < rhs.xs_.size(); ++ii)
       xs_.push_back(std::move(rhs.xs_[ii]));

    // move unique_ptr
    ys_.clear();
    for (unsigned int ii = 0; ii < rhs.ys_.size(); ++ii)
        ys_.push_back(std::move(rhs.ys_[ii]));

    return *this;
}

Function::~Function()
{
    clearBuffer();
}

// do not know what to do about Variables that have already been
// connected by message.
void Function::clearBuffer()
{
    numVar_ = 0;
    parser_->ClearVariables();
}

void Function::showError(moose::Parser::exception_type &e) const
{
    cout << "Error occurred in parser.\n"
         << "Message:  " << e.GetMsg() << "\n"
         // << "Formula:  " << e.GetExpr() << "\n"
         // << "Token:    " << e.GetToken() << "\n"
         // << "Position: " << e.GetPos() << "\n"
         // << "Error code:     " << e.GetCode() << endl;
         << endl;
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

   If the name starts with anything other than `x` or `y`, then it is taken
   to be a named constant, which must be set before any expression or
   variables and error is thrown.

   NOTE: this is called not on setting expression but on first attempt
   at evaluation of the same, i.e. when you access `value` of the
   Function object.
 */
double* Function::addVariable(const char* name)
{
    double* ret = nullptr;
    string strname(name);

    // Names starting with x are variables, everything else is constant.
    if (strname[0] == 'x')
    {
        int index = atoi(strname.substr(1).c_str());

        // Only when i of xi is larger than the size of current xs_, we need to
        // resize the container.
        if ((size_t)index >= xs_.size())
        {
            // Equality with index because we cound from 0.
            for (size_t i = xs_.size(); i <= (size_t) index; i++) 
                xs_.push_back(nullptr);
        }

        if( xs_[index] == nullptr)
        {
            xs_[index] = std::move(unique_ptr<Variable>(new Variable()));
            // Add this varibale to parser. If already exists then
            // following function does nothing.
            parser_->DefineVar(strname, xs_[index]->value);
        }
        numVar_ = xs_.size();
        return &(xs_[index]->value);
    }

    if (strname[0] == 'y')
    {
        int index = atoi(strname.substr(1).c_str());
        if ((unsigned)index >= ys_.size())
        {
            ys_.reserve(index+1);
            for (int ii = 0; ii <= index; ++ii)
            {
                if (ys_[ii] == nullptr)
                {
                    ys_[ii] = std::move(unique_ptr<double>(new double()));
                    parser_->DefineVar(strname, *ys_[ii].get());
                }
            }
        }
        return ys_[index].get();
    }

    if (strname == "t")
    {
        parser_->DefineVar(strname, t_);
        return &t_;
    }

    cerr << "Got an undefined symbol: " << name << endl
         << "Variables must be named xi, yi, where i is integer index."
         << " You must define the constants beforehand using LookupField c: c[name]"
         " = value"
         << endl;
    throw moose::Parser::ParserException("Undefined constant.");
    return nullptr;
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Find all x\d+ and y\d+ in the experssion.
 *
 * @Param expr
 * @Param vars
 */
/* ----------------------------------------------------------------------------*/

void Function::setExpr(const Eref& eref, string expr)
{
    MOOSE_DEBUG( this << " : Setting expression " <<  expr );
    this->innerSetExpr( eref, expr ); // Refer to the virtual function here.
}

// Virtual function, this does the work.
void Function::innerSetExpr(const Eref& eref, string expr)
{
    valid_ = false;
    clearBuffer();
    xs_.resize(numVar_);

    // Reinitialize the parser.
    parser_->Reinit();

    // Find all variables x\d+ or y\d+ etc, and add them to variable buffer.
    vector<string> xs;
    vector<string> ys;

    moose::MooseParser::findXsYs( expr, xs, ys);

    // Now create a map which maps the variable name to location of values. This
    // is critical to make sure that pointers remain valid when multi-threaded
    // encironment is used.
    addVariable("t");
    for( size_t i = 0; i < xs.size(); i++ )
    {
        addVariable( xs[i].c_str());
        // get the address of Variable's value. Is it safe in multi-threaded
        // environment? I hope so.
        map_[xs[i]] = &(xs_[i]->value);
    }

    for( size_t i = 0; i < ys.size(); i++ )
    {
        addVariable( ys[i].c_str());
        map_[ys[i]] = ys_[i].get();
    }

    try
    {
        // Set parser expression. Send the map and the array of values as well.
        parser_->SetVariableMap( map_ );
        valid_ = parser_->SetExpr( expr );
    }
    catch (moose::Parser::exception_type &e)
    {
        cerr << "Error setting expression on: " << eref.objId().path() << endl;
        valid_ = false;
        showError(e);
        clearBuffer();
    }

    MOOSE_DEBUG( this << "   Valid = " << valid_);
}

string Function::getExpr( const Eref& e ) const
{
    if (!valid_)
    {
        cout << "Error: " << e.objId().path() << "::getExpr() - invalid parser state" << endl;
        return "";
    }
    return parser_->GetExpr();
}

void Function::setMode(unsigned int mode)
{
    mode_ = mode;
}

unsigned int Function::getMode() const
{
    return mode_;
}

void Function::setUseTrigger(bool useTrigger )
{
    useTrigger_ = useTrigger;
}

bool Function::getUseTrigger() const
{
    return useTrigger_;
}

void Function::setDoEvalAtReinit(bool doEvalAtReinit )
{
    doEvalAtReinit_ = doEvalAtReinit;
}

bool Function::getDoEvalAtReinit() const
{
    return doEvalAtReinit_;
}

double Function::getValue() const
{
    return parser_->Eval( );
}


double Function::getRate() const
{
    if (!valid_)
    {
        cout << "Error: Function::getValue() - invalid state" << endl;
    }
    return rate_;
}

void Function::setIndependent(string var)
{
    independent_ = var;
}

string Function::getIndependent() const
{
    return independent_;
}

vector< double > Function::getY() const
{
    vector < double > ret(ys_.size());
    for (unsigned int ii = 0; ii < ret.size(); ++ii)
    {
        ret[ii] = *ys_[ii];
    }
    return ret;
}

double Function::getDerivative() const
{
    double value = 0.0;
    if (!valid_)
    {
        cout << "Error: Function::getDerivative() - invalid state" << endl;
        return value;
    }
    moose::Parser::varmap_type variables = parser_->GetVar();
    moose::Parser::varmap_type::const_iterator item = variables.find(independent_);
    if (item != variables.end())
    {
        try
        {
            value = parser_->Diff(item->second, item->second);
        }
        catch (moose::Parser::exception_type &e)
        {
            showError(e);
        }
    }
    return value;
}

void Function::setNumVar(const unsigned int num)
{
    clearBuffer();
    for (unsigned int ii = 0; ii < num; ++ii)
        addVariable(("x"+std::to_string(ii)).c_str());
}

unsigned int Function::getNumVar() const
{
    return numVar_;
}

void Function::setVar(unsigned int index, double value)
{
    cout << "xs_[" << index << "]->setValue(" << value << ")" << endl;
    if (index < xs_.size())
        xs_[index]->setValue(value);
    else
        cerr << "Function: index " << index << " out of bounds." << endl;
}

Variable * Function::getVar(unsigned int ii)
{
    static Variable dummy;
    if ( ii < xs_.size())
        return xs_[ii].get();

    MOOSE_WARN( "Warning: Function::getVar: index: "
                << ii << " is out of range: "
                << xs_.size() );
    return &dummy;
}

void Function::setConst(string name, double value)
{
    parser_->DefineConst(name.c_str(), value);
}

double Function::getConst(string name) const
{
    moose::Parser::varmap_type cmap = parser_->GetConst();
    if (! cmap.empty() )
    {
        moose::Parser::varmap_type::const_iterator it = cmap.find(name);
        if (it != cmap.end())
        {
            return it->second;
        }
    }
    return 0;
}

void Function::process(const Eref &e, ProcPtr p)
{
    if( ! valid_ )
        return;

    // Update values of incoming variables.
    vector < double > databuf;
    requestOut()->send(e, &databuf);

    t_ = p->currTime;
    value_ = getValue();
    rate_ = (value_ - lastValue_) / p->dt;

    for (size_t ii = 0; (ii < databuf.size()) && (ii < ys_.size()); ++ii)
        *ys_[ii] = databuf[ii];

    if ( useTrigger_ && value_ < TriggerThreshold )
    {
        lastValue_ = value_;
        return;
    }

    if( 1 == mode_ )
    {
        valueOut()->send(e, value_);
        lastValue_ = value_;
        return;
    }
    if( 2 == mode_ )
    {
        derivativeOut()->send(e, getDerivative());
        lastValue_ = value_;
        return;
    }
    if( 3 == mode_ )
    {
        rateOut()->send(e, rate_);
        lastValue_ = value_;
        return;
    }
    else
    {
        valueOut()->send(e, value_);
        derivativeOut()->send(e, getDerivative());
        rateOut()->send(e, rate_);
        lastValue_ = value_;
        return;
    }
}

void Function::reinit(const Eref &e, ProcPtr p)
{
    if (!valid_)
    {
        cout << "Error: Function::reinit() - invalid parser state. Will do nothing." << endl;
        return;
    }

    t_ = p->currTime;

    if (doEvalAtReinit_)
        lastValue_ = value_ = getValue();
    else
        lastValue_ = value_ = 0.0;

    rate_ = 0.0;

    if (1 == mode_)
    {
        valueOut()->send(e, value_);
        return;
    }
    if( 2 == mode_ )
    {
        derivativeOut()->send(e, 0.0);
        return;
    }
    if( 3 == mode_ )
    {
        rateOut()->send(e, rate_);
        return;
    }
    else
    {
        valueOut()->send(e, value_);
        derivativeOut()->send(e, 0.0);
        rateOut()->send(e, rate_);
        return;
    }
}
