/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
** Written by Raamesh Deshpande 2007
** Modified by Upi Bhalla 2010
**********************************************************************/

#include "header.h"
#include "MathFunc.h"

using namespace MathFuncNames;

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		static SrcFinfo1< double > output( 
			"output", 
			"Sends out result of computation"
		);

const Cinfo* MathFunc::initCinfo()
{

	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		static ValueFinfo< MathFunc, string > mathML( 
			"mathML", 
			"MathML version of expression to compute",
			&MathFunc::setMathMl,
			&MathFunc::getMathML
		);
		static ValueFinfo< MathFunc, string > func( 
			"function", 
			"function is for functions of form f(x, y) = x + y",
			&MathFunc::setFunction,
			&MathFunc::getFunction
		);
		static ReadOnlyValueFinfo< MathFunc, double > result( 
			"result",
			"result value",
			&MathFunc::getR
		);
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		/*
		static SrcFinfo1< double > output( 
			"output", 
			"Sends out result of computation"
		);
		*/
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		static DestFinfo process( "process", 
			"Handle process call",
			new ProcOpFunc< MathFunc >( &MathFunc::processFunc )
		);

		static DestFinfo reinit( "reinit", 
			"Handle reinit call",
			new ProcOpFunc< MathFunc >( &MathFunc::reinitFunc )
		);
	
		/*
		static DestFinfo args( "args",
			"Handle args one at a time", 
			new OpFunc1< MathFunc, double >( &MathFunc::argFunc )
		);
		*/
	
		static DestFinfo arg1( "arg1",
			"Handle arg1",
			new OpFunc1< MathFunc, double >( &MathFunc::arg1Func )
		);
	
		static DestFinfo arg2( "arg2",
			"Handle arg2",
			new OpFunc1< MathFunc, double >( &MathFunc::arg2Func )
		);
	
		static DestFinfo arg3( "arg3",
			"Handle arg3",
			new OpFunc1< MathFunc, double >( &MathFunc::arg3Func )
		);
	
		static DestFinfo arg4( "arg4",
			"Handle arg4",
			new OpFunc1< MathFunc, double >( &MathFunc::arg4Func )
		);
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, 
			sizeof( procShared ) / sizeof( const Finfo* ) );

	static Finfo* mathFuncFinfos[] = {
		&mathML,	// Value
		&func,		// Value
		&result,	// Value
		&output,	// SrcFinfo
//		&args, 		// DestFinfo
		&arg1, 		// DestFinfo
		&arg2, 		// DestFinfo
		&arg3, 		// DestFinfo
		&arg4, 		// DestFinfo
		&proc, 		// SharedFinfo
	};

	/*
	static string doc[] =
	{
		"Name", "MathFunc",
		"Author", "Raamesh Deshpande, 2007, NCBS",
		"Description", "MathFunc: Object for parsing function definitions and executing them in a simulation.",
	};	
	*/
	static Cinfo mathFuncCinfo(
		"MathFunc",
		Neutral::initCinfo(),
		mathFuncFinfos,
		sizeof( mathFuncFinfos )/sizeof(Finfo *),
		new Dinfo< MathFunc >()
	);

	return &mathFuncCinfo;
}

///////////////////////////////////////////////////////////////////////
// Class Definition stuff.
///////////////////////////////////////////////////////////////////////

static const Cinfo* mathFuncCinfo = MathFunc::initCinfo();

MathFunc::MathFunc()
{
  expect_ = EXPRESSION;
  
  status_ = BLANK;
  
  /*used for infix to prefix function*/
  precedence_[MINUS] = 1;
  precedence_[PLUS] = 0;
  precedence_[TIMES] = 2; 
  precedence_[DIVIDE] = 3;
  precedence_[POWER] = 4;
  precedence_[SIN] = 5;
  precedence_[COS] = 5;
  precedence_[TAN] = 5;
  precedence_[ARCSIN] = 5;
  precedence_[ARCCOS] = 5;
  precedence_[ARCTAN] = 5;
  precedence_[SQRT] = 5;
  precedence_[SUM] = 5;
  precedence_[VARIANCE] = 5;
  precedence_[MEAN] = 5;
  precedence_[SDEV] = 5;
  precedence_[PRODUCT] = 5;
  precedence_[RPAREN] = -1;
}


//////////////////////////////////////////////////////////////////
// MOOSE field function definitions.
//////////////////////////////////////////////////////////////////

void MathFunc::setMathMl( string value )
{
  mmlstring_ = value;
  status_ = MMLSTRING;
}

string MathFunc::getMathML() const
{
	return mmlstring_;
}

void MathFunc::setFunction(string value)
{
  fn_ = value;
  status_ = FNSTRING;
}

string MathFunc::getFunction() const
{
  return fn_; 
}

double MathFunc::getR() const
{
  return result_;
}

///////////////////////////////////////////////////////////////////////
// Process DestFinfos
///////////////////////////////////////////////////////////////////////

void MathFunc::processFunc( const Eref& e, ProcPtr info)
{
  /*if the mml function function has not been initialized then don't care about anything*/
  if (status_ == BLANK || status_ == ERROR) 
    return;
  
  if (status_ == MMLSTRING)
    executeFunction();
  else if (status_ == FNSTRING)
    infixToPrefix();
  else 
    assert(0);/*control should never reach here.*/
  
  
  /*If the execute function or the infix to prefix has not worked properly then ERROR is set in status_*/
  if (status_ == ERROR){
    cout << "Error!" << endl;
    return;
  }
  
  /* if the argument is the vector or there are enough arguments to satify the function then continue else return*/
  if (!(vector_name_ != "" || v_.size () <= v.size()))
    return;
  result_ = getResult();
  if (status_ == ERROR){
    result_ = 0;
    cout << "Error!" << endl;
    return;
  }
  //cout << result_ << endl;
  // send1< double >( e, outputSlot, result_ );
  output.send( e, info, result_ );
  v.clear();
}

double MathFunc::op( const vector< double >& args )
{
	v = args;
  if (status_ == BLANK || status_ == ERROR) 
    return 0.0;
  
  if (status_ == MMLSTRING)
    executeFunction();
  else if (status_ == FNSTRING)
    infixToPrefix();
  else 
    assert(0);/*control should never reach here.*/
  
  
  /*If the execute function or the infix to prefix has not worked properly then ERROR is set in status_*/
  if (status_ == ERROR){
    cout << "Error!" << endl;
    return 0.0;
  }
  
  /* if the argument is the vector or there are enough arguments to satify the function then continue else return*/
  if (!(vector_name_ != "" || v_.size () <= v.size()))
    return 0.0;
  result_ = getResult();
  if (status_ == ERROR){
    result_ = 0;
    cout << "Error!" << endl;
    return 0.0;
  }
  return result_;
}

void MathFunc::reinitFunc( const Eref& e, ProcPtr info ) 
{
  // cout << "reiniting..." << endl;
  v.clear();
  status_ = BLANK;
}

//////////////////////////////////////////////////////////////////
// Other DestFinfo definitions
//////////////////////////////////////////////////////////////////

/*
void MathFunc::argFunc( double d )
{
  // v.push_back(&d); // Ugh. There is vector < double* > v_, and v both.
  	// Ugh. Why is he storing a pointer to a temporary?
}
*/

void MathFunc::arg1Func( double d )
{
	while ( v.size() != 1)  // surely this should be 1?
		v.push_back( d ); // Upi Sez: I'm dubious about this.
}

void MathFunc::arg2Func( double d )
{
  	while (v.size() != 2) 
		v.push_back(d);
}

void MathFunc::arg3Func( double d )
{
	while ( v.size() != 3) 
		v.push_back( d );
}

void MathFunc::arg4Func( double d )
{
	while ( v.size() != 4) 
		v.push_back( d );
}

//////////////////////////////////////////////////////////////////
// Utility and parsing funcs
//////////////////////////////////////////////////////////////////


void MathFunc::error(int lineno, string errormsg){
  cout << "Error detected at line number " << lineno << ": " << errormsg << endl;
  status_ = ERROR;
}

void MathFunc::error(string errormsg){
  cout << errormsg << endl;
  status_ = ERROR;
}

void trim(string &s){
    if (s.size() > 0)
    {
        if (s[0] == ' ')
        {
            int i = 0;
            while (s[i++]==' ')
                ;
            s.erase(0, i-1);
        }
        if (s[s.size()-1] == ' ')
        {
            int i = 0;
            while (s[s.size() - 1 - i++]==' ')
                ;
            s.erase(s.size() - i+1);
        }
    }    
}

void getLine(string &mmlfile, string &s){
  size_t pos = mmlfile.find("\n");
  if (pos == string::npos){
    s = mmlfile;
    mmlfile.erase();
  }
  else{
    s = mmlfile.substr(0, pos);
    mmlfile.erase(0, pos+1);
  }
}

bool eof (string mmlfile){
  return (mmlfile.size() == 0);
}

string getNextToken(string &s, string &mmlfile, int &lineno){
  if ( s.size() > 0 )
    trim(s);
  if (s.size() == 0) {
    if (eof(mmlfile)) return "done!";
    getLine (mmlfile, s);
    trim(s);
    lineno++;
  }
  string ret;
  int start = 0;
  size_t length;
  if (s[0] == '<'){ length = s.find(">") + 1; }	  // if string is like this <ci>...
  else { 					  
    length = s.find("<");			  // if string is like this 45 </ci>...
    if (length == string::npos) {ret = s; s.erase(); return ret;}	  // if string is like this 45
  }
  ret = s.substr( start, length);
  trim(ret);
  s.erase(0, length);
  return ret;
}

void trim1(string &token){ //assumes the token are trim_med. This removes double spaces and spaces before and after <>
  if (token.find(" ") == string::npos) return;
  size_t i = 0;
  while (i < token.size()){
    switch(token[i]){
      case '<':
        while(token[i+1] == ' ') { assert(i < token.size() - 1); token.erase(i+1, 1);}
        break;
      case ' ':
        while(token[i+1] == ' ') { assert(i < token.size() - 1); token.erase(i+1, 1);}
        if (token[i+1] == '>' || token[i+1] == '=') {token.erase(i, 1); i--;}
        break;
      case '=':
        while (token[i+1] == ' ') { assert(i < token.size() - 1); token.erase(i+1, 1);}
        break;
      default:
        break;//do nothing
    }
    i++;
  }
}

int whatToken(string token, int &expect){
  trim1(token);
  size_t pos;
  if ((pos = token.find("<ci")) != string::npos){
    if (token == "<ci>"){
      if (expect == CIF) {expect = FUNCTION; }
      else expect = VARIABLE;
      return CI; 
    }
    if (token == "<ci type=\"function\">") {expect = FUNCTION; return CIF;}
    if (token == "<ci type=\"vector\">") {expect = MathFuncNames::VECTOR; return CIV;}
    else cout << "Unknown ci: " << token << endl;
    return CIF;
  }
  if (token == "<apply>") 	{expect = FUNCTION; return NOTHING;}
  if (token == "<eq/>") 	{expect = EXPRESSION; return EQ;} 
  if (token == "<sin/>") 	{expect = CNI; return SIN;}
  if (token == "<cos/>") 	{expect = CNI; return COS;}
  if (token == "<tan/>") 	{expect = CNI; return TAN;}
  if (token == "<arctan/>") 	{expect = CNI; return ARCTAN;}
  if (token == "<arcsin/>") 	{expect = CNI; return ARCSIN;}
  if (token == "<arccos/>") 	{expect = CNI; return ARCCOS;}
  if (token == "<times/>") 	{expect = CNI; return TIMES;}
  if (token == "<plus/>") 	{expect = CNI; return PLUS;}
  if (token == "<minus/>") 	{expect = CNI; return MINUS;}
  if (token == "<divide/>") 	{expect = CNI; return DIVIDE;}
  if (token == "<power/>")	{expect = CNI; return POWER;}
  if (token == "<sqrt/>") 	{expect = CNI; return SQRT;}
  if (token == "<sum/>") 	{expect = CNI; return SUM;}
  if (token == "<product/>") 	{expect = NUMBER; return PRODUCT;}
  if (token == "<mean/>") 	{expect = CNI; return MEAN;}
  if (token == "<sdev/>") 	{expect = CNI; return SDEV;}
  if (token == "<variance/>") 	{expect = CNI; return VARIANCE;}
  if (token == "<apply>") 	{expect = CIF; return APPLY;}
  if (token == "</apply>") 	{expect = DONTKNOW; return APPLYOVER;}
  if (token == "<cn>") 		{expect = NUMBER; return CN;}
  if (token == "</cn>") 	{expect = DONTKNOW; return CNOVER;}
  if (token == "</ci>") 	{expect = DONTKNOW; return CIOVER;}
  if (token == "<bvar>") 	{expect = CNI; return BVAR;}
  if (token == "</bvar>") 	{expect = LOWLIMIT; return BVAROVER;}
  if (token == "<lowlimit>") 	{expect = CNI; return LOWLIMIT;}
  if (token == "</lowlimit>") 	{expect = UPLIMIT; return LOWLIMITOVER;}
  if (token == "<uplimit>") 	{expect = CNI; return UPLIMIT;}
  if (token == "</uplimit>") 	{expect = EXPRESSION; return UPLIMITOVER;}  
  if (token == "<selector/>")	{expect = CI; return SELECTOR;}
  if (token == "done!")		{expect = NOTHING; return DONE;}
  if (expect == FUNCTION)	{expect = CIOVER; return FUNCTION;}
  if (expect == VARIABLE)	{expect = CIOVER; return VARIABLE;}
  if (expect == NUMBER)		{expect = CNOVER; return NUMBER;}
  if (expect == MathFuncNames::VECTOR)		{expect = CIOVER; return MathFuncNames::VECTOR;}
  return ERROR;
}

//////////////////////////////////////////////////////////////////
// Evalutaion funcs
//////////////////////////////////////////////////////////////////

double trigEval(int trigFunc, double angle){
  switch(trigFunc){
    case SIN:
      return sin(angle);
    case COS:
      return cos(angle);
    case TAN:
      return tan(angle);
    case ARCSIN:
      return asin(angle);
    case ARCCOS:
      return acos(angle);
    case ARCTAN:
      return atan(angle);
    default:
      assert(0);
  }
  assert(0);
  return 0;
}

void MathFunc::evaluate(int pos, int arity){
  //int arity = stack.size() - pos - 1;
  double result;
  if (arity == 0) {
      error("The function has no arguments to evaluate...Assuming the function result is 1 and continuing...");
      stack_.erase(stack_.begin()+pos, stack_.begin() + pos + arity);
      stack_.push_back(1);//work
      function_.erase(function_.begin()+pos, function_.begin() + pos  + arity);
      function_.push_back(0);//work
      return;
  }
  switch ((int)stack_[pos]){
    case SIN:
    case COS:
    case TAN:
    case ARCTAN:
    case ARCCOS:
    case ARCSIN:
      if (arity != 1) error("The Trig function has more than one arguments! However continuing using the first argument...");
      result = trigEval((int)stack_[pos], stack_[pos+1]);
      break;
    case PLUS:
      result = 0;
      for (size_t i = pos +1; i < MIN((size_t)pos +1+ arity, stack_.size()); i++)
        result += stack_[i];
      break;
    case MINUS:
      if (arity == 1) result = -stack_[pos+1];
      if (arity == 2) result = stack_[pos+1] - stack_[pos+2];
      if (arity > 2) {
        result = stack_[pos+1] - stack_[pos+2];
        error("More operators than required. Continuing using only 2 needed for minus...");
      }
      break;
    case TIMES: 
      result = 1;
      for (size_t i = pos +1; i < MIN((size_t)pos +1+ arity, stack_.size()); i++)
        result *= stack_[i];
      break;
    case DIVIDE:
      if (arity == 1) {error("No Divisor!! Not doing any division!"); result = stack_[pos+1];}
      if (arity == 2) result = stack_[pos+1] / stack_[pos+2];
      if (arity > 2) {
        result = stack_[pos+1] / stack_[pos+2];
        error("More operators than required. Continuing using only 2 needed for division...");
      }
      break;
    case POWER:
      if (arity == 1) {error("No power index! Not doing any powering.."); result = stack_[pos+1];}
      if (arity == 2) result = pow(stack_[pos+1],  stack_[pos+2]);
      if (arity > 2) {
        result = pow(stack_[pos+1] , stack_[pos+2]);
        error("More operators than required. Continuing using only 2 needed....");
      }
      break;
    case SQRT:
      if (arity > 1) error("More operators than required. Continuing using only 1 needed for sqrt...");
      result = sqrt(stack_[pos+1]);
      break;
    case SUM:
      if (arity != 1) error("More operators than required. Continuing using only 1 needed for sum...");
      if (function_[pos + 1] != 3) error("Vector need for sum");
      result = 0;
      for (size_t i = (size_t)stack_[pos+1]; i < v_.size(); i++){ result+=*v_[i]; }
      break;
    case PRODUCT:
      if (arity != 1) error("More operators than required. Continuing using only 1 vector needed for product...");
      if (function_[pos + 1] != 3) error("Vector need for product");
      result = 0;
      for (size_t i = 0; i < v_.size(); i++) result*=*v_[i];
      break;
    case MEAN:
      if (arity != 1) error("More operators than required. Continuing using only 1 vector needed for sum...");
      if (function_[pos + 1] != 3) error("Vector need for mean");
      result = 0;
      
      for (size_t i = 0; i < v_.size(); i++) result+=*v_[i];
      result /= v_.size();
      break;
    case VARIANCE:
      {
      if (arity != 1) error("More operators than required. Continuing using only 1 vector needed for variance...");
      if (function_[pos + 1] != 3) error("Vector need for variance");
      double sum = 0, sum2 = 0;
      for (size_t i = 0; i < v_.size(); i++) sum+=*v_[i];
      double mean = sum / v_.size();
      for (size_t i = 0; i < v_.size(); i++) sum2+=(*v_[i])*(*v_[i]);
      result = (sum2 - mean*mean)/ v_.size();
      }
      break;
    case SDEV:
      {
      if (arity != 1) error("More operators than required. Continuing using only 1 vector needed for variance...");
      if (function_[pos + 1] != 3) error("Vector need for sdev");
      double sum = 0, sum2 = 0;
      for (size_t i = 0; i < v_.size(); i++) sum+=*v_[i];
       double mean = sum / v_.size();
      for (size_t i = 0; i < v_.size(); i++) sum2+=(*v_[i])*(*v_[i]);
      result = sqrt((sum2 - mean*mean)/ v_.size());
      }
      break;
    default:
      break;
  }
  stack_.erase(stack_.begin()+pos, stack_.begin() + pos + arity +1 );
  //stack_.push_back(result);
  stack_.insert(stack_.begin() + pos, 1, result);
  function_.erase(function_.begin()+pos, function_.begin() + pos + arity+1);
  //function_.push_back(false);
  function_.insert(function_.begin() + pos, 1, 0);
}

void MathFunc::executeFunction(){ //now this filename is the whole file string...hoho
  //ifstream mmlfile (filename.c_str());
  clear();
  string mmlfile = mmlstring_;
  status_ = MMLSTRING;
  vector_name_ = "";
  string line = "";
  string token; 
  int lineno = 0;
  while((token = getNextToken(line, mmlfile, lineno))!="done!"){
    int currentToken = whatToken(token, expect_);
    //map<string,double>::iterator iter;
    switch (currentToken){
      case SIN:
      case COS:
      case TAN:
      case ARCTAN:
      case ARCSIN:
      case ARCCOS:
      case PLUS:
      case MINUS:
      case TIMES:
      case DIVIDE:
      case POWER:
      case SQRT:
      case SUM:
      case PRODUCT:
      case MEAN:
      case SDEV:
      case VARIANCE:
        stack_.push_back(currentToken);
        function_.push_back(1);
        break;
      case APPLY:
      case EQ:
      case FUNCTION:
      case CIF:
      case CIOVER:
      case CNOVER:
      case CI:
      case CIV:
      case CN:
        break;//neglect
      case NUMBER:
        stack_.push_back(atof(token.c_str()));
        function_.push_back(0);
        break;
      case VARIABLE:
        {
        if (token == vector_name_) {stack_.push_back(MathFuncNames::VECTOR); function_.push_back(3); break;}
        int pos = -1;
        for (size_t i = 0; i < vname_.size(); i++){
          if (token == vname_[i]) {pos = i; break;} 
        }
        if (pos != -1) {
          stack_.push_back(pos);
          function_.push_back(2);
          break;
        }
        map<string,double>::iterator iter = symtable_.find(token);
        if( iter == symtable_.end() ) { 
		  symtable_[token] = 0.0;
		  iter = symtable_.find( token );
		  v_.push_back( &( iter->second ) );
          vname_.push_back(token);
          break;
        }
        else {
          int pos = -1;
          for (size_t i = 0; i < vname_.size(); i++){
            if (vname_[i] == token) {pos = i; break;}
          }
          assert (pos != -1);
          stack_.push_back(pos);
          function_.push_back(2);
        }
        }
        break;
      case APPLYOVER:
        /*{
        if (function_.size() <= 1) break;
        int pos = -1;
        for (int i = function_.size() - 1; i >= 0; i--)
          if (function_[i] == 1) {pos = i; break;}
        if (pos == -1)
          if (function_.size() != 0) error(lineno, "No operator present!");
        evaluate(stack_, function_, pos, lineno, v);
        }*/
        stack_.push_back(APPLYOVER);
        function_.push_back(1);
        break;
      case MathFuncNames::VECTOR:
        //vname_.push_back(token);
        vector_name_ = token;
        break;
      case ERROR:
        error(lineno, "");
        break;
      default:
        break;
    }
  }
  for (size_t k = 0; k < stack_.size(); k++)
    if (function_[k] == 3) stack_[k] = v_.size();
}

double MathFunc::getResult(){
  if (status_ == ERROR || status_ == BLANK){error("function not initialized properly"); return 0; } 
  for (size_t k = 0; k < v_.size(); k++)
    *v_[k] = v[k];
  for (size_t k = v_.size(); k < v.size(); k++)
    v_.push_back(&v[k]);
  for (size_t k = 0; k < stack_.size(); k++)
    if(function_[k] == 2){
       string str = vname_[(int)stack_[k]];
       stack_[k] = symtable_[str];
       function_[k] = 0;
     }
  assert(v.size()==v_.size());
  for (size_t i = 0; i < stack_.size(); i++){
    //cout << stack_[i] << endl;
    if (function_[i] == 1 && stack_[i] == APPLYOVER) {
      //for (size_t k = 0; k < stack_.size(); k++) cout << stack_[k] << " " ; cout << endl;
       /*Don't delete this comment. It has been extremely helpful in debugging*/
       /*This commented piece of code prints out the stack as it is being executed*/
      /*for (size_t k = 0; k < stack_.size(); k++) {
        if (function_[k] == 1){
          switch ((int)stack_[k]){
          case PLUS:
            cout << "+" << "\t";
            break;
          case MINUS:
            cout << "-" << "\t";
            break;
          case TIMES:
            cout << "*" << "\t";
            break;
          case DIVIDE:
            cout << "/" << "\t";
            break;
          case POWER:
            cout << "^" << "\t";
            break;
          default:
            cout << "f:" << stack_[k] << "\t";
          }
        }
        else cout << stack_[k] << "\t" ; 
      }
      cout << endl;
      cout << i << endl;*/
      stack_.erase(stack_.begin() + i);
      function_.erase(function_.begin() + i);
      if (i == 0 ) continue;
      i--;
      int pos = -1;
      for (int j = i; j >=0; j--)
        if (function_[j]== 1) {pos = j; break;}
      if (pos == -1 && i != 0) error("No operator present!");
      int arity = i - pos ;
      /*hack..*/
      if(status_ == FNSTRING){
        switch((int)stack_[pos]){
          case PLUS:
          case TIMES:
          case POWER:
          case DIVIDE:
            arity = MIN(arity, 2);
            break;
          case MINUS:
            if (function_[pos+1] != 1 && function_[pos+2] != 1)
              arity = 2;
            else arity = 1;
            break;
          default:
            arity = 1;
            break;
        }
      }
      
      /*for (size_t k = 0; k < stack_.size(); k++) {
        if (function_[k] == 1){
          switch ((int)stack_[k]){
          case PLUS:
            cout << "+" << "\t";
            break;
          case MINUS:
            cout << "-" << "\t";
            break;
          case TIMES:
            cout << "*" << "\t";
            break;
          case DIVIDE:
            cout << "/" << "\t";
            break;
          case POWER:
            cout << "^" << "\t";
            break;
          default:
            cout << "f:" << stack_[k] << "\t";
          }
        }
        else cout << stack_[k] << "\t" ; 
      }
      cout << endl;*/
      //for (size_t k = 0; k < function_.size(); k++) cout << function_[k] << "\t" ; cout << endl;
      //cout << pos << " " << arity << endl;
      evaluate(pos, arity);
      i = pos;
    }
  }
  if (stack_.size() != 1) status_ = ERROR;
  else status_ = BLANK;
  return stack_[0];
}


/*returns VARIABLE, NUMBER, trigs, etc*/
int getTokenType(string &token){
/*according to the token send the number (SIN, PLUS, etc)*/
  char ch = token[0];
  if(ch == '(') return LPAREN;
  if(ch == ')') return RPAREN;
  if(ch == '+') return PLUS;
  if(ch == '-') return MINUS;
  if(ch == '/') return DIVIDE;
  if(ch == '*') return TIMES;
  if(ch == '^') return POWER;
  if (token == "sin") return SIN;
  if (token == "cos") return COS;
  if (token == "tan") return TAN;
  if (token == "arcsin") return ARCSIN;
  if (token == "arccos") return ARCCOS;
  if (token == "arctan") return ARCTAN;
  if (token == "mean") return MEAN;
  if (token == "sum") return SUM;
  if (token == "product") return PRODUCT;
  if (token == "sdev") return SDEV;
  if (token == "variance") return VARIANCE;
  if (((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z' )) && token[token.size()- 1] == '_') return MathFuncNames::VECTOR;
  if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z' )) return VARIABLE;
  if (ch >= '0' && ch <= '9') return NUMBER;
  if (ch == '.' && token.size() >= 2 ){
    if (token[1] >= '0' && token[1] <= '9') return NUMBER;
    else return ERROR;
  }  
  return ERROR;
}

/*Verified? -- error up but correction to be made here.  */ 
void getInfixToken(string &s, string &token){
  /*a token can be a number, a variable - string, parenthesis, operators +-/^*, 
  trigs, sqrt, vector ops - sum, product, sdev, variance, mean*/
  /*Use the s[0] to find out what the token could be.*/
  if (s[0] == '(') {token = "("; s = s.substr(1); return;}
  if (s[0] == ')') {token = ")"; s = s.substr(1); return;}
  if (s[0] == '+') {token = "+"; s = s.substr(1); return;}
  if (s[0] == '-') {token = "-"; s = s.substr(1); return;}
  if (s[0] == '/') {token = "/"; s = s.substr(1); return;}
  if (s[0] == '*') {token = "*"; s = s.substr(1); return;}
  if (s[0] == '^') {token = "^"; s = s.substr(1); return;}
  if (s[0] >= '0' && s[0] <= '9'){
    size_t i = 0;
    bool decimal_used = false;
    bool exponent_used = false;
    while ((s[i] >= '0' && s[i] <= '9') && i < s.size() ){
      //cout << i << " " << s << " " << token << endl;
      token += s[i];
      /*Decimal: If the next char is decimal and it is the first decimal encountered then append it to token*/
      if (i +1 < s.size() && s[i+1] == '.' && !decimal_used ) {
        token += s[++i];
        decimal_used = true;
      }
      /*Exponent: allows 4.3E5, etc.*/
      else if (i +1 < s.size() && (s[i+1] == 'E' || s[i+1] == 'e') && !exponent_used) {
        token += s[++i];
        exponent_used = true;
      }
      else if (i + 2 < s.size() && s[i+1] == '-' && (s[i+2] == 'E' || s[i+2] == 'e')){
        token += s[++i];
        token += s[++i];
      }
      else if (i + 2 < s.size() && s[i+1] == '+' && (s[i+2] == 'E' || s[i+2] == 'e')){
        token += s[++i];
        token += s[++i];
      }
      i++;
    }
    s = s.substr(i);
    
    /*size_t pos = s.find_first_not_of("0123456789", 0);
    if (pos == string::npos) {token = s; s = ""; return;}
    token = s.substr(0, pos);
    s = s.substr(pos);*/
    return;
  }
  if ((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z') ||(s[0] == '_')){
    /*find out first non alphanumberic char*/
    size_t i = 1;
    while ((s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= 'a' && s[i] <= 'z') ||  (s[i] >= '0' && s[i] <= '9'))
      i++;
    token = s.substr(0, i);
    s = s.substr(i);
  }
  return;
}

bool testGetInfixToken(){
  string str = "x+y*z^43-(3.77)+12E-56/45+sin(12e-45)";
  string token;
  getInfixToken(str, token); if (token != "x") return false;
  getInfixToken(str, token); if (token != "+") return false;
  getInfixToken(str, token); if (token != "y") return false;
  getInfixToken(str, token); if (token != "*") return false;
  getInfixToken(str, token); if (token != "z") return false;
  getInfixToken(str, token); if (token != "^") return false;
  getInfixToken(str, token); if (token != "43") return false;
  getInfixToken(str, token); if (token != "-") return false;
  getInfixToken(str, token); if (token != "(") return false;
  getInfixToken(str, token); if (token != "3.77") return false;
  getInfixToken(str, token); if (token != ")") return false;
  getInfixToken(str, token); if (token != "+") return false;
  getInfixToken(str, token); if (token != "12E-56") return false;
  getInfixToken(str, token); if (token != "/") return false;
  getInfixToken(str, token); if (token != "45") return false;
  getInfixToken(str, token); if (token != "+") return false;
  getInfixToken(str, token); if (token != "sin") return false;
  getInfixToken(str, token); if (token != "(") return false;
  getInfixToken(str, token); if (token != "12e-45") return false;
  getInfixToken(str, token); if (token != ")") return false;
  getInfixToken(str, token); if (token != "") return false;
  return true;
}

bool MathFunc::storeArgNames(string args){
  /*break with [, ] as delimiters, store the names in vname_, create places in 
  symtable*/
  size_t pos1, pos2;
  pos1 = args.find("(");
  pos2 = args.find(")");
  if (pos1 == string::npos || pos2 == string::npos || pos1 > pos2)
    {error("Function declaration not correct."); return false;}
  if (pos1 == pos2) return true;
  args = args.substr(pos1+1, pos2 - pos1 - 1);
  size_t index = 0;
  while (index < args.size()){
    size_t pos1 = args.find_first_not_of(", ", index);
    size_t pos2 = args.find_first_of(", ", pos1);
    if (pos2 == string::npos) pos2 = args.size();
    if (pos1 == string::npos || pos2 == string::npos || pos1 > pos2) 
      {error("Error in argument part"); return false;}
    index = pos2;
    string variable = args.substr(pos1, pos2 - pos1); 
    /*check whether the variable is a vector or not!!*/
    if (variable[variable.size() - 1] == '_'){
      vector_name_ = variable;
      continue;
    }
    
    /*not a vector therefore variable*/
    map<string,double>::iterator iter = symtable_.find(variable);
    if (iter == symtable_.end()){
		symtable_[variable] = 0.0;
		iter = symtable_.find( variable );
		v_.push_back( &( iter->second ) );
    }
    vname_.push_back(variable);
  }
  return true;
}

bool MathFunc::testStoreArgNames(){
  string args = "x, y, z, k, helloWorld123";
  map<string,double>::iterator iter;
  vector <string> vname_temp = vname_;
  vname_.clear();
  storeArgNames(args);
  if (vname_.size() != 5) return false;
  if (vname_[0] != "x") return false;
  if (vname_[0] != "y") return false;
  if (vname_[0] != "z") return false;
  if (vname_[0] != "k") return false;
  if (vname_[0] != "helloWorld123") return false;
  for(size_t i = 0; i < vname_.size(); i++){
    iter = symtable_.find(vname_[i]);
    if (iter == symtable_.end()) return false;
    for (size_t j = i+1; j < vname_.size(); i++){
      assert (i!=j);
      if (symtable_[vname_[i]] == symtable_[vname_[j]]) return false;
    }
  }
  return true;
}





/*return true if prec(op1) >= prec(op2)*/
bool MathFunc::precedence(int op1, int op2){
  /*make a static array of precedences, keep size say 50*//*made a map instead*/
  /* MINUS < PLUS < TIMES < DIVIDE < POWER */
  /*SQRT and trigs involve single argument..think about it...*/
  /*I have given equal precendence to all trigs and vector ops(sum, mean...)*/
  if (precedence_[op1] >= precedence_[op2]) return true;
  return false;
}

void reverse(string &str){
  string s = "";
  for (int i = str.size()-1; i >=0 ; i--)
    s += str[i];
  str = s;
}

void MathFunc::infixToPrefix(){
  /*var inits*/
  clear();
  vector <int> operator_stack;
  status_ = FNSTRING;
  string str = fn_;
  
  /*removal of spaces*/
  string s;
  for (size_t i = 0; i < str.size() ; i++)
    if (str[i]!=' ') s += str[i];
  str = s;
  
  /*breaking into fn and expression parts*/
  size_t pos;
  pos = str.find("=");
  if (pos == string::npos) {
    error ("Could not find '='. Syntax: f(arg1, arg2...) = expression involving arg1, arg2..."); 
    return;
  }
  string args = str.substr(0, pos);
  
  
  /*store the arg names*/
  if (!storeArgNames(args)) {error("Arguments not in correct format"); return;}
  
  /*expression part*/
  s = str.substr(pos + 1);
  
  /*reverse the string and remove the blank spaces*/
  reverse(s);
  
  /*place where APPLYOVER should be added*/
    int aindex = 0;
    
  /*for each token..*/
  while(s!=""){
    /*get token type*/
    string token;
    getInfixToken(s, token);
    reverse(token);
    //cout << token << endl;
    int token_type = getTokenType(token);
    
    /*depending upon the token type*/
    switch(token_type){
      case MathFuncNames::VECTOR:
        if (token == vector_name_){
            stack_.push_back(MathFuncNames::VECTOR);
            function_.push_back(3);
        }
        break;
      /*supposed to be variable*/
      case VARIABLE:
        {
          /*check whether the string is a variable. If it is then enter the 
          index of the variable in vname_ else error*/
          int index = -1;
          for (size_t i = 0; i < vname_.size(); i++)
            if (vname_[i] == token) {index = i; break;} 
          if (index == -1) {error("The variable name " + token + " does not exist"); return;}
          stack_.push_back(index);
          function_.push_back(2);
          //cout << index  << " " << token  << " " << stack_.back() << endl;
        }
        break;
      case NUMBER:
        stack_.push_back(atof(token.c_str()));
        function_.push_back(0);
        break;
      case PLUS:
      case MINUS:
      case TIMES:
      case DIVIDE:
      case SQRT:
      case POWER:
      case SIN:
      case COS:
      case TAN:
      case ARCTAN:
      case ARCCOS:
      case ARCSIN:
      case SUM:
      case PRODUCT:
      case MEAN:
      case VARIANCE:
      case SDEV:
        
        if (operator_stack.size() == 0){
          operator_stack.push_back(token_type);
          break;
        }
        while (!precedence(token_type, operator_stack.back())){
          stack_.push_back(operator_stack.back());
          stack_.insert(stack_.begin() + aindex, APPLYOVER);
          operator_stack.pop_back();
          function_.push_back(1);
          function_.insert(function_.begin() + aindex, 1);
          if(operator_stack.size() == 0) {
            //aindex = stack_.size();
            //cout << "hiihihihi " << aindex << endl;
            break;
          }
        }
        operator_stack.push_back(token_type);
        //for (size_t  k = 0; k < stack_.size(); k++) cout << stack_[k] << " "; cout << endl;
        break;
      case LPAREN:
        while (operator_stack.back()!= RPAREN){
          stack_.push_back(operator_stack.back());
          stack_.insert(stack_.begin() + aindex, APPLYOVER);
          operator_stack.pop_back();
          function_.push_back(1);
          function_.insert(function_.begin() + aindex, 1);
          if (operator_stack.size() == 0) {error("Unmatched parentheses");return;}
        }
        operator_stack.pop_back();
        break;
      case RPAREN:
        operator_stack.push_back(token_type);
        break; 
      case ERROR:
        vname_.clear();
        v_.clear();
        stack_.clear();
        function_.clear();
        status_ = ERROR;
        error("error");
        return;
        
        break;
      default:
        assert(0);
        break;
    }
  }
  
  /*emptying the operator stack*/
  while (operator_stack.size() != 0){
    stack_.push_back(operator_stack.back());
    stack_.insert(stack_.begin() + aindex, APPLYOVER);
    operator_stack.pop_back();
    function_.push_back(1);
    function_.insert(function_.begin() + aindex, 1);
  }
  
  //for (size_t k = 0; k < stack_.size() ; k ++) cout << stack_[k] << " " ; cout << endl;
  /*reversing the output sequence*/
  for (size_t i = 0; i < stack_.size()/2; i++){
    double stemp = stack_[i]; 
    stack_[i] = stack_[stack_.size() - 1 - i];
    stack_[stack_.size() - 1 - i] = stemp;
    int ftemp = function_[i];
    function_[i] = function_[stack_.size() - 1 - i];
    function_[stack_.size() - 1 - i] = ftemp;
  }
  //for (size_t k = 0; k < stack_.size(); k++) cout << stack_[k] << " " ; cout << endl;
}


void MathFunc::clear(){
  /*its important if the executeFunction for example is called the 2nd time.*/
  /*Clear all the private variables*/
  vname_.clear();
  stack_.clear();
  function_.clear();
  symtable_.clear();
  vector_name_ = "";
  v_.clear();
  expect_ = NOTHING;
}

