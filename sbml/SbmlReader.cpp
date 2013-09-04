/*******************************************************************
 * File:            SbmlReader.cpp
 * Description:      
 * Author:          
 * E-mail:          
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef USE_SBML

#include <cmath>
#include <sbml/SBMLTypes.h>
#include <sbml/UnitDefinition.h>
#include <sbml/units/UnitFormulaFormatter.h>
#include <sbml/units/FormulaUnitsData.h>
#include <string>
#include <stdlib.h>
#include "header.h"
#include "../shell/Shell.h"

#include "SbmlReader.h"

using namespace std;
map< string,double > parmValueMap;
map< string,double> :: iterator pvm_iter;

/*  Harsha : TODO in 
    -Compartment
      --Need to add group
      --Need to deal with compartment outside
    -Molecule
      -- need todo group 
      -- Func pool and its math calculation need to be added.
 */
/* read a model into MOOSE  */
int SbmlReader::read( string filename,string location )
{
  FILE * fp = fopen( filename.c_str(), "r" );
  if ( fp == NULL){
    cout << filename << " : File does not exist." << endl;
  }
  document_ = readSBML( filename.c_str() );
  unsigned num_errors = document_->getNumErrors();
  
  if ( num_errors > 0 )
    {
      cerr << "Errors encountered while reading" << endl;
      document_->printErrors( cerr );
      errorFlag_ = true;
      return 0;
    }
  model_= document_->getModel();
  //cout << "level and version " << model_->getLevel() << model_->getVersion() << endl;
  
  if ( model_ == 0 )
    {
      cout << "No model present." << endl;
      errorFlag_ = true;
      return 0;
    }
	
  if ( !errorFlag_ )
    { 
      map< string,Id > comptIdMap;
      map< string,Id > molcomptMap;
      
      if ( !errorFlag_ )
	comptIdMap = createCompartment( location);
      if ( !errorFlag_ )
	molcomptMap = createMolecule( comptIdMap);
      if ( !errorFlag_ )
	getRules();
      if ( !errorFlag_ )
	createReaction( molcomptMap );
      if ( errorFlag_ )
	return 0;
    }
  return 1;
}

/* Pulling COMPARTMENT  */

map< string,Id > SbmlReader::createCompartment( string location)
{
  /* In compartment: pending to add 
     -- the group
     -- outside
     -- units of volume
 */
  Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
  vector< int > dims( 1, 1 );
  map< string,Id > comptIdMap;	
  map< string,string > outsideMap;
  map< string,string > ::iterator iter;
  double msize = 0.0, size = 0.0;	
  ::Compartment* compt;
  unsigned int num_compts = model_->getNumCompartments();
  for ( unsigned int i = 0; i < num_compts; i++ )
    {
      compt = model_->getCompartment(i);
      std::string id = "";
      if ( compt->isSetId() ){
	id = compt->getId();
      }
      std::string name = "";
      if ( compt->isSetName() ){
	name = compt->getName();
      } 
      std::string outside = "";
      if ( compt->isSetOutside() ){
	outside = compt->getOutside ();
      }
      if ( compt->isSetSize() ){
	msize = compt->getSize();
      }
      UnitDefinition * ud = compt->getDerivedUnitDefinition();
      size = transformUnits( msize,ud , "compartment",0);
      unsigned int dimension = compt->getSpatialDimensions();
      if (dimension < 3)
	cout << "\n ###### Spatial Dimension is " << dimension <<" volume should not be converted from liter to cubicmeter which is happening as default check \n";
      
      if (name.empty()){
	if(! id.empty() )
	  {
	    name = id;
	    cout << "Compartment name is empty so id is used";
	  }
	else
	  cout << "Compartment name and Id is empty";
      }

      Id neutral = s->doCreate( "Neutral", Id(), "SBMLtoMoose", dims );
      Id compt = s->doCreate( "CubeMesh", neutral, name, dims );
      Id meshEntry =  Neutral::child( compt.eref(), "mesh" );

      comptIdMap[id] = compt;
      if (size != 0.0)
	Field< double >::set( compt, "volume", size );

      if (dimension != 0)
	continue;
	//Field < int > :: set(compt, "numDimensions", dimension);
    }

    return comptIdMap;
}

/* create MOLECULE */
map< string,Id > SbmlReader::createMolecule( map< string,Id > &comptIdMap)
{
  Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
  vector< int > dims( 1, 1 );
  map< string, Id >molcomptMap;
  int num_species = model_->getNumSpecies();
  //cout << " num species" << num_species;
  for ( int sindex = 0; sindex < num_species; sindex++ )
    {
      Species* spe = model_->getSpecies(sindex);
      if (!spe){
	continue;
      }
      std::string compt = "";		
      if ( spe->isSetCompartment() ){
	compt = spe->getCompartment();
      }
      if (compt.length()< 1){
	//cout << "compt is empty for species "<< sindex << endl;
	continue;
      }
      string id = spe->getId();
      if (id.length() < 1){
	continue;
      }
      std::string name = "";
      if ( spe->isSetName() )
	name = spe->getName();
      else
	cout << " ";
	//cout << "Species name is empty"<< endl;
      if (name.empty())
	name = id;
		
      double initvalue =0.0;
      if ( spe->isSetInitialConcentration() )
	initvalue = spe->getInitialConcentration();
      else if ( spe->isSetInitialAmount() )
	initvalue = spe->getInitialAmount() ;
      else {
	unsigned int nr = model_->getNumRules();
	bool found = false;
	for ( unsigned int r = 0;r < nr;r++ )
	  {
	    Rule * rule = model_->getRule(r);
	    bool assignRule = rule->isAssignment();
	    if ( assignRule ){
	      string rule_variable = rule->getVariable();
	      if (rule_variable.compare(id) == 0)
		{ found = true;
		  break;
		}
	    }
	  }
	if (found == false){
	  cout << "Invalid SBML: Either initialConcentration or initialAmount must be set or it should be found in assignmentRule but non happening for " << spe->getName() <<endl;
	  return molcomptMap;
	}
      }
      Id comptEl = comptIdMap[compt];
      Id meshEntry = Neutral::child( comptEl.eref(), "mesh" );
      bool constant = spe->getConstant(); 
      Id pool;
      //If constant is true then its equivalent to BuffPool in moose

      if (constant == true)
	pool = shell->doCreate("BufPool",meshEntry,name);
      else
	pool = shell->doCreate("Pool", meshEntry, name );

      molcomptMap[id] = comptEl;
      elmtMolMap_[id] = pool;
      shell->doAddMsg( "OneToOne",pool, "mesh", meshEntry, "mesh" );
      bool bcondition = spe->getBoundaryCondition();
      if ( constant == true && bcondition == false)
	cout <<"The species "<< name << " should not appear in reactant or product as per sbml Rules"<< endl;

      unsigned int spatialDimen =Field< unsigned int >::get( comptEl, "numDimensions");
      
      UnitDefinition * ud = spe->getDerivedUnitDefinition();
      assert(ud != NULL);
      bool hasonlySubUnit = spe->getHasOnlySubstanceUnits();
      double transvalue = 0.0;

      transvalue = transformUnits(1,ud,"substance",hasonlySubUnit);
      if (hasonlySubUnit){
	// In Moose, no. of molecules (nInit) and unit is "item"
	if (spatialDimen > 0 && spe->isSetInitialAmount() ){
	  transvalue *= initvalue;
	  Field < double> :: set( pool, "nInit", transvalue);
	}
      }
      else{
	transvalue *=initvalue;
	Field <double> :: set(pool, "concInit",transvalue);
      }
    }
  return molcomptMap;
  
}
/* Assignment Rule */

void SbmlReader::getRules()
{
  unsigned int nr = model_->getNumRules();
  if (nr > 0)
    cout << "\n ##### Need to populate funcpool and sumtotal here \n";
  /*
  for ( unsigned int r = 0;r < nr;r++ ){
    
    Rule * rule = model_->getRule(r);
    bool assignRule = rule->isAssignment();
    if ( assignRule ){
      string rule_variable = rule->getVariable();
      map< string,Id >::iterator v_iter;
      map< string,Id >::iterator m_iter;
      v_iter = elmtMolMap_.find( rule_variable );

      if (v_iter != elmtMolMap_.end()){
	Id rVariable = elmtMolMap_.find(rule_variable)->second;

	cout << " \n \n variable:" << rule_variable;
	cout << " = " << rule->getFormula() << endl;
	const ASTNode * ast = rule->getMath();
	vector< string > ruleMembers;
	ruleMembers.clear();
	printMembers( ast,ruleMembers );
	for ( unsigned int rm = 0; rm < ruleMembers.size(); rm++ ){
	   m_iter = elmtMolMap_.find( ruleMembers[rm] );
	   if ( m_iter != elmtMolMap_.end() ){
	     Id rMember = elmtMolMap_.find(ruleMembers[rm])->second;
	     string test = elmtMolMap_.find(ruleMembers[rm])->first;
	     //cout << rMember << " test " <<test <<endl;
	  }
	  else{
	    cerr << "SbmlReader::getRules: Assignment rule member is not a species" << endl;
	    // uncomment this, at this time comment as biomodel9 has molecule which is not a molecule but constant which give rise error but for futher programming comminting this
	    //errorFlag_ = true;
	    }
	}
      }
    }
    bool rateRule = rule->isRate();
    if ( rateRule ){
      cout << "warning : for now Rate Rule is not handled " << endl;
      errorFlag_ = true;
    }

    bool  algebRule = rule->isAlgebraic(); 
    if ( algebRule ){
      cout << "warning: for now Algebraic Rule is not handled" << endl;
      errorFlag_ = true;
    }
  }
  */
}

//REACTION
void SbmlReader::createReaction( map< string, Id > &molcomptMap )
{ 
  vector < int > dims(1,1);
  Reaction* reac;
  map< string,double > rctMap;
  map< string,double >::iterator rctMap_iter;
  map< string,double >prdMap;
  map< string,double >::iterator prdMap_iter;
  map< string,EnzymeInfo >enzInfoMap;
  for ( unsigned int r = 0; r < model_->getNumReactions(); r++ ){
    Id reaction_;
    reac = model_->getReaction( r ); 
    
    std:: string id; //=reac->getId();
    if ( reac->isSetId() )
      id = reac->getId();
    std::string name;
    
    if ( reac->isSetName() )
      name = reac->getName();
     
    if (name.empty()){
      if (id.empty())
	assert("Reaction id and name is empty");
      else
	name = id;
    }
    string grpname = getAnnotation( reac,enzInfoMap );
    if ( (grpname != "") && (enzInfoMap[grpname].stage == 3) )
      setupEnzymaticReaction( enzInfoMap[grpname],grpname ,molcomptMap);
    else if ( grpname == "" )
      { 
	if (reac->getNumModifiers() > 0)
	  setupMMEnzymeReaction( reac,id ,molcomptMap);
	else{
	  bool rev=reac->getReversible();
	  bool fast=reac->getFast();
	  if ( fast ){
	    cout<<"warning: for now fast attribute is not handled"<<endl;
	    errorFlag_ = true;
	  }
	  int numRcts = reac->getNumReactants();
	  int numPdts = reac->getNumProducts();
	  
	  if ( numRcts == 0 && numPdts != 0 ){
	    const SpeciesReference* pdt = reac->getProduct( 0 );
	    std::string spName = pdt->getSpecies();     
	    Id parent = molcomptMap.find( spName )->second; //gives compartment of spName
	    cout << " \n \t ################################# Sub = 0 and prd != 0 need to the reac ############### ";
	    const SpeciesReference* rect=reac->getReactant(0);
	    std::string sp=rect->getSpecies();
	    Id comptRef = molcomptMap.find(sp)->second; //gives compartment of sp
	    Id meshEntry = Neutral::child( comptRef.eref(), "mesh" );
	    Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	    reaction_ = shell->doCreate("Reac", meshEntry, name, dims);
	    shell->doAddMsg( "Single", meshEntry, "remeshReacs", reaction_, "remesh");
	    //Get Substrate 
	    addSubPrd(reac,reaction_,"prd");
	  } //if numRcts == 0
	  else {
	    const SpeciesReference* rect=reac->getReactant(0);
	    std::string sp=rect->getSpecies();
	    Id comptRef = molcomptMap.find(sp)->second; //gives compartment of sp
	    Id meshEntry = Neutral::child( comptRef.eref(), "mesh" );
	    Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	    reaction_ = shell->doCreate("Reac", meshEntry, name, dims);
	    shell->doAddMsg( "Single", meshEntry, "remeshReacs", reaction_, "remesh");
	    //Get Substrate 
	    addSubPrd(reac,reaction_,"sub");
	    //Get Product
	    addSubPrd(reac,reaction_,"prd");
	  }
	  if ( reac->isSetKineticLaw() ){
	    KineticLaw * klaw=reac->getKineticLaw();
	    //vector< double > rate = getKLaw( klaw,rev );
	    vector< double > rate;
	    rate.clear();
	    getKLaw( klaw,rev,rate );
	    if ( errorFlag_ )
	      return;
	    else if ( !errorFlag_ ){
	      Field < double > :: set( reaction_, "kf", rate[0] ); 
	      Field < double > :: set( reaction_, "kb", rate[1] );	
	    }
	  } //issetKineticLaw
	  } //else
      } // else grpname == ""
  }//for unsigned
} //reaction

/* Enzymatic Reaction */
void SbmlReader::setupEnzymaticReaction( const EnzymeInfo & einfo,string name, map< string, Id > &molcomptMap){
  Id enzyme = einfo.enzyme;
  string enzname = Field<string>::get(ObjId(enzyme),"name");
  Id comptRef = molcomptMap.find(enzname)->second; //gives compartment of sp
  Id meshEntry = Neutral::child( comptRef.eref(), "mesh" );

  Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
  vector < int > dims(1,1);

  Id enzyme_ = shell->doCreate("Enz", meshEntry, name, dims);
  shell->doAddMsg( "Single", meshEntry, "remeshReacs", enzyme_, "remesh");

  Id complex = einfo.complex;
  shell->doAddMsg("OneToOne",enzyme_,"cplx",complex,"reac");

  vector< Id >::const_iterator sub_itr;
  for ( sub_itr = einfo.substrates.begin(); sub_itr != einfo.substrates.end(); sub_itr++ )
    { Id S = (*sub_itr);
      Id b = shell->doAddMsg( "OneToOne", enzyme_, "sub" ,S , "reac" );
    }
 
  vector< Id >::const_iterator prd_itr;
  for ( prd_itr = einfo.products.begin(); prd_itr != einfo.products.end(); prd_itr++ )
    {	Id P = (*prd_itr);
      shell->doAddMsg ("OneToOne",enzyme_,"prd", P,"reac");
      
    }
  Field < double > :: set( enzyme_, "k1", einfo.k1 ); 
  Field < double > :: set( enzyme_, "k2", einfo.k2 ); 
  Field < double > :: set( enzyme_, "k3", einfo.k3 ); 
}

/* get annotation */
string SbmlReader::getAnnotation( Reaction* reaction,map<string,EnzymeInfo> &enzInfoMap )
{
  XMLNode * annotationNode = reaction->getAnnotation();
  EnzymeInfo einfo;
  string grpname = "",stage;
  
  if( annotationNode != NULL ){	
    unsigned int num_children = annotationNode->getNumChildren();
    //cout<<"num of children :"<< num_children << endl;
    for( unsigned int child_no = 0; child_no < num_children; child_no++ ){
      XMLNode childNode = annotationNode->getChild( child_no );
      unsigned int num_grand_children = childNode.getNumChildren();
      //cout << "child no " << child_no << ": name:" << childNode.getName() << "has num_grand_children =" << num_grand_children << endl; 
      for( unsigned int gchild_no = 0; gchild_no < num_grand_children; gchild_no++ ){
	XMLNode &grandChildNode = childNode.getChild( gchild_no );
	if ( grandChildNode.getPrefix() == "moose" && grandChildNode.getName() == "EnzymaticReaction" ){
	  //cout << " \n here " << grandChildNode.getName();
	  unsigned int num_ggchildren = grandChildNode.getNumChildren();
	  //cout<<"num of grand children: "<<num_ggchildren<<endl; 
	  for( unsigned int ggchild_no = 0; ggchild_no < num_ggchildren; ggchild_no++ ){
	    //cout<<"iteration at"<<ggchild_no<<endl;
	    XMLNode &greatGrandChildNode = grandChildNode.getChild( ggchild_no );
	    string nodeName = greatGrandChildNode.getName();
	    //cout << "nodename:" << nodeName << endl;
	    string nodeValue;
	    if (greatGrandChildNode.getNumChildren() == 1 ){
	      nodeValue = greatGrandChildNode.getChild(0).toXMLString();
	      
	    } 
	    else {
	      cout << "Error: expected exactly ONE child of " << nodeName << endl;
	    }
	    
	    if ( nodeName == "enzyme" )
		einfo.enzyme=elmtMolMap_.find(nodeValue)->second; 

	    else if ( nodeName == "complex" )
	      einfo.complex=elmtMolMap_.find(nodeValue)->second; 

	    else if ( nodeName == "substrates"){
	      Id elem = elmtMolMap_.find(nodeValue)->second; 
	      einfo.substrates.push_back(elem);
	    }
	    else if ( nodeName == "products" ){
	      Id elem = elmtMolMap_.find(nodeValue)->second; 
	      einfo.products.push_back(elem);
	    }
	    else if ( nodeName == "groupName" )
	      grpname = nodeValue;
	    
	    else if ( nodeName == "stage" )
	      stage = nodeValue;
	  }
	  // Stage == '1' means E+S -> ES*;
	  if ( stage == "1" ){
	    enzInfoMap[grpname].substrates = einfo.substrates;
	    enzInfoMap[grpname].complex = einfo.complex;
	    enzInfoMap[grpname].enzyme = einfo.enzyme;
	    einfo.stage = 1;
	    enzInfoMap[grpname].stage = einfo.stage;
	    //cout << "stage:" << enzInfoMap[grpname].stage << endl;
	    KineticLaw * klaw=reaction->getKineticLaw();
	    vector< double > rate ;
	    rate.clear();						
	    getKLaw( klaw,true,rate );
	    if ( errorFlag_ )
	      exit(0);
	    else if ( !errorFlag_ ){
	      enzInfoMap[grpname].k1 = rate[0];
	      enzInfoMap[grpname].k2 = rate[1];
	    }
	  }
	  //Stage =='2' means ES* -> E+P;
	  else if ( stage == "2" ){
	    enzInfoMap[grpname].products = einfo.products;
	    einfo.stage = 2;
	    enzInfoMap[grpname].stage += einfo.stage;
	    //cout << "stage:" << enzInfoMap[grpname].stage << endl;
	    KineticLaw * klaw=reaction->getKineticLaw();
	    vector< double > rate;	
	    rate.clear();
	    getKLaw( klaw,false,rate );
	    if ( errorFlag_ )
	      exit(0);
	    else if ( !errorFlag_ )
	      enzInfoMap[grpname].k3 = rate[0];
	  } 
	}//if line 440
      }
    }
  }
  return grpname;
}

/*    set up Michalies Menten reaction  */
void SbmlReader::setupMMEnzymeReaction( Reaction * reac,string rid,map< string, Id > &molcomptMap )
{string::size_type loc = rid.find( "_MM_Reaction_" );
  if( loc != string::npos ){
    int strlen = rid.length(); 
    rid.erase( loc,strlen-loc );
  }
  
  unsigned int num_modifr = reac->getNumModifiers();
  for ( unsigned int m = 0; m < num_modifr; m++ )
    { 
      const ModifierSpeciesReference* modifr=reac->getModifier( m );
      std::string sp = modifr->getSpecies();
      Id enzyme_;
      Id E = elmtMolMap_.find(sp)->second;
      
      Id comptRef = molcomptMap.find(sp)->second; //gives compartment of sp
      Id meshEntry = Neutral::child( comptRef.eref(), "mesh" );
      vector < int > dims(1,1);
      Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
      enzyme_ = shell->doCreate("MMenz",E,rid,dims);
      shell->doAddMsg( "Single", meshEntry, "remeshReacs", enzyme_, "remesh");
      shell->doAddMsg("Single",E,"nOut",enzyme_,"enzDest");

      KineticLaw * klaw=reac->getKineticLaw();      
      vector< double > rate;
      rate.clear();
      getKLaw( klaw,true,rate );
      if ( errorFlag_ )
	return;
      else if ( !errorFlag_ ){
	for ( unsigned int rt = 0; rt < reac->getNumReactants(); rt++ )
	  {const SpeciesReference* rct = reac->getReactant( rt );
	    sp=rct->getSpecies();
	    Id S = elmtMolMap_.find(sp)->second;
	    shell->doAddMsg( "OneToOne", enzyme_, "sub" ,S , "reac" );
	  }
	for ( unsigned int pt = 0; pt < reac->getNumProducts(); pt++ )
	  {
	    const SpeciesReference* pdt = reac->getProduct(pt);
	    sp = pdt->getSpecies();
	    Id P = elmtMolMap_.find(sp)->second;
	    shell->doAddMsg( "OneToOne", enzyme_, "prd" ,P, "reac" );
	  }
	Field < double > :: set( enzyme_, "kcat", rate[0] );
	Field < double > :: set( enzyme_, "Km", rate[1] );
      }
      
    }
}

/*    get Parameters from Kinetic Law */
void SbmlReader::getParameters( const ASTNode* node,vector <string> & parameters )
{
  assert( parameters.empty() );	
  
  if ( node->getType() == AST_MINUS ){
    const ASTNode* lchild = node->getLeftChild();  
    pushParmstoVector( lchild,parameters );
    
    if ( parameters.size() == 1 ) {
      const ASTNode* rchild = node->getRightChild();
      pushParmstoVector( rchild,parameters );		
    }
    
  }
  
  else if ( node->getType() == AST_DIVIDE ){
    const ASTNode* lchild = node->getLeftChild();  
    pushParmstoVector( lchild,parameters );
    if (( parameters.size() == 1 ) || ( parameters.size() == 0 )) {
      const ASTNode* rchild = node->getRightChild();
      pushParmstoVector( rchild,parameters );	
    }
  }
  else if ( node->getType() == AST_TIMES )
    pushParmstoVector( node,parameters );
  else if ( node->getType() == AST_PLUS )
    pushParmstoVector( node,parameters );
  else if ( node->getType() == AST_NAME )
    pushParmstoVector( node,parameters );
  if ( parameters.size() > 2 ){
    cout<<"Sorry! for now MOOSE cannot handle more than 2 parameters ."<<endl;
    errorFlag_ = true;
  }
  		
}
/*   push the Parameters used in Kinetic law to a vector */

void SbmlReader::pushParmstoVector(const ASTNode* p,vector <string> & parameters)
{string parm = "";
 if ( p->getType() == AST_NAME ){
    pvm_iter = parmValueMap.find( std::string(p->getName()) );
    if ( pvm_iter != parmValueMap.end() ){
      parm = pvm_iter->first;
      parameters.push_back( parm );
    }
  }
  int num = p->getNumChildren();
  for( int i = 0; i < num; ++i ){  
    const ASTNode* child = p->getChild(i);
    pushParmstoVector( child,parameters );
  }
}
/*     get Kinetic Law */
void SbmlReader::getKLaw( KineticLaw * klaw,bool rev,vector< double > & rate )
{ 							
  std::string id;
  double value = 0.0;
  UnitDefinition * kfud;
  UnitDefinition * kbud;
  int np = klaw->getNumParameters();
  bool flag = true;
  
  for ( int pi = 0; pi < np; pi++ ){
    Parameter * p = klaw->getParameter(pi);
    
    if ( p->isSetId() )
      id = p->getId();
    if ( p->isSetValue() )		
      value = p->getValue();
    parmValueMap[id] = value;
    flag = false;
    }
  double kf = 0.0,kb = 0.0,kfvalue,kbvalue;
  string kfparm,kbparm;
  vector< string > parameters;
  parameters.clear();	
  const ASTNode* astnode=klaw->getMath();
  //cout << "\nkinetic law is :" << SBML_formulaToString(astnode) << endl;
  getParameters( astnode,parameters );
  //cout << "getKLaw " << errorFlag_;
   if ( errorFlag_ )
    return;
  else if ( !errorFlag_ )
    {
      if ( parameters.size() == 1 ){
	kfparm = parameters[0];
	kbparm = parameters[0];
      }
      else if ( parameters.size() == 2 ){
	kfparm = parameters[0];
	kbparm = parameters[1];
      }
      //cout << "\n parameter "<< parameters.size();
      //cout << "$$ "<< parmValueMap[kfparm];
      //cout << " \t \t " << parmValueMap[kbparm];
      
      kfvalue = parmValueMap[kfparm];
      kbvalue = parmValueMap[kbparm];
      Parameter* kfp;
      Parameter* kbp;
      if ( flag ){
	kfp = model_->getParameter( kfparm );
	kbp = model_->getParameter( kbparm );
      }
      else{
	kfp = klaw->getParameter( kfparm );
	kbp = klaw->getParameter( kbparm );
      }
      //cout << "\t \n \n" << kfp << " " <<kbp;
      
      if ( kfp->isSetUnits() ){
	kfud = kfp->getDerivedUnitDefinition();
	//cout << "parameter unit :" << UnitDefinition::printUnits(kfp->getDerivedUnitDefinition())<< endl;
	double transkf = transformUnits( 1,kfud ,"substance",true);	
	//cout<<"parm kf trans value : "<<transkf<<endl;
	//cout<<"kfvalue :"<<kfvalue<<endl;
	kf = kfvalue * transkf;
	kb = 0.0;
      }
      else if (! kfp->isSetUnits() ){
	kf = kfvalue;
	kb = 0.0;
      }
      if ( ( kbp->isSetUnits() ) && ( rev ) ){ 
	kbud = kbp->getDerivedUnitDefinition();	
	//cout << "parameter unit :" << UnitDefinition::printUnits(kbp->getDerivedUnitDefinition()) << endl;
	double transkb = transformUnits( 1,kbud,"substance",true );
	//cout<<"parm kb trans value : "<<transkb<<endl;
	//cout<<"kbvalue :"<<kbvalue<<endl;
	kb = kbvalue * transkb;	
      }			
      if ( (! kbp->isSetUnits() ) && ( rev ) ){
	kb = kbvalue;
      }
      rate.push_back( kf );
      rate.push_back( kb );
      
    }
  
} 

void SbmlReader::addSubPrd(Reaction * reac,Id reaction_,string type)
{ map< string,double > rctMap;
  map< string,double >::iterator rctMap_iter;
  double rctcount=0.0;	
  Shell * shell = reinterpret_cast< Shell* >( Id().eref().data() );
  rctMap.clear();
  unsigned int nosubprd;
  const SpeciesReference* rct;
  if (type == "sub")
    nosubprd = reac->getNumReactants();
  else
    nosubprd = reac->getNumProducts();
  for ( unsigned int rt=0;rt<nosubprd;rt++ ){	
    if (type == "sub")
      rct = reac->getReactant(rt);
    else
      rct = reac->getProduct(rt);
    std:: string sp = rct->getSpecies();
    rctMap_iter = rctMap.find(sp);			
    if ( rctMap_iter != rctMap.end() )	
      rctcount = rctMap_iter->second;
    else
      rctcount = 0.0;
    
    rctcount += rct->getStoichiometry();
    rctMap[sp] = rctcount;
    for ( int i=0;(int)i<rct->getStoichiometry();i++ )
      shell->doAddMsg( "OneToOne", reaction_, type ,elmtMolMap_[sp] , "reac" );
  }
}
/* Transform units from SBML to MOOSE 
   MOOSE units for
   volume -- cubic meter
*/

double SbmlReader::transformUnits( double mvalue,UnitDefinition * ud,string type, bool hasonlySubUnit )
{
  double lvalue = mvalue;
  if (type == "compartment"){ 
    for ( unsigned int ut = 0; ut < ud->getNumUnits(); ut++ )
      {
	Unit * unit = ud->getUnit(ut);
	double exponent = unit->getExponent();
	double multiplier = unit->getMultiplier();
	int scale = unit->getScale();
	double offset = unit->getOffset(); 
	lvalue *= pow( multiplier * pow(10.0,scale), exponent ) + offset;
	// Need to check if spatial dimension is less than 3 then, 
	// then volume conversion e-3 to convert cubicmeter shd not be done.
	if ( unit->isLitre() ){
	  lvalue *= pow(1e-3,exponent);
	  return lvalue;
	}
      }
  }
  else if(type == "substance"){
    for ( unsigned int ut = 0; ut < ud->getNumUnits(); ut++ ){
      Unit * unit = ud->getUnit(ut);
      //cout << " :) " << UnitKind_toString(unit->getKind());
      if ( unit->isMole() ){
	double exponent = unit->getExponent();
	double multiplier = unit->getMultiplier();
	int scale = unit->getScale();
	double offset = unit->getOffset(); 
	lvalue *= pow( multiplier * pow(10.0,scale), exponent ) + offset;
	if (hasonlySubUnit)
	  // if hasonlySubstanceUnit is true, then unit is subtance
	  // In Moose nInit = no. of molecules( unit is items)
	  // no. of molecules (items) = mole * Avogadro no.
	  // In SBML if initial Amount is set to mole then convert from mole to items (#)
	  lvalue *= pow( NA ,exponent);
	else
	  // if hasonlySubstanceUnit is false, then unit is in substance/size = Molar
	  //Then convert Molar to milli Molar for moose as concentration units are in milliMolar
	  lvalue *= pow(1e+3,exponent); 
	return lvalue;
      }
      else if(unit->isItem())
	return lvalue;
      else if(unit->isSecond())
	return lvalue;
      else {
	cout << "check this units type " <<UnitKind_toString(unit->getKind());
	return lvalue;
      }
    }
  }
  return lvalue;
}
void SbmlReader::printMembers( const ASTNode* p,vector <string> & ruleMembers )
{
  if ( p->getType() == AST_NAME ){
    //cout << "_NAME" << " = " << p->getName() << endl;
    ruleMembers.push_back( p->getName() );
  }
  int num = p->getNumChildren();
  for( int i = 0; i < num; ++i )
    {  
      const ASTNode* child = p->getChild(i);
      printMembers( child,ruleMembers );
    }
} 

#endif // USE_SBML
