/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <cmath>
#include <sbml/SBMLTypes.h>
#include <sbml/UnitDefinition.h>
#include <sbml/units/UnitFormulaFormatter.h>
#include <sbml/units/FormulaUnitsData.h>
#include "element/Neutral.h"
#include "kinetics/KinCompt.h"
#include "kinetics/Enzyme.h"
#include "SbmlReader.h"

using namespace std;
map<string,double>parmValueMap;
map<string,double>::iterator pvm_iter;
map<string,string>parmUnitMap;

//function to get the parameter used in the kinetic law
void SbmlReader::prn_parm(const ASTNode* p,vector <string> & parameters)
{
	string parm = "";	
	if ( p->getType() == AST_NAME){
        	cout << "_NAME" << " = " << p->getName() << endl;
	   	pvm_iter = parmValueMap.find(std::string(p->getName()));			
	    	if (pvm_iter != parmValueMap.end()){
			parm = pvm_iter->first;
			parameters.push_back(parm);
		}
	}
       	int num = p->getNumChildren();
    	for( int i = 0; i < num; ++i )
    	{  
        	const ASTNode* child = p->getChild(i);
       	 	prn_parm(child,parameters);
        	
    	}
    	
 } 
/* function to get members in the mathml of  rule */ 
void SbmlReader::printMembers(const ASTNode* p,vector <string> & ruleMembers)
{
	if ( p->getType() == AST_NAME ){
	   	cout << "_NAME" << " = " << p->getName() << endl;
	   	ruleMembers.push_back(p->getName());
	}
	
       	int num = p->getNumChildren();
    	for( int i = 0; i < num; ++i )
    	{  
        	const ASTNode* child = p->getChild(i);
       	 	printMembers(child,ruleMembers);
        	
    	}
    	
 } 

/* function to read a model into MOOSE  */
void SbmlReader::read(string filename,Id location)
{
	FILE * fp = fopen(filename.c_str(), "r");
	if ( fp == NULL){
		cout << filename << " : File does not exist." << endl;
	}
	document_ = readSBML(filename.c_str());
	unsigned num_errors = document_->getNumErrors();
	
	if ( num_errors > 0)
	{
		cerr << "Errors encountered " << endl;
		document_->printErrors(cerr);
		return;
	}
	model_= document_->getModel();
	if (model_ == 0)
	{
		cout << "No model present." << endl;
	}
	if (!model_->isSetId()){
		cout << "Id not set." << endl;
	}
	createCompartment(location);
}

/* function to transform units */
double SbmlReader::transformUnits(double mvalue,UnitDefinition * ud)
{	cout<<"num units :"<<ud->getNumUnits()<<endl;
	double lvalue = mvalue;
	//cout<<"derived unit defn:"<<UnitDefinition::printUnits(prm->getDerivedUnitDefinition())<<endl;
	for (int ut=0;ut<ud->getNumUnits();ut++)
	{
		Unit * unit=ud->getUnit(ut);
		double exponent=unit->getExponent();
		cout<<"exponent  :"<<exponent<<endl;
		double multiplier=unit->getMultiplier();
		cout<<"multiplier :"<<multiplier<<endl;
		int scale=unit->getScale();
		cout<<"scale :"<<scale<<endl;
		double ofset=unit->getOffset(); 
		lvalue *= pow( multiplier * pow(10.0,scale), exponent ) + ofset;
		cout<<"lvalue "<<lvalue<<endl;
		if (unit->isLitre()){
			cout<<"unit is litre";
			lvalue *= pow(1e-3,exponent);
		}
		if (unit->isMole()){
			cout<<"unit is mole"<<endl;
			lvalue *= pow(6.02214199e23,exponent);	
		}
	}
	cout<<"value before return "<<lvalue<<endl;
	return lvalue;
}
/* create COMPARTMENT  */
void SbmlReader::createCompartment(Id location)
{
	static const Cinfo* kincomptCinfo = initKinComptCinfo();
	static const Finfo* sizeFinfo = kincomptCinfo->findFinfo( "size" );
	static const Finfo* dimensionFinfo = kincomptCinfo->findFinfo( "numDimensions" );
	map<string,Id> idMap;	
	Id outcompt; //outside compartment	
	double msize = 0.0,size=0.0;	
	::Compartment* compt;
	for (int i=0;i<model_->getNumCompartments();i++)
	{
		compt = model_->getCompartment(i);
		std::string id;
		if (compt->isSetId()){
			id = compt->getId();
		}
		std::string name;
		if (compt->isSetName()){
			name = compt->getName();
		} 
		std::string type;
		if (compt->isSetCompartmentType()){
			type =compt->getCompartmentType ();
		}
		std::string outside;
		if ( compt->isSetOutside()){
			outside =compt->getOutside ();
			
		}
		if (compt->isSetSize()){
			msize =compt->getSize();
		}
		UnitDefinition * ud=compt->getDerivedUnitDefinition();
		cout<<"compartment unit:"<<UnitDefinition::printUnits(compt->getDerivedUnitDefinition())<<endl;
		size=transformUnits(msize,ud);
		//cout<<"size returned from function is : "<<size<<endl;
		
		unsigned int dimension=compt->getSpatialDimensions();
		
		if (outside==""){		
			comptEl_ = Neutral::create( "KinCompt",id, location, Id::scratchId() ); //create Compartment 
			idMap[id]=comptEl_->id(); 
		}
		else{
			outcompt=idMap.find(outside)->second;
			comptEl_ = Neutral::create( "KinCompt",id, outcompt, Id::scratchId() ); //create Compartment inside 
			idMap[id]=comptEl_->id();
			static const Finfo* outsideFinfo = kincomptCinfo->findFinfo( "outside" );
			static const Finfo* insideFinfo = kincomptCinfo->findFinfo( "inside" );
			Eref(comptEl_ ).add(outsideFinfo->msg(),outcompt(),insideFinfo->msg(),ConnTainer::Default);
		}
		if (size != 0.0){
			
                    set< double >( comptEl_, sizeFinfo, size );
                                        
		}
		if (dimension != 0){
			
			set< unsigned int >( comptEl_,dimensionFinfo,dimension );		
		}
	}
	createMolecule(idMap);
}

/* create MOLECULE */
void SbmlReader::createMolecule(map<string,Id> &idMap)
{	
	map<string,Id>molMap;
	map<string,string>cmptMap;
	static const Cinfo* moleculeCinfo = initMoleculeCinfo();
	static const Finfo* modeFinfo = moleculeCinfo->findFinfo( "mode" );
	static const Finfo* nInitFinfo = moleculeCinfo->findFinfo( "nInit" );	
	//static const Finfo* concInitFinfo = moleculeCinfo->findFinfo( "concInit" );
	static const Cinfo* kincomptCinfo = initKinComptCinfo();
	static const Finfo* dimensionFinfo = kincomptCinfo->findFinfo( "numDimensions" );
	static const Finfo* sizeFinfo = kincomptCinfo->findFinfo( "size" );
	
	for (unsigned int m=0;m<model_->getNumSpecies();m++)
	{
		Species* s = model_->getSpecies(m);
		std::string compt;		
		if (s->isSetCompartment())		
			compt = s->getCompartment();
		string id=s->getId();
		cout<<"species is :"<<id<<endl;
		cmptMap[id]=compt;
		Id comptEl=idMap[compt];
		molecule_= Neutral::create( "Molecule",id,comptEl,Id::scratchId() );//create Molecule
		molMap[id] = comptEl; 
		elmtMap_[id] = Eref(molecule_);
		//printNotes(s);
		UnitDefinition * ud = s->getDerivedUnitDefinition();
		cout<<"species unit :"<<UnitDefinition::printUnits(s->getDerivedUnitDefinition())<<endl;
		double initvalue;
		if (s->isSetInitialConcentration())
			initvalue = s->getInitialConcentration();
		if (s->isSetInitialAmount())
			initvalue = s->getInitialAmount() ;
		double transvalue = transformUnits(1,ud);
		transvalue *= initvalue;
		//bool has_subunits = s->getHasOnlySubstanceUnits();
		//cout<<"has_sub "<<has_subunits<<endl;
		unsigned int dimension;
                get< unsigned int >(comptEl.eref(), dimensionFinfo,dimension);
		if (dimension > 0 && s->isSetInitialConcentration() ) { //has_subunits == false ){
					
			double size;
			get< double > (comptEl.eref(),sizeFinfo,size); 			
			transvalue *= size;			
		}
		cout<<"n init is :"<<transvalue<<endl;
		set< double >(molecule_, nInitFinfo, transvalue); //initialAmount 	
		bool cons=s->getConstant(); 
		bool bcondition = s->getBoundaryCondition();
		if ((cons = true) && (bcondition = true))
			set< int >(molecule_,modeFinfo,4); //getConstant=True indicates a buffered molecule
		else if ((cons = false) && (bcondition = true))
			set< int >(molecule_,modeFinfo,1); //indicates the species has assignment rule
		else 
			set< int >(molecule_,modeFinfo,0);
	}
	getRules();
	createReaction(molMap);
}

void SbmlReader::getRules()
{
	unsigned int nr = model_->getNumRules();
	cout<<"no of rules:"<<nr<<endl;
	for (int r=0;r<nr;r++)
	{
		Rule * rule =model_->getRule(r);
		cout<<"rule :"<<rule->getFormula()<<endl;
		bool assignRule = rule->isAssignment();
		cout<<"is assignment :"<<assignRule<<endl;
		if (assignRule){
			string rule_variable = rule->getVariable();
			cout<<"variable :"<<rule_variable<<endl;
			Eref rVariable = elmtMap_.find(rule_variable)->second;
			const ASTNode * ast=rule->getMath();
			vector< string > ruleMembers;
			printMembers(ast,ruleMembers);
			for (int rm=0;rm<ruleMembers.size();rm++)
			{
				Eref rMember = elmtMap_.find(ruleMembers[rm])->second;				
				rMember.add("nSrc",rVariable,"sumTotal",ConnTainer::Default); 	
						
			}
			
		}
	}
}

/* print annotation */
string SbmlReader::printAnnotation(SBase *sb,map<string,EnzymeInfo> &enzInfoMap)
{
	XMLNode * annotationNode = sb->getAnnotation();
	EnzymeInfo einfo;
	string grpname = "",stage;
	if( annotationNode != NULL )
	{	cout<<"num of children :"<<annotationNode->getNumChildren()<<endl;
		for( int l = 0; l < annotationNode->getNumChildren(); l++ )
		{
			XMLNode &childNode = annotationNode->getChild( l );
			for( int m = 0; m < childNode.getNumChildren(); m++ )
			{
				XMLNode &grandChildNode = childNode.getChild( l );
				if (grandChildNode.getPrefix() == "moose" && grandChildNode.getName() == "EnzymaticReaction")
				{	for( int n = 0; n < grandChildNode.getNumChildren(); n++ )
					{
						XMLNode &greatGrandChildNode = grandChildNode.getChild( n );
						Eref elem;
						if (greatGrandChildNode.getName() == "substrates"){
							string str = greatGrandChildNode.getChild(0).toXMLString();
							elem=elmtMap_.find(str)->second; 
							einfo.substrates.push_back(elem.id());
							
						}
						else if (greatGrandChildNode.getName() == "products"){
							string str = greatGrandChildNode.getChild(0).toXMLString();
							elem=elmtMap_.find(str)->second; 
							einfo.products.push_back(elem.id());
							
						}
						else if (greatGrandChildNode.getName() == "complex"){
							string str = greatGrandChildNode.getChild(0).toXMLString();
							elem=elmtMap_.find(str)->second; 							
							einfo.complex=elem.id();
						}
						else if (greatGrandChildNode.getName() == "enzyme"){
							string str = greatGrandChildNode.getChild(0).toXMLString();
							elem=elmtMap_.find(str)->second; 
							einfo.enzyme=elem.id();
						}
						else if (greatGrandChildNode.getName() == "groupName"){
							grpname = greatGrandChildNode.getChild(0).toXMLString();
							einfo.groupName=grpname;
						}
						else if (greatGrandChildNode.getName() == "stage"){
							stage = greatGrandChildNode.getChild(0).toXMLString();
															
						}
					}
					if (stage == "1"){
						enzInfoMap[grpname].substrates = einfo.substrates;
						enzInfoMap[grpname].complex = einfo.complex;
						enzInfoMap[grpname].enzyme = einfo.enzyme;
						einfo.stage = 1;
						enzInfoMap[grpname].stage = einfo.stage;
						cout<<"stage:"<<enzInfoMap[grpname].stage <<endl;
						
					}					
					else if (stage == "2"){
						enzInfoMap[grpname].products = einfo.products;
						einfo.stage = 2;
						enzInfoMap[grpname].stage += einfo.stage;
						cout<<"stage:"<<enzInfoMap[grpname].stage<<endl;
					}
				}
			}
		}
	}
	return grpname;
}

/* Enzymatic Reaction */
void SbmlReader::setupEnzymaticReaction(const EnzymeInfo & einfo)
{
	
	Eref E = einfo.enzyme();

	Element* enzyme_ = Neutral::create( "Enzyme","kenz",E.id(),Id::scratchId() );//create Enzyme
	set< bool >(enzyme_,"mode",0);
	Eref complx = einfo.complex(); 
	Eref(enzyme_).add("enz",E,"reac",ConnTainer::Default); 
 	Eref(enzyme_).add("cplx",complx,"reac",ConnTainer::Default); 
	vector<Id>::const_iterator sub_itr;
	for (sub_itr = einfo.substrates.begin(); sub_itr != einfo.substrates.end(); sub_itr++)
	{	Eref S = (*sub_itr)();
		Eref(enzyme_).add("sub",S,"reac",ConnTainer::Default); 

	}
	vector<Id>::const_iterator prd_itr;
	for (prd_itr = einfo.products.begin(); prd_itr != einfo.products.end(); prd_itr++)
	{	Eref P = (*prd_itr)();
		Eref(enzyme_).add("prd",P,"prd",ConnTainer::Default);

	}
	set(complx,"destroy");
}
void SbmlReader::setupMMEnzymeReaction(Reaction * reac)
{
	for (int m=0;m<reac->getNumModifiers();m++)
	{	
		const ModifierSpeciesReference* modfr=reac->getModifier(m);
		string sp = modfr->getSpecies();
		Eref E = elmtMap_.find(sp)->second;
		Element* enzyme_ = Neutral::create( "Enzyme","kenz",E.id(),Id::scratchId() );//create Enzyme
		set< bool >(enzyme_,"mode",1);
		Eref(enzyme_).add("enz",E,"reac",ConnTainer::Default); 
		for (int rt=0;rt<reac->getNumReactants();rt++)
		{	
			const SpeciesReference* rct=reac->getReactant(rt);
			sp=rct->getSpecies();
			Eref S = elmtMap_.find(sp)->second;
			Eref(enzyme_).add("sub",S,"reac",ConnTainer::Default); 
		}
		for (int pt=0;pt<reac->getNumProducts();pt++)
		{
			const SpeciesReference* pdt=reac->getProduct(pt);
			sp=pdt->getSpecies();
			Eref P = elmtMap_.find(sp)->second;
			Eref(enzyme_).add("prd",P,"prd",ConnTainer::Default);
		}

	}	
}
//print PARAMETERS
void SbmlReader::printParameter()
{	
	//map<string,double>parmValueMap;
	//map<string,double>::iterator pvm_iter;
	//map<string,string>parmUnitMap;
	for (int pm=0;pm<model_->getNumParameters();pm++)
	{
		Parameter* prm=model_->getParameter(pm);
		std::string id,unit;
		if (prm->isSetId()){
			id = prm->getId();
		}
		double value;		
		if (prm->isSetValue()){		
			value=prm->getValue();	
		}
		parmValueMap[id]=value;
				
	}
	cout<<"inside model->parameter()"<<endl;
}
//create REACTION
void SbmlReader::createReaction(map<string,Id> &molMap)
{	
	map<string,double>rctMap;
	map <string, double>::iterator rctMap_iter;
	map<string,double>pdtMap;
	map <string, double>::iterator pdtMap_iter;
	map<string,Eref>::iterator elemt_iter;
	map<string,EnzymeInfo> enzInfoMap;
	double rctorder,pdtorder;
	static const Cinfo* moleculeCinfo = initMoleculeCinfo();
	static const Finfo* reacFinfo =moleculeCinfo->findFinfo( "reac" );	
	static const Cinfo* reactionCinfo = initReactionCinfo();
	static const Finfo* subFinfo = reactionCinfo->findFinfo( "sub" );
	static const Finfo* prdFinfo = reactionCinfo->findFinfo( "prd" );
	static const Finfo* kfFinfo = reactionCinfo->findFinfo( "kf" );	
	static const Finfo* kbFinfo = reactionCinfo->findFinfo( "kb" );	
	Reaction* reac;	
	for (int r=0;r<model_->getNumReactions();r++)
	{	
		reac=model_->getReaction(r); 
		const string id=reac->getId();
		cout<<"reaction is "<<id<<endl;
		string grpname = printAnnotation(reac,enzInfoMap);
		if ((grpname != "") && (enzInfoMap[grpname].stage == 3))
			setupEnzymaticReaction(enzInfoMap[grpname]);
		else if (grpname == "")
		{
			if (reac->getNumModifiers()> 0)
				 setupMMEnzymeReaction(reac);
			else{
				bool rev=reac->getReversible();
				//bool fast=reac->getFast();
				cout<<"is rev"<<rev<<endl;  
				const SpeciesReference* rect=reac->getReactant(0);
				std::string sp=rect->getSpecies();
				Id m=molMap.find(sp)->second; //gives compartment of sp
				reaction_=Neutral::create( "Reaction",id,m,Id::scratchId() ); //create Reaction
				double rctcount=0.0;	
				rctMap.clear();
				for (int rt=0;rt<reac->getNumReactants();rt++)
				{	
					const SpeciesReference* rct=reac->getReactant(rt);
					sp=rct->getSpecies();
					rctMap_iter = rctMap.find(sp);			
					if (rctMap_iter != rctMap.end()){	
						rctcount = rctMap_iter->second;
					}		
					else {
						rctcount = 0.0;
					}
					rctcount += rct->getStoichiometry();
					rctMap[sp] = rctcount;
					for (int i=0;(int)i<rct->getStoichiometry();i++)
					{	
						Eref(reaction_).add(subFinfo->msg(),elmtMap_[sp],reacFinfo->msg(),ConnTainer::Default);
				
					}
				}
				double pdtcount = 0.0;
				pdtMap.clear();
				for (int pt=0;pt<reac->getNumProducts();pt++)
				{
					const SpeciesReference* pdt=reac->getProduct(pt);
					sp=pdt->getSpecies();	
					pdtMap_iter = pdtMap.find(sp);
					if (pdtMap_iter != pdtMap.end()){	
						pdtcount = pdtMap_iter->second;
					}		
					else {
						pdtcount = 0.0;
					}
					pdtcount += pdt->getStoichiometry();
					pdtMap[sp] = pdtcount;	
					for (int i=0;i<pdt->getStoichiometry();i++)
					{	
						Eref(reaction_).add(prdFinfo->msg(),elmtMap_[sp],reacFinfo->msg(),ConnTainer::Default);
					}
			
				}
				//order of reactants
				rctorder = 0.0;	
				string rsp = "",psp = "";
				for (rctMap_iter=rctMap.begin();rctMap_iter!=rctMap.end();rctMap_iter++)
				{
					rctorder += rctMap_iter->second;
					rsp=rctMap_iter->first;	//species of the reactant
					//cout<<"rsp "<<rsp<<endl;	
				}	
				cout<<"rct order = "<<rctorder<<endl;
				//order of products
				pdtorder = 0.0;
				for (pdtMap_iter=pdtMap.begin();pdtMap_iter!=pdtMap.end();pdtMap_iter++)
				{
					pdtorder += pdtMap_iter->second;
					psp=pdtMap_iter->first;	//species of the product	
					//cout<<"psp "<<psp<<endl;	
			
				}
				cout<<"pdt order = "<<pdtorder<<endl;
				if (reac->isSetKineticLaw())
				{	KineticLaw * klaw=reac->getKineticLaw();
					string timeunit = klaw->getTimeUnits(); 
					string subunit=klaw->getSubstanceUnits();
					std::string id,unit;
					double value = 0.0;
					UnitDefinition * kfud;
					UnitDefinition * kbud;
					const ASTNode* astnode=klaw->getMath();
					cout <<SBML_formulaToString(astnode) << endl;	
					int np = klaw->getNumParameters();
					cout<<"no of parms : "<<np<<endl;
					vector< string > parameters;
					bool flag = true;
					for (int pi=0;pi<np;pi++)
					{
						Parameter * p = klaw->getParameter(pi);
						if (p->isSetId()){
							id = p->getId();
						}
						if (p->isSetValue()){		
							value=p->getValue();
							cout<<"value of param in kl:"<<value<<endl;	
						}
						parmValueMap[id]=value;
						parameters.push_back(id);
						flag = false;
					}
					double kf=0.0,kb=0.0,kfvalue,kbvalue;
					string kfparm,kbparm;
					if (flag){
						printParameter(); //invoke the function 'parameter'	
						prn_parm(astnode,parameters);
					}
					if (parameters.size() > 2 ){
						cout<<"Sorry! for now MOOSE cannot handle more than 2 parameters .";
						return;
					}
					else if (parameters.size() == 1){
						kfparm = parameters[0];
						kbparm = parameters[0];
					}
					else{
						kfparm = parameters[0];
						kbparm = parameters[1];
					}
					kfvalue = parmValueMap[kfparm];
					kbvalue = parmValueMap[kbparm];
					Parameter* kfp;
					Parameter* kbp;
					if (flag){
						kfp = model_->getParameter(kfparm);
						kbp = model_->getParameter(kbparm);
					}
					else{
						kfp = klaw->getParameter(kfparm);
						kbp = klaw->getParameter(kbparm);
					}			
					kfud = kfp->getDerivedUnitDefinition();
					kbud = kbp->getDerivedUnitDefinition();
					double transkf = transformUnits(1,kfud);	
					cout<<"parm kf trans value : "<<transkf<<endl;
					cout<<"kfvalue :"<<kfvalue<<endl;
					kf = kfvalue * transkf;
					double transkb = transformUnits(1,kbud);
					cout<<"parm kb trans value : "<<transkb<<endl;
					cout<<"kbvalue :"<<kbvalue<<endl;
					kb = kbvalue * transkb;
					set< double >( reaction_, kfFinfo, kf); 
					set< double >( reaction_, kbFinfo, kb);
					//const double NA = 6.02214199e17;
					//kf = kfvalue / pow(NA,rctorder-1);
					//kb = kbvalue / pow(NA,pdtorder-1);
					 
				} //kinetic law	
			}//else modifier
		}//else 	
	}//reaction 
}//create reaction


