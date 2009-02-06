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
#include "SbmlReader.h"

using namespace std;
map<string,double>parmValueMap;
map<string,double>::iterator pvm_iter;
//function to get the parameter used in the kinetic law
string SbmlReader::prn_parm(const ASTNode* p)
{
	string parm = "";	
	bool flag=0;  
	switch ( p->getType() ){
        case (AST_NAME):
	    //cout << "_NAME" << " = " << p->getName() << endl;
	    pvm_iter = parmValueMap.find(std::string(p->getName()));			
	    if (pvm_iter != parmValueMap.end()){
		parm = pvm_iter->first;
		flag = 1;
	    }
	    else flag = 0;
	    break;				
    	}
    if (flag){ 
	return parm;
    }
    int num = p->getNumChildren();
    for( int i = 0; i < num; ++i )
    {  
        const ASTNode* child = p->getChild(i);
        string tmp = prn_parm(child);
        if (!tmp.empty())
        {
            parm = tmp;
            break;
        }
    }
    return parm;
 } 
	
//read a model into MOOSE 
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
	Model* model= document_->getModel();
	if (model == 0)
	{
		cout << "No model present." << endl;
	}
	if (!model->isSetId()){
		cout << "Id not set." << endl;
	}
	//printUnit(model);
	createCompartment(model,location);
}
double SbmlReader::transformUnits(double mvalue,UnitDefinition * ud)
{	//cout<<"num units :"<<ud->getNumUnits()<<endl;
	for (int ut=0;ut<ud->getNumUnits();ut++)
	{
		Unit * unit=ud->getUnit(ut);
		double exponent=unit->getExponent();
		//cout<<"exponent  :"<<exponent<<endl;
		double multiplier=unit->getMultiplier();
		//cout<<"multiplier :"<<multiplier<<endl;
		int scale=unit->getScale();
		//cout<<"scale :"<<scale<<endl;
		double ofset=unit->getOffset(); 
		double lvalue = multiplier * pow((double)10,scale) * pow(mvalue,exponent) + ofset;
		if (unit->isLitre()){
			//cout<<"unit is litre";
			lvalue *= pow(1e-3,exponent);
			//cout<<"size in function is : "<<lsize<<endl;	
		}
		if (unit->isMole()){
			lvalue *= pow(6e23,exponent);		
		}
		return lvalue;
	}
}
// create COMPARTMENT  
void SbmlReader::createCompartment(Model* model,Id location)
{
	static const Cinfo* kincomptCinfo = initKinComptCinfo();
	static const Finfo* sizeFinfo = kincomptCinfo->findFinfo( "size" );
	static const Finfo* dimensionFinfo = kincomptCinfo->findFinfo( "numDimensions" );
	map<string,Id> idMap;	
	Id outcompt; //outside compartment	
	double msize = 0.0,size=0.0;	
	::Compartment* compt;
	for (int i=0;i<model->getNumCompartments();i++)
	{
		compt = model->getCompartment(i);
				
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
                    /*double set_size;
                    get <double> ( comptEl_, sizeFinfo, set_size);
                    //cout << comptEl_->id().path() << "::setting size: " << size << " actually set: " <<  set_size << endl;
                    for ( map<string, Id>::iterator iter = idMap.begin(); iter != idMap.end(); ++iter )
                    {
                        Id id = iter->second;
                        get<double>(id.eref(), sizeFinfo, set_size);
                        //cout << id.path() << "::size = " << set_size << endl;

                    }*/
                    
		}
		if (dimension != 0){
			
			set< unsigned int >( comptEl_,dimensionFinfo,dimension );		
		}
		
		/*if (id != ""){		
			static const Finfo* idFinfo = kincomptCinfo->findFinfo( "id" );
			set< string >( comptEl_,idFinfo,id );
		}
		if (name != ""){		
			static const Finfo* nameFinfo = kincomptCinfo->findFinfo( "name" );
			set< string >( comptEl_,nameFinfo,name );
		}
		if (type != ""){
			static const Finfo* typeFinfo = kincomptCinfo->findFinfo( "type" );
			set< string >( comptEl_,typeFinfo,type );
		}
		if (outside != ""){
			static const Finfo* outsideFinfo = kincomptCinfo->findFinfo( "outside" );
			set< string >( comptEl_,outsideFinfo,outside );
		}*/
		
		
							
	}
	createMolecule(model,idMap);
	
}

//create MOLECULE
void SbmlReader::createMolecule(Model* model,map<string,Id> &idMap)
{	
	
	map<string,Id>molMap;
	map<string,Eref>elmtMap;
	map<string,string>cmptMap;
	static const Cinfo* moleculeCinfo = initMoleculeCinfo();
	static const Finfo* modeFinfo = moleculeCinfo->findFinfo( "mode" );
	static const Finfo* nInitFinfo = moleculeCinfo->findFinfo( "nInit" );	
	static const Finfo* concInitFinfo = moleculeCinfo->findFinfo( "concInit" );
	static const Cinfo* kincomptCinfo = initKinComptCinfo();
	static const Finfo* dimensionFinfo = kincomptCinfo->findFinfo( "numDimensions" );
	static const Finfo* sizeFinfo = kincomptCinfo->findFinfo( "size" );
	
	for (unsigned int m=0;m<model->getNumSpecies();m++)
	{
		Species* s = model->getSpecies(m);
		std::string compt;		
		if (s->isSetCompartment())		
			compt = s->getCompartment();
		const string id=s->getId();
		cmptMap[id]=compt;
		Id comptEl=idMap[compt];
		molecule_= Neutral::create( "Molecule",id,comptEl,Id::scratchId() );//create Molecule
		molMap[id] = comptEl; 
		elmtMap[id]= Eref(molecule_);
		
		UnitDefinition * ud = s->getDerivedUnitDefinition();
		double initvalue;
		if (s->isSetInitialConcentration()){
			initvalue = s->getInitialConcentration();
		}
		
		if (s->isSetInitialAmount()){
			initvalue = s->getInitialAmount() ;
		}
		double transvalue = transformUnits(initvalue,ud);
		
		bool bcondition = s->getBoundaryCondition();
		bool has_subunits = s->getHasOnlySubstanceUnits();
		//cout<<"has_sub "<<has_subunits<<endl;
		unsigned int dimension;
                get< unsigned int >(comptEl.eref(), dimensionFinfo,dimension);
		if (dimension > 0 && s->isSetInitialConcentration() ) { //has_subunits == false ){
					
			double size;
			get< double > (comptEl.eref(),sizeFinfo,size); 			
			transvalue *= size;			
		}
		set< double >(molecule_, nInitFinfo, transvalue); //initialAmount 	
		/*if (dimension > 0 || has_subunits == false)
			set< double >(molecule_, nInitFinfo, transvalue); //initialAmount 			
					
				
		else
			set< double >( molecule_, concInitFinfo, transvalue); //initial concentration	*/
		bool cons=s->getConstant(); 
		if (cons)
			set< int >(molecule_,modeFinfo,4); //getConstant=True indicates a buffered molecule
		else 
			set< int >(molecule_,modeFinfo,0);
			
		
		/*std::string sptype;		
		if (s->isSetSpeciesType())
			sptype=s->getSpeciesType();
		
		if (s->isSetSubstanceUnits()) 
			cout<<"substance units :"<<s->getSubstanceUnits()<<endl;*/
					
	}
	createReaction(model,molMap,elmtMap);
}

//print PARAMETERS
void SbmlReader::printParameter(Model* model)
{	
	//map<string,double>parmValueMap;
	//map<string,double>::iterator pvm_iter;
	//map<string,string>parmUnitMap;
	for (int pm=0;pm<model->getNumParameters();pm++)
	{
		Parameter* prm=model->getParameter(pm);
		std::string id;
		if (prm->isSetId()){
			id = prm->getId();
		}
		double value;		
		if (prm->isSetValue()){		
			value=prm->getValue();	
		}
		parmValueMap[id]=value;
		
	}
}


//create REACTION
void SbmlReader::createReaction(Model* model,map<string,Id> &molMap,map<string,Eref> &elmtMap)
{	
	map<string,double>rctMap;
	map <string, double>::iterator rctMap_iter;
	map<string,double>pdtMap;
	map <string, double>::iterator pdtMap_iter;
	map<string,Eref>::iterator elemt_iter;
	double rctorder,pdtorder;
	static const Cinfo* moleculeCinfo = initMoleculeCinfo();
	static const Finfo* reacFinfo =moleculeCinfo->findFinfo( "reac" );	
	static const Cinfo* reactionCinfo = initReactionCinfo();
	static const Finfo* subFinfo = reactionCinfo->findFinfo( "sub" );
	static const Finfo* prdFinfo = reactionCinfo->findFinfo( "prd" );
	static const Finfo* kfFinfo = reactionCinfo->findFinfo( "kf" );	
	static const Finfo* kbFinfo = reactionCinfo->findFinfo( "kb" );	
	static const Cinfo* kincomptCinfo = initKinComptCinfo();
	static const Finfo* sizeFinfo = kincomptCinfo->findFinfo( "size" );
	static const Finfo* concInitFinfo = moleculeCinfo->findFinfo( "concInit" );
	//printParameter(model); //invoke the function 'parameter'	
	Reaction* reac;	
	for (int r=0;r<model->getNumReactions();r++)
	{	
		reac=model->getReaction(r); 
		const string id=reac->getId();
		cout<<"reaction is "<<id<<endl;
		bool rev=reac->getReversible();
		bool fast=reac->getFast();
		cout<<"is fast"<<fast<<endl;  
		const SpeciesReference* rect=reac->getReactant(0);
		std::string sp=rect->getSpecies();
		Id m=molMap.find(sp)->second; //gives compartment of sp
		reaction_=Neutral::create( "Reaction",id,m,Id::scratchId() ); //create Reaction
		
		//numreact=reac->getNumReactants();
		//cout<<"num of rct :"<<numreact<<endl;
		double rctcount=0.0;	
		rctMap.clear();
		double frate=1.0,brate=1.0;
		for (int rt=0;rt<reac->getNumReactants();rt++)
		{	
			const SpeciesReference* rct=reac->getReactant(rt);
			sp=rct->getSpecies();
			cout<<"reactant is "<<sp<<endl;
			rctMap_iter = rctMap.find(sp);			
			if (rctMap_iter != rctMap.end()){	
				rctcount = rctMap_iter->second;
			}		
			else {
				rctcount = 0.0;
			}
			rctcount += rct->getStoichiometry();
			rctMap[sp] = rctcount;
			m=molMap.find(sp)->second;
			double size;
               		get<double>(m.eref(), sizeFinfo, size); //getting compartment size
			frate *= size;
			Eref molec=elmtMap[sp];	//to get the initial concentration of sp
			double initconc;			
			get< double >( molec, concInitFinfo, initconc);
			//cout<<"initial con of "<<sp <<" is "<<initconc<<endl;
			frate *= pow(initconc,rctcount);
			for (int i=0;(int)i<rct->getStoichiometry();i++)
			{	
				Eref(reaction_).add(subFinfo->msg(),elmtMap[sp],reacFinfo->msg(),ConnTainer::Default);
				
			}
		}
		//numpdt=reac->getNumProducts();
		//cout<<"no of pdt :"<<numpdt<<endl;
		double pdtcount = 0.0;
		pdtMap.clear();
		for (int pt=0;pt<reac->getNumProducts();pt++)
		{
			const SpeciesReference* pdt=reac->getProduct(pt);
			sp=pdt->getSpecies();	
			cout<<"product is "<<sp<<endl;
			pdtMap_iter = pdtMap.find(sp);
			if (pdtMap_iter != pdtMap.end()){	
				pdtcount = pdtMap_iter->second;
			}		
			else {
				pdtcount = 0.0;
			}
			pdtcount += pdt->getStoichiometry();
			pdtMap[sp] = pdtcount;	
			m=molMap.find(sp)->second;
			double size;
                	get<double>(m.eref(), sizeFinfo, size); //getting compartment size
			brate *= size;
			Eref molec=elmtMap[sp];	//to get the initial concentration of sp
			double initconc;			
			get< double >( molec, concInitFinfo, initconc);
			brate *= pow(initconc,pdtcount);	
			for (int i=0;i<pdt->getStoichiometry();i++)
			{	
				Eref(reaction_).add(prdFinfo->msg(),elmtMap[sp],reacFinfo->msg(),ConnTainer::Default);
			}
			
		}
		//order of reactants
		rctorder=0.0;	
		string rsp= "",psp="";
		for (rctMap_iter=rctMap.begin();rctMap_iter!=rctMap.end();rctMap_iter++)
		{
			rctorder += rctMap_iter->second;
			rsp=rctMap_iter->first;	//species of the reactant
			cout<<"rsp "<<rsp<<endl;	
		}	
		//cout<<"rct order = "<<rctorder<<endl;
		Id r=molMap.find(rsp)->second;
		cout<<"r"<< r <<endl;
		
		//order of products
		pdtorder=0.0;
		for (pdtMap_iter=pdtMap.begin();pdtMap_iter!=pdtMap.end();pdtMap_iter++)
		{
			pdtorder += pdtMap_iter->second;
			psp=pdtMap_iter->first;	//species of the product	
			cout<<"psp "<<psp<<endl;	
			
		}
		//cout<<"pdt order = "<<pdtorder<<endl;
		Id p;
		bool noproduct = false;
		if (psp != ""){		
			p=molMap.find(psp)->second;
			cout<<"p"<< p <<endl;
		}
		else if (psp == "")
			noproduct = true;
		if (reac->isSetKineticLaw())
		{	KineticLaw * klaw=reac->getKineticLaw();
			string timeunit = klaw->getTimeUnits(); 
			cout<<"timeunit "<<timeunit<<endl;
			string subunit=klaw->getSubstanceUnits();
			cout<<"subunit "<<subunit<<endl;
			std::string id;
			double value = 0.0;
			int np = klaw->getNumParameters();
			//cout<<"no of parms : "<<np<<endl;
			for (int pi=0;pi<np;pi++)
			{
				Parameter * p = klaw->getParameter(pi);
				if (p->isSetId()){
					id = p->getId();
					//cout<<"id of param in kl :"<<id<<endl;
				}
				if (p->isSetValue()){		
					value=p->getValue();
					//cout<<"value of param in kl:"<<value<<endl;	
				}
			}
			double kf=0.0,kb=0.0;
			string parm;
			if (value == 0.0){
				printParameter(model); //invoke the function 'parameter'	
				const ASTNode* astnode=klaw->getMath();
				cout <<SBML_formulaToString(astnode) << endl;			
				parm = prn_parm(astnode);
				pvm_iter = parmValueMap.find(parm);			
		    		if (pvm_iter != parmValueMap.end()){
					value = pvm_iter->second;
					
				}
			}
			/*double NA = 6.02214199e23; //Avogardo's number	
			if (parm == "k1" || parm == "k2"){
				
				kf=NA*n*value*pow((1/6e23*n),rctorder-1);
				if (rev)				
					kb=NA*n*value*pow((1/6e23*n),pdtorder-1);
			}
			else if (parm == "KT" || parm == "k3"){
				kf=n*value;
				if (rev)
					kb=n*value;
			}*/
			if (noproduct){
				double size;
                		get<double>(r.eref(), sizeFinfo, size); 
				cout<<"size "<<size<<endl;				
				kf = size * value;
				kb = 0;	
				cout<<"kf = "<<kf<<"kb = "<<kb<<endl;	
			}							
			else if (r != p){
				double size;
                		get<double>(p.eref(), sizeFinfo, size); 
				cout<<"size "<<size<<endl;				
				kf = size * value;
				if (rev)
					kb = size * value;	
				cout<<"kf = "<<kf<<"kb = "<<kb<<endl;		
			}
			else if ((r == p) && (noproduct == false)){ 
				kf = frate * value;
				if (rev)				
					kb = brate * value;
			}
			set< double >( reaction_, kfFinfo, kf); 
			set< double >( reaction_, kbFinfo, kb); 
		} //kinetic law		
	}//reaction 
}//create reaction


