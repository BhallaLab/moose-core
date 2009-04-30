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
#include <sbml/SBMLTypes.h>
#include "element/Neutral.h"
#include "element/Wildcard.h"
#include "kinetics/KinCompt.h"
#include <sstream>
#include <set>
#include <algorithm>
#include "SbmlWriter.h"
#include <sbml/annotation/ModelHistory.h>
#include <time.h>

void SbmlWriter::write(string filename,Id location)
{
  	SBMLDocument* sbmlDoc = 0;
  	bool SBMLok = false;
	sbmlDoc = createModel(filename); 
  	SBMLok  = validateModel(sbmlDoc);
	filename += ".xml";
    	if (SBMLok) 
		writeModel(sbmlDoc, filename);
    	delete sbmlDoc;
	if (!SBMLok) {
		cerr << "Errors encountered " << endl;
		return ;
	}
}

/**
 * Create an SBML model in the SBML Level 2 Version 4 Specification.
 */

SBMLDocument* SbmlWriter::createModel(string filename)
{
	// Create an SBMLDocument object using in Level 2 Version 4 format:
	SBMLDocument* sbmlDoc = new SBMLDocument(2, 4);
	Model* model = sbmlDoc->createModel();
  	model->setId(filename);
	model->setMetaId(filename);
	model->setName(filename);

       	ModelHistory * h = new ModelHistory();

	ModelCreator *c = new ModelCreator();
	c->setFamilyName("Bhalla");
	c->setGivenName("Upi");
	c->setEmail("bhalla@ncbs.res.in");
	c->setOrganisation("National Centre for Biological Sciences");

	h->addCreator(c);

	Date * date = new Date("2006-10-18T16:39:27");
	Date * date2 = new Date("2009-03-11T09:39:00-02:00");

	h->setCreatedDate(date);
	h->setModifiedDate(date2);

	model->setModelHistory(h);

  	UnitDefinition* unitdef;
 	Unit* unit;

  	// (UnitDefinition1) Create an UnitDefinition object for "per_second".
  	unitdef = model->createUnitDefinition();
  	unitdef->setId("per_second");
  	// Create an Unit inside the UnitDefinition object above:
  	/*unit = unitdef->createUnit();
	unit->setKind(UNIT_KIND_LITRE);
	unit->setExponent(1);*/

  	unit = unitdef->createUnit();
  	unit->setKind(UNIT_KIND_SECOND);
  	unit->setExponent(-1);

  	// Create an UnitDefinition object for "litre_per_mole_per_second".  
      	unitdef = model->createUnitDefinition();
 	unitdef->setId("per_umole_per_second");
    	// Create individual unit objects that will be put inside
  	// the UnitDefinition to compose "litre_per_mole_per_second".
	/*unit = unitdef->createUnit();
	unit->setKind(UNIT_KIND_LITRE);
	unit->setExponent(1);*/

	unit = unitdef->createUnit();
	unit->setKind(UNIT_KIND_MOLE);
	unit->setExponent(-1);
	unit->setScale(-6);

	unit = unitdef->createUnit();
	unit->setKind(UNIT_KIND_SECOND);
	unit->setExponent(-1);
	
	 //Create an UnitDefinition object for "substance".  
	 unitdef = model->createUnitDefinition();
 	 unitdef->setId("substance");
	 // Create individual unit objects that will be put inside
  	 // the UnitDefinition to compose "substance".
	 unit = unitdef->createUnit();
	 unit->setKind(UNIT_KIND_MOLE);
	 unit->setExponent(1);
	 unit->setScale(-6);
	 
	// Create an UnitDefinition object for "litre_sq_per_mole_sq_per_second".
	
	unitdef = model->createUnitDefinition();
 	unitdef->setId("per_umole_sq_per_second");
    	// Create individual unit objects that will be put inside
  	// the UnitDefinition to compose "litre_sq_per_mole_sq_per_second".
	 unit = unitdef->createUnit();
	 unit->setKind(UNIT_KIND_MOLE);
	 unit->setExponent(-2);
	 unit->setScale(-6);
	
	 unit = unitdef->createUnit();
	 unit->setKind(UNIT_KIND_SECOND);
	 unit->setExponent(-1);	
	 
 	// Create a string for the identifier of the compartment.  

	static const Cinfo* kincomptCinfo = initKinComptCinfo();
	static const Finfo* sizeFinfo = kincomptCinfo->findFinfo( "size" );
	static const Finfo* dimensionFinfo = kincomptCinfo->findFinfo( "numDimensions" );
	Eref comptEl;
	vector< Id > compts;
	vector< Id > kinSpecies;
	vector< Id >::iterator itr;
	wildcardFind("/kinetics/##[TYPE=KinCompt]", compts);
	wildcardFind("/kinetics/##[TYPE=Molecule]", kinSpecies);
 	vector< Eref > outcompt;	
	bool flag = true;
	if ((compts.size() > 0) && (kinSpecies.size() > 0))
	{
		Id kinetics( "/kinetics" );//here kinetics is added with other compartments in the model.
		compts.push_back(kinetics); 
	}
	if (compts.empty())     
	{
		Id kinetics( "/kinetics" );//here kinetics is considered as the default compartment as no compartment is specified in the model.
		compts.push_back(kinetics); 
		flag = false;
	}	
	for (itr = compts.begin(); itr != compts.end(); itr++)
	{
		string parentCompt;	
		ostringstream cid;	
		comptEl = ( *itr )();
		::Compartment* compt = model->createCompartment();
		string comptName = comptEl.name();
		cid << comptEl.id().id()<<"_"<<comptEl.id().index();
		comptName = nameString(comptName); //removes special characters from name
		comptName =  changeName(comptName,cid.str());
		string newcomptName = idBeginWith(comptName);
		compt->setId(newcomptName);		
		double size;
               	get<double>(comptEl, sizeFinfo, size);
		unsigned int numD;
		get<unsigned int>(comptEl, dimensionFinfo, numD);
		compt->setSpatialDimensions(numD);
		if (numD == 3)		
			compt->setSize(size*1e3);
		else
			compt->setSize(size);
		
		outcompt.clear();
		if (flag ){
			targets(comptEl,"outside",outcompt);
			if ( outcompt.size() > 1 )
			cerr << "Warning: SbmlWriter: Too many outer compartments for " << newcomptName << ". Ignoring all but one.\n";
			if ( outcompt.size() >= 1 ) {
				ostringstream outId;				
				string outName = outcompt[ 0 ].name();
				outName = nameString(outName);
				outId<<outcompt[ 0 ].id().id()<<"_"<<outcompt[ 0 ].id().index();
				outName = changeName(outName,outId.str());
				string newoutName = idBeginWith(outName);
				compt->setOutside(newoutName);
			}
		}
		// Create the Species objects inside the Model object. 
		static const Cinfo* moleculeCinfo = initMoleculeCinfo();	
		static const Finfo* nInitFinfo = moleculeCinfo->findFinfo( "nInit" );	
		static const Finfo* modeFinfo = moleculeCinfo->findFinfo( "mode" );
		Eref moleEl;	
		vector< Id > molecs;
		vector< Id >::iterator mitr;
		string comptPath = comptEl.id().path();
		wildcardFind(comptPath + "/#[TYPE=Molecule]", molecs);
		for (mitr = molecs.begin(); mitr != molecs.end(); mitr++)
		{		
			moleEl = ( *mitr )();	
			ostringstream mid;
			Id parent = Neutral::getParent(moleEl);
			parentCompt = getParentFunc(moleEl);	
			Species *sp = model->createSpecies();
			string molName = (moleEl)->name();
			mid << moleEl.id().id()<< "_"<<moleEl.id().index();
			molName = nameString(molName); //removes special characters from name
			string newSpName = changeName(molName,mid.str());
			newSpName = idBeginWith(newSpName);
			sp->setId(newSpName);
			sp->setCompartment(parentCompt);
			sp->setHasOnlySubstanceUnits(true);
			int mode;		
			get< int >(moleEl,modeFinfo,mode);
			if (mode == 0){
				sp->setConstant(false);
				sp->setBoundaryCondition(false); 
			}	
			else if ((mode == 4) || (mode == 5) || (mode == 6))
			{
				sp->setConstant(true);
				sp->setBoundaryCondition(true); 
			}
			else if ((mode == 1) ||(mode == 2))
			{
				sp->setConstant(false);
				sp->setBoundaryCondition(true); 
			}
			double initamt = 0.0;
			get< double >(moleEl,nInitFinfo,initamt); 
			initamt /= 6.02214199e17 ; 
			sp->setInitialAmount(initamt);			
			vector< Eref > sumtotal;
			targets(moleEl,"sumTotal",sumtotal);
			//cout<<" sumtotal size :"<<sumtotal.size()<<endl;
			int sumtot_count = sumtotal.size();
			if (sumtot_count > 0)
			{
				ostringstream sumtotal_formula;				
				for (unsigned int i=0; i<sumtotal.size();i++) 
				{	
					ostringstream spId;					
					sumtot_count -= 1;				
					string spName = sumtotal[ i ].name();	
					spName = nameString(spName);	//removes special characters from name
					spId<<sumtotal[ i ].id().id()<<"_"<<sumtotal[ i ].id().index();
					string newName = changeName(spName,spId.str());
					newName = idBeginWith(newName);
					if (sumtot_count == 0)
						sumtotal_formula << newName;
					else
						sumtotal_formula << newName << "+";
				}
				//cout<<"sumtotal formula is :"<<	sumtotal_formula.str()<<endl;
				Rule * rule = model->createAssignmentRule();
				rule->setVariable(newSpName);
				rule->setFormula(sumtotal_formula.str());
				
			}

			/* Enzyme */
			vector< Id > enzms;
			string molecPath = moleEl.id().path();
			wildcardFind(molecPath + "/#[TYPE=Enzyme]", enzms);
			printEnzymes(enzms,model);
			
		
		} //molecule
		static const Cinfo* reactionCinfo = initReactionCinfo();
		static const Finfo* kbFinfo = reactionCinfo->findFinfo( "kb" );	
		static const Finfo* kfFinfo = reactionCinfo->findFinfo( "kf" );	
		Eref rectnEl;
		vector< Id > reaction;
		vector< Id >::iterator ritr;
		vector< Eref > rct;
		vector< Eref > pdt;				
		wildcardFind(comptPath + "/#[TYPE=Reaction]", reaction);
		for (ritr = reaction.begin(); ritr != reaction.end(); ritr++)
		{
			rectnEl = ( *ritr )();	
			ostringstream rtnId;	
			Reaction* reaction;
			SpeciesReference* spr;
			KineticLaw* kl;
			Parameter*  para; 
		 	// Create Reactant objects inside the Reaction object 
			reaction = model->createReaction();
			string rtnName = (rectnEl)->name();
			rtnId<<rectnEl->id().id()<<"_"<<rectnEl->id().index();
			rtnName = nameString(rtnName); //removes special characters from name
			string newName = changeName(rtnName,rtnId.str());
			newName = idBeginWith(newName);
			reaction->setId(newName);
			cout<<"reaction :"<<newName<<endl;
			double kb=0.0,kf=0.0;
			get< double >( rectnEl, kbFinfo, kb); 
			get< double >( rectnEl, kfFinfo, kf); 
			if (kb == 0.0)
				reaction->setReversible(false);
			else
				reaction->setReversible(true);
			//  Create a Reactant object that references particular Species in the model.  
			rct.clear();
			targets(rectnEl,"sub",rct);
			std::set< Eref > rctUniq;
			double rctstoch ;
			rctUniq.insert(rct.begin(),rct.end());
			std::set< Eref >::iterator ri;
			double rct_order = 0.0;
			for(ri=rctUniq.begin(); ri != rctUniq.end(); ri++)
			{	
				ostringstream rctId;	
				spr = reaction->createReactant();
				string rctName = (*ri).name();	
				rctName = nameString(rctName); //removes special characters from name
				Eref e = *ri;
				rctId<<e.id().id()<<"_"<<e.id().index();
				string newrctName = changeName(rctName,rctId.str());
				newrctName = idBeginWith(newrctName);
				spr->setSpecies(newrctName);				
				rctstoch = count(rct.begin(),rct.end(),*ri);
				spr->setStoichiometry(rctstoch);
				rct_order += rctstoch;
				
				
			}
			cout<<"rct_order is "<<rct_order<<endl;
			// Create a Product object that references particular Species  in the model. 		
			pdt.clear();
			targets(rectnEl,"prd",pdt);
			std::set < Eref > pdtUniq;
			double pdtstoch;
			pdtUniq.insert(pdt.begin(),pdt.end());
			std::set < Eref > ::iterator pi;
			double pdt_order = 0.0;
			for(pi = pdtUniq.begin(); pi != pdtUniq.end(); pi++)
			{
				ostringstream pdtId;			
				spr = reaction->createProduct();
				string pdtName = (*pi).name();	
				pdtName = nameString(pdtName); //removes special characters from name
				Eref e = *pi;
				pdtId<< e.id().id()<<"_"<<e.id().index();
				string newpdtName = changeName(pdtName,pdtId.str()); 
				newpdtName = idBeginWith(newpdtName);	
				spr->setSpecies(newpdtName);				
				pdtstoch=count(pdt.begin(),pdt.end(),*pi);
				spr->setStoichiometry(pdtstoch);
				pdt_order += pdtstoch;
				
			}
			cout<<"pdt_order is "<<pdt_order<<endl;
			// Create a KineticLaw object inside the Reaction object 
			ostringstream rate_law,kfparm,kbparm;
			if (kf != 0.0 ){
				kfparm<<rtnName<<"_"<<"kf";
				kbparm<<rtnName<<"_"<<"kb";
				rate_law <<kfparm.str();
				double rstoch=0.0,r_order=0.0;
				for(ri = rctUniq.begin(); ri != rctUniq.end(); ri++)
				{
					ostringstream rId;					
					rstoch = count(rct.begin(),rct.end(),*ri);
					r_order += rstoch;
					string riName = nameString((*ri).name()); //removes special characters from name
					Eref e = *ri;
					rId<<e.id().id()<<"_"<<e.id().index();
					string newriName = changeName(riName,rId.str());	
					newriName = idBeginWith(newriName);
					if (rstoch == 1)
						rate_law<<"*"<<newriName;
					else
						rate_law <<"*"<<newriName<<"^"<<rstoch;
   				} 
				
			}
			if (kb != 0.0){
				rate_law <<"-"<<kbparm.str();
				std::set < Eref > ::iterator i;
				//pdtcount=pdtUniq.size();
				for(pi = pdtUniq.begin(); pi != pdtUniq.end(); pi++)
				{	
					ostringstream pId;					
					pdtstoch=count(pdt.begin(),pdt.end(),*pi);
					string piName = nameString((*pi).name()); //removes special characters from name
					Eref e = *pi;
					pId<<e.id().id()<<"_"<<e.id().index();
					string newpiName = changeName(piName,pId.str());
					newpiName = idBeginWith(newpiName);
					if (pdtstoch == 1)
						rate_law<<"*"<<newpiName;
					else
						rate_law <<"*"<<newpiName<<"^"<<pdtstoch;
   										
				} 
			}
			kl  = reaction->createKineticLaw();
			kl->setFormula(rate_law.str());
			cout<<"rate law: "<<rate_law.str()<<endl; 
			// Create local Parameter objects inside the KineticLaw object. 
			para = kl->createParameter();
			para->setId(kfparm.str());
			string unit=parmUnit(rct_order-1);
			para->setUnits(unit);
			double rvalue,pvalue;
			const double m = 6.02214199e17; // 1uMole=6.023e17
			rvalue = kf *(pow(m,rct_order-1));
			para->setValue(rvalue);
			if (kb != 0.0){
				
				pvalue = kb * (pow(m,pdt_order-1));
				para = kl->createParameter();
				para->setId(kbparm.str());
				string unit=parmUnit(pdt_order-1);
				para->setUnits(unit);
				para->setValue(pvalue);
			}
		} //reaction
	} //compartment
 	
	return sbmlDoc;

}
string SbmlWriter::getParentFunc(Eref p)
{	
	string parentName;
	ostringstream parentId;
	Id parent = Neutral::getParent(p);
	parentName = parent()->name();
	parentId << parent()->id().id()<<"_"<<parent()->id().index();
	parentName = nameString(parentName);
	parentName = changeName(parentName,parentId.str());
	parentName = idBeginWith(parentName);
	return parentName;
}
				
void SbmlWriter::printParameters(KineticLaw* kl,string k,double kvalue,string unit)
{
	Parameter* para = kl->createParameter();
	para->setId(k);
	para->setValue(kvalue);
	para->setUnits(unit);
}
void SbmlWriter::getEnzyme(vector< Eref > enz,vector <string> &enzsName)
{
	string parentName;
	for (unsigned int i=0; i<enz.size();i++)
	{	
		ostringstream enzId;		
		string enzName = enz[ i ].name();	
		enzName = nameString(enzName);	//removes special characters from name
		enzId << enz[ i ].id().id()<< "_"<< enz[ i ].id().index();	
		string newenzName = changeName(enzName,enzId.str());
		newenzName = idBeginWith(newenzName);
		enzsName.push_back("<moose:enzyme>"+newenzName +"</moose:enzyme> \n");

	}
}

void SbmlWriter::getSubstrate(vector< Eref > sub,vector <string> &subsName)
{
	string parentName;
	for (unsigned int i=0; i<sub.size();i++)
	{	
		ostringstream subId;		
		string subName = sub[ i ].name();
		subName = nameString(subName);	//removes special characters from name	
		subId <<sub[i].id().id()<<"_"<<sub[i].id().index();	
		string newsubName = changeName(subName,subId.str());
		newsubName = idBeginWith(newsubName);
		subsName.push_back("<moose:substrates>"+newsubName +"</moose:substrates> \n");
	}
}
void SbmlWriter::getProduct(vector< Eref > prd,vector <string> &prdsName)
{
	string parentName;
	for (unsigned int i=0; i<prd.size();i++)
	{	
		ostringstream prdId;		
		string prdName = prd[ i ].name();	
		prdName = nameString(prdName);	//removes special characters from name	
		prdId <<  prd[ i ].id().id()<<"_"<<prd[ i ].id().index();
		string newprdName = changeName(prdName,prdId.str());	
		newprdName = idBeginWith(newprdName);	
		prdsName.push_back("<moose:products>"+newprdName+"</moose:products> \n");	
	}
}
void SbmlWriter::printReactants(Reaction* reaction,vector< Eref > sub,ostringstream& rlaw)
{	
	for (unsigned int i=0; i<sub.size();i++)
	{	
		ostringstream subId;		
		string subName = sub[ i ].name();
		subName = nameString(subName);	//removes special characters from name	
		subId << sub[ i ].id().id()<<"_"<<sub[ i ].id().index();
		string newsubName = changeName(subName,subId.str());
		newsubName = idBeginWith(newsubName);
		SpeciesReference* spr = reaction->createReactant();
		spr->setSpecies(newsubName);
		rlaw<<"*"<<newsubName;
				
	}
}
void SbmlWriter::printProducts(Reaction* reaction,vector< Eref > prd,ostringstream& rlaw)
{
	for (unsigned int i=0; i<prd.size();i++)
	{	
		ostringstream prdId;
		string prdName = prd[ i ].name();
		prdName = nameString(prdName);	//removes special characters from name	
		prdId << prd[ i ].id().id()<<"_"<< prd[ i ].id().index();	
		string newprdName = changeName(prdName,prdId.str());
		newprdName = idBeginWith(newprdName);
		SpeciesReference* spr = reaction->createProduct();
		spr->setSpecies(newprdName);
		bool rev=reaction->getReversible();
		if (rev)
			rlaw<<"*"<<newprdName;
	}
}

/* Enzyme */
void SbmlWriter::printEnzymes(vector< Id > enzms,Model* model)
{	
	Eref enzEl;	
	vector< Id >::iterator ezitr;
	vector< Eref > enz;
	vector< Eref > sub;
	vector< Eref > cplx;
	vector< Eref > prd;
	vector <string> enzsName;
	vector <string> subsName;
	vector <string> prdsName;
	vector <string> cpxName;
	for (ezitr = enzms.begin(); ezitr != enzms.end(); ezitr++)
	{		
		enzEl = ( *ezitr )();
		ostringstream  parentMoleId,enzId,complexId;
		Id enzParent=Neutral::getParent(enzEl);
		string parentMole=enzParent()->name();
		parentMole = nameString(parentMole); //removes special characters from name
		parentMoleId << enzParent()->id().id()<<"_"<< enzParent()->id().index();
		string parentCompt = getParentFunc(enzParent()); //grand Parent
		string enzName = (enzEl)->name();
		enzName = nameString(enzName); //removes special characters from name
		enzId << (enzEl)->id().id()<<"_"<< (enzEl)->id().index();
		string newenzName = changeName(enzName,enzId.str());
		static const Cinfo* enzymeCinfo = initEnzymeCinfo();
		static const Finfo* k1Finfo = enzymeCinfo->findFinfo( "k1" );	
		static const Finfo* k2Finfo = enzymeCinfo->findFinfo( "k2" );
		static const Finfo* k3Finfo = enzymeCinfo->findFinfo( "k3" );
		static const Finfo* kmFinfo = enzymeCinfo->findFinfo( "Km" );
		static const Finfo* kcatFinfo = enzymeCinfo->findFinfo( "kcat" );
		static const Finfo* emodeFinfo = enzymeCinfo->findFinfo( "mode" );
		double k1,k2,k3,Km,kcat;
		bool emode;
		get< double >(enzEl,k1Finfo,k1);
		get< double >(enzEl,k2Finfo,k2);
		get< double >(enzEl,k3Finfo,k3);
		get< double >(enzEl,kmFinfo,Km);
		get< double >(enzEl,kcatFinfo,kcat);
		get< bool >(enzEl,emodeFinfo,emode);
		
	  	if (emode ==  0){
			Reaction* react;
			KineticLaw* kl;
			ostringstream rlaw,grpName;
			rlaw<<"k1";
			react = model->createReaction();
			string name = idBeginWith(newenzName);
			string rctnName = changeName(name,"Complex_formation");
			react->setId(rctnName);
			//react->setMetaId("reaccomplex");
			react->setReversible(true);
			enz.clear();
			targets(enzEl,"enz",enz);
			printReactants(react,enz,rlaw);
			sub.clear();
			targets(enzEl,"sub",sub);
			printReactants(react,sub,rlaw);
			rlaw<<"-"<<"k2";			
			cplx.clear();
			targets(enzEl,"cplx",cplx);
			string cplxName = cplx[ 0 ].name();
			complexId << cplx[ 0 ].id().id()<<"_"<< cplx[ 0 ].id().index();	
			cplxName = nameString(cplxName); //removes special characters from name	
			string newcplxName = changeName(cplxName,complexId.str());	
			newcplxName = idBeginWith(newcplxName);
			cpxName.clear();
			cpxName.push_back("<moose:complex>"+newcplxName+"</moose:complex> \n");
			Species *sp = model->createSpecies(); // create the complex species 
			sp->setId(newcplxName);
			sp->setCompartment(parentCompt);
			sp->setInitialAmount(0.0);	
			sp->setHasOnlySubstanceUnits(true);
			SpeciesReference* spr = react->createProduct();
			spr->setSpecies(newcplxName);
			bool rev=react->getReversible();
			if (rev)
				rlaw<<"*"<<newcplxName;
			//sp->setNotes("<body xmlns=\"http://www.w3.org/1999/xhtml\">\n\t\t this is a complex species \n\t    </body>");
			cout<<"rate law of complex formation :"<<rlaw.str()<<endl; 
			kl  = react->createKineticLaw();
			kl->setFormula(rlaw.str());
			//kl->setNotes("<body xmlns=\"http://www.w3.org/1999/xhtml\">\n\t\t" + rlaw.str() + "\n\t    </body>");
			double f = 6.02214199e17;
			k1 = k1 * (pow(f,1));
			printParameters(kl,"k1",k1,"per_umole_per_second"); //set the parameters
			printParameters(kl,"k2",k2,"per_second"); 
			string reaction_notes = "<body xmlns:moose=\"http://www.moose.ncbs.res.in\">\n\t\t";
			reaction_notes += "<moose:EnzymaticReaction> \n";
			enzsName.clear();
			getEnzyme(enz,enzsName);
			for(unsigned int i =0; i< enzsName.size(); i++)
			{
				reaction_notes += enzsName[i] ;			
			}
			subsName.clear();			
			getSubstrate(sub,subsName);
			for(unsigned int i =0; i< subsName.size(); i++)
			{
				reaction_notes += subsName[i] ;			
			}
			for(unsigned int i =0; i< cpxName.size(); i++)
			{
				reaction_notes += cpxName[i] ;			
			}
			grpName <<"<moose:groupName>"<<name<<"</moose:groupName>";
			reaction_notes += grpName.str();
			reaction_notes += "<moose:stage>1</moose:stage> \n";
			reaction_notes += "</moose:EnzymaticReaction> \n";
			reaction_notes += "</body>";
			XMLNode* xnode =XMLNode::convertStringToXMLNode(reaction_notes);
			react->setAnnotation(xnode);	
			Eref ezmoleEl;		
			vector< Id >::iterator emitr;	
			vector< Eref > ezprd;
			vector< Eref > ezsub;
			vector< Eref > ezrct;
			vector< Eref > ezm;
			string enzPath = enzEl.id().path();
			vector< Id > ezmole;
			ostringstream law;
			react = model->createReaction();
			rctnName = changeName(newenzName,"Product_formation");
			rctnName = idBeginWith(rctnName);
			react->setId(rctnName);
			react->setReversible(false);
			wildcardFind(enzPath + "/#[TYPE=Molecule]", ezmole);
			for (emitr = ezmole.begin(); emitr != ezmole.end(); emitr++)
			{
				ezmoleEl = ( *emitr )();
				string ezMole = (ezmoleEl)->name();
				ezMole = nameString(ezMole); //removes special characters from name
				law<<"k3";
				cplx.clear();
				SpeciesReference* spr = react->createReactant();
				spr->setSpecies(newcplxName);
				law<<"*"<<newcplxName;
				Id ezMParent=Neutral::getParent(ezmoleEl);
				string parentEMole=ezMParent()->name();
				parentEMole = nameString(parentEMole); //removes special characters from name
				ezm.clear();
				targets(ezMParent(),"enz",ezm);
				printProducts(react,ezm,law); //invoke function createProducts
				ezprd.clear();
				targets(ezMParent(),"prd",ezprd);
				printProducts(react,ezprd,law); //invoke function createProducts
				cout<<"rate law of product formation :"<<law.str()<<endl; 
				kl  = react->createKineticLaw();
				kl->setFormula(law.str());
				//kl->setNotes("<body xmlns=\"http://www.w3.org/1999/xhtml\">\n\t\t" + law.str() + "\n\t    </body>");
				printParameters(kl,"k3",k3,"per_second"); //set the parameters
				string reaction1_notes = "<body xmlns:moose=\"http://www.moose.ncbs.res.in\">\n\t\t";
				reaction1_notes += "<moose:EnzymaticReaction> \n";
				for(unsigned int i =0; i< enzsName.size(); i++)
				{
					reaction1_notes += enzsName[i] ;			
				}
				for(unsigned int i =0; i< cpxName.size(); i++)
				{
					reaction1_notes += cpxName[i] ;			
				}
				prdsName.clear();
				getProduct(ezprd,prdsName);
				for(unsigned int i =0; i< prdsName.size(); i++)
				{
					reaction1_notes += prdsName[i] ;			
				}
				reaction1_notes += grpName.str();
				reaction1_notes += "<moose:stage>2</moose:stage> \n";
				reaction1_notes += "</moose:EnzymaticReaction> \n";
				reaction1_notes += "</body>";
				XMLNode* x1node =XMLNode::convertStringToXMLNode(reaction1_notes);
				react->setAnnotation(x1node);
				} //enzyme molecule
		} //if
		else if (emode == 1){
			Reaction* react;
			KineticLaw* kl;
			ostringstream rlaw;
			rlaw<<"kcat";
			react = model->createReaction();
			string rctnName = changeName(newenzName,"MM_Reaction");	
			rctnName = idBeginWith(rctnName);
			react->setId(rctnName);
			cout<<"reaction : "<<rctnName<<endl;
			react->setReversible(false);
			sub.clear();
			targets(enzEl,"sub",sub);
			printReactants(react,sub,rlaw); //invoke function createReactants
			//cout<<"no of substrates :"<<sub.size()<<endl;	
			enz.clear();
			targets(enzEl,"enz",enz);
			//cout<<"no of enzyme :"<<enz.size()<<endl;
			for (unsigned int i=0; i<enz.size();i++)
			{	
				ostringstream enzId;				
				string enzName = enz[ i ].name();
				enzName = nameString(enzName);	//removes special characters from name		
				enzId << enz[i].id().id()<<"_"<<enz[i].id().index();
				string newenzName = changeName(enzName,enzId.str());
				newenzName = idBeginWith(newenzName);	
				rlaw<<"*"<<newenzName;
				ModifierSpeciesReference * mspr = model->createModifier();
				mspr->setSpecies(newenzName);
							
			}

			prd.clear();
			targets(enzEl,"prd",prd);
			printProducts(react,prd,rlaw);
			//cout<<"no of prds :"<<prd.size()<<endl;
			rlaw<<"/"<<"("<<"Km"<<" +";
			int subcount=sub.size();
			for (unsigned int i=0; i<sub.size();i++)
	  		{	
				ostringstream subId;				
				subcount -= 1;				
				string subName = sub[ i ].name();
				subName = nameString(subName);	//removes special characters from  name
				subId << sub[i].id().id()<<"_"<< sub[i].id().index();
				string newsubName = changeName(subName,subId.str());
				newsubName = idBeginWith(newsubName);	
				if (subcount == 0)		
					rlaw<<"*"<<newsubName;
				else
					rlaw<<"*"<<newsubName<<"*";
				
			}
			rlaw<<")";
			cout<<"rate law of MM Reaction :"<<rlaw.str()<<endl; 
			kl  = react->createKineticLaw();
			kl->setFormula(rlaw.str());
			kl->setNotes("<body xmlns=\"http://www.w3.org/1999/xhtml\">\n\t\t" + rlaw.str() + "\n\t    </body>");
			printParameters(kl,"kcat",kcat,"per_second"); //set the parameters
			printParameters(kl,"Km",Km,"substance"); 
						
		}
					
	} //enzyme	
}					

string SbmlWriter::parmUnit(double rct_order)
{
	ostringstream unit;
	int order = (int)rct_order;	
	switch(order)
	{
		case 0:
			unit<<"per_second";
			break;		
		case 1:
			unit<<"per_umole_per_second";
			break;
		case 2:
			unit<<"per_umole_sq_per_second";
			break;
		
		default:
			unit<<"per_umole_"<<rct_order<<"_per_second";
			break;
		
	}
	return unit.str();

}
string SbmlWriter::changeName(string parent, string child)
{
	string newName = parent + "_" + child + "_";
	return newName;
}
string SbmlWriter::idBeginWith(string name)
{
	string changedName = name;	
	if (isdigit(name.at(0)))
		changedName = "_" + name;
	return changedName;
}
string SbmlWriter::nameString(string str)
{	
		
	string str1;
	int len = str.length();
	int i= 0;
	do
	{	
		
		switch(str.at(i))
		{
			case '-':
				str1 = "_minus_";	
				str.replace(i,1,str1);
				len += str1.length()-1;	
				break;
			case '+':
				str1 = "_plus_"	;			
				str.replace(i,1,str1);
				len += str1.length()-1;
				break;
			case '*':
				str1 = "_star_"	;			
				str.replace(i,1,str1);
				len += str1.length()-1;
				break;
			case '/':
				str1 = "_slash_";				
				str.replace(i,1,str1);
				len += str1.length()-1;
				break;
			case '(':
				str1 = "_bo_";				
				str.replace(i,1,str1);
				len += str1.length()-1;
				break;
			case ')':
				str1 = "_bc_";				
				str.replace(i,1,str1);
				len += str1.length()-1;
				break;
			case '[':
				str1 = "_sbo_";				
				str.replace(i,1,str1);
				len += str1.length()-1;
				break;
			case ']':
				str1 = "_sbc_";				
				str.replace(i,1,str1);
				len += str1.length()-1;
				break;
			case '.':
				str1 = "_dot_";	
				str.replace(i,1,str1);
				len += str1.length()-1;	
				break;
		}//switch 
	i++;
	}while ( i < len );
	return str;
}
/**
 *
 * Writes the given SBMLDocument to the given file.
 *
 */ 
bool SbmlWriter::writeModel(const SBMLDocument* sbmlDoc, const string& filename)
{
	  SBMLWriter sbmlWriter;
	  bool result = sbmlWriter.writeSBML(sbmlDoc, filename);
	  if (result)
	  {
	  	cout << "Wrote file \"" << filename << "\"" << endl;
	  	return true;
	  }
	  else
	  {
	  	cerr << "Failed to write \"" << filename << "\"" << endl;
	  	return false;
  	  }
}

int SbmlWriter::targets(Eref object,const string& msg,vector< Eref >& target,const string& type )
{
	unsigned int oldSize = target.size();
	Eref found;
	Conn* i = object->targets( msg, 0 );
	for ( ; i->good(); i->increment() ) {
		found = i->target();
		if ( type != "" && !isType( found, type ) )	// speed this up
			continue;
		
		target.push_back( found );		
	}
	delete i;
	
	return target.size() - oldSize;
}

bool SbmlWriter::isType( Eref object, const string& type )
{
	return object->cinfo()->isA( Cinfo::find( type ) );
}

bool SbmlWriter::validateModel(SBMLDocument* sbmlDoc)
{
	  if (!sbmlDoc)
	  {
	    cerr << "validateModel: given a null SBML Document" << endl;
	    return false;
	  }
	 
	  string consistencyMessages;
	  string validationMessages;
	  bool noProblems                     = true;
	  unsigned int numCheckFailures       = 0;
	  unsigned int numConsistencyErrors   = 0;
	  unsigned int numConsistencyWarnings = 0;
	  unsigned int numValidationErrors    = 0;
	  unsigned int numValidationWarnings  = 0;

	  // Once the whole model is done and before it gets written out, 
	  // it's important to check that the whole model is in fact complete, consistent and valid.

	  numCheckFailures = sbmlDoc->checkInternalConsistency();
	  if ( numCheckFailures > 0 )
	  {
		    noProblems = false;
		    for (unsigned int i = 0; i < numCheckFailures; i++)
		    {
		      const SBMLError* sbmlErr = sbmlDoc->getError(i);
		      if ( sbmlErr->isFatal() || sbmlErr->isError() )
		      {
			++numConsistencyErrors;
		      }
		      else
		      {
			++numConsistencyWarnings;
		      }      
		    } 
		    ostringstream oss;
		    sbmlDoc->printErrors(oss);
		    consistencyMessages = oss.str(); 
	  }

	  // If the internal checks fail, it makes little sense to attempt
	  // further validation, because the model may be too compromised to
	  // be properly interpreted.

	  if (numConsistencyErrors > 0)
	  {
	    consistencyMessages += "Further validation aborted."; 
	  }
	  else
	  {
		    numCheckFailures = sbmlDoc->checkConsistency();
		    if ( numCheckFailures > 0 )
		    {
			      noProblems = false;
			      for (unsigned int i = 0; i < numCheckFailures; i++)
			      {
					const SBMLError* sbmlErr = sbmlDoc->getError(i);
					if ( sbmlErr->isFatal() || sbmlErr->isError() )
					{
					  ++numValidationErrors;
					}
					else
					{
					  ++numValidationWarnings;
					}      
			      } 
			      ostringstream oss;
			      sbmlDoc->printErrors(oss);
			      validationMessages = oss.str(); 
		    }
	  }

	  if (noProblems)
	    return true;
	  else
	  {
		    if (numConsistencyErrors > 0)
		    {
		      cout << "ERROR: encountered " << numConsistencyErrors 
			   << " consistency error" << (numConsistencyErrors == 1 ? "" : "s")
			   << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
		    }
		    if (numConsistencyWarnings > 0)
		    {
		      cout << "Notice: encountered " << numConsistencyWarnings
			   << " consistency warning" << (numConsistencyWarnings == 1 ? "" : "s")
			   << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
		    }
		    cout << endl << consistencyMessages;

		    if (numValidationErrors > 0)
		    {
		      cout << "ERROR: encountered " << numValidationErrors
			   << " validation error" << (numValidationErrors == 1 ? "" : "s")
			   << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
		    }
		    if (numValidationWarnings > 0)
		    {
		      cout << "Notice: encountered " << numValidationWarnings
			   << " validation warning" << (numValidationWarnings == 1 ? "" : "s")
			   << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
		    }
		    cout << endl << validationMessages;

		    return (numConsistencyErrors == 0 && numValidationErrors == 0);
	  }
}


