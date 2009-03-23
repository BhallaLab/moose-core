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
  	unitdef->setId("litre_per_second");

  	// Create an Unit inside the UnitDefinition object above:
  	unit = unitdef->createUnit();
	unit->setKind(UNIT_KIND_LITRE);
	unit->setExponent(1);

  	unit = unitdef->createUnit();
  	unit->setKind(UNIT_KIND_SECOND);
  	unit->setExponent(-1);

  	// Create an UnitDefinition object for "litre_per_mole_per_second".  
    
  	unitdef = model->createUnitDefinition();
 	unitdef->setId("litre_per_mole_per_second");
    
	// Create individual unit objects that will be put inside
  	// the UnitDefinition to compose "litre_per_mole_per_second".

	unit = unitdef->createUnit();
	unit->setKind(UNIT_KIND_LITRE);
	unit->setExponent(1);

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
 	unitdef->setId("litre_sq_per_mole_per_second");
    	// Create individual unit objects that will be put inside
  	// the UnitDefinition to compose "litre_sq_per_mole_sq_per_second".

	 unit = unitdef->createUnit();
	 unit->setKind(UNIT_KIND_MOLE);
	 unit->setExponent(-1);
	 unit->setScale(-6);

	 unit = unitdef->createUnit();
	 unit->setKind(UNIT_KIND_LITRE);
	 unit->setExponent(2);

	 unit = unitdef->createUnit();
	 unit->setKind(UNIT_KIND_SECOND);
	 unit->setExponent(-1);	
	 
 	// Create a string for the identifier of the compartment.  

	static const Cinfo* kincomptCinfo = initKinComptCinfo();
	static const Finfo* sizeFinfo = kincomptCinfo->findFinfo( "size" );
	static const Finfo* dimensionFinfo = kincomptCinfo->findFinfo( "numDimensions" );
	Eref comptEl;
	vector< Id > compts;
	vector< Id >::iterator itr;
	wildcardFind("/kinetics/##[TYPE=KinCompt]", compts);
 	vector< Eref > outcompt;	
	bool flag = true;
	if (compts.empty())     
	{
		Id kinetics( "/kinetics" );//here kinetics is considered as the default compartment as no compartment is specified in the model.
		compts.push_back(kinetics); 
		flag = false;
	}
	for (itr = compts.begin(); itr != compts.end(); itr++)
	
	{
		comptEl = ( *itr )();
		::Compartment* compt = model->createCompartment();
		string comptName = comptEl.name();
		comptName = nameString(comptName);
		compt->setId(comptName);		
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
			cerr << "Warning: SbmlWriter: Too many outer compartments for " << comptName << ". Ignoring all but one.\n";
		
			if ( outcompt.size() >= 1 ) {
				string outName = outcompt[ 0 ].name();
				compt->setOutside(outName);
			
			}
		}
		// Create the Species objects inside the Model object. 
		static const Cinfo* moleculeCinfo = initMoleculeCinfo();	
		static const Finfo* nInitFinfo = moleculeCinfo->findFinfo( "nInit" );	
		static const Finfo* concInitFinfo = moleculeCinfo->findFinfo( "concInit" );
		static const Finfo* modeFinfo = moleculeCinfo->findFinfo( "mode" );
		
		Eref moleEl;	
		vector< Id > molecs;
		vector< Id >::iterator mitr;
		string comptPath = comptEl.id().path();
		wildcardFind(comptPath + "/#[TYPE=Molecule]", molecs);
		for (mitr = molecs.begin(); mitr != molecs.end(); mitr++)
		{		
			moleEl = ( *mitr )();		
			Species *sp = model->createSpecies();
			string molName = (moleEl)->name();
			molName = nameString(molName);
			sp->setId(molName);
			Id parent=Neutral::getParent(moleEl);
			string parentCompt=parent()->name();
			parentCompt = nameString(parentCompt);
			sp->setCompartment(parentCompt);
			//SBase *sb = sp;
			//sb->setSBOTerm (0000247);
			int mode;		
			get< int >(moleEl,modeFinfo,mode);
			if (mode==0){
				sp->setConstant(false);
				sp->setBoundaryCondition(false); 
			}	
			else{
				sp->setConstant(true);
				sp->setBoundaryCondition(true); 
			}
			unsigned int dimension;
		        get< unsigned int >(parent.eref(),dimensionFinfo,dimension);
			/*double initamt = 0.0;
			get< double >(moleEl,nInitFinfo,initamt); 
			initamt /= 6e26 ; //b'coz given unit of initamt is mol/litre and hence the conversion factor is 6e26
			//sp->setInitialAmount(initamt);*/			
			double initconc = 0.0;
			get< double >(moleEl,concInitFinfo,initconc); 
			sp->setInitialConcentration(initconc);
			
			/* Enzyme */
			vector< Id > enzms;
			string molecPath = moleEl.id().path();
			cout<<"molecpath:"<<molecPath<<endl;	
			wildcardFind(molecPath + "/#[TYPE=Enzyme]", enzms);
			printEnzymes(enzms,parentCompt,size,model);
			
		
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
					Reaction* reaction;
			  		SpeciesReference* spr;
					//ModifierSpeciesReference * mspr;	
			 		KineticLaw* kl;
					Parameter*  para; 
			  		
			  		// Create Reactant objects inside the Reaction object 

			 		reaction = model->createReaction();
					string rtnName = (rectnEl)->name();
					rtnName = nameString(rtnName);
					reaction->setId(rtnName);
					//cout<<"reaction :"<<(rectnEl)->name()<<endl;
					cout<<"reaction :"<<rtnName<<endl;
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
						spr = reaction->createReactant();
						string rctName = (*ri).name();	
						rctName = nameString(rctName);
						//cout<<"rct name :"<<rctName<<endl;	
						spr->setSpecies(rctName);				
						rctstoch=count(rct.begin(),rct.end(),*ri);
						spr->setStoichiometry(rctstoch);
						//cout<<"stoichiometry :"<<rctstoch<<endl;
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
						spr = reaction->createProduct();
						string pdtName = (*pi).name();	
						pdtName = nameString(pdtName);
						//cout<<"pdt name :"<<pdtName<<endl;	
						spr->setSpecies(pdtName);				
						pdtstoch=count(pdt.begin(),pdt.end(),*pi);
						spr->setStoichiometry(pdtstoch);
						//cout<<"stoichiometry :"<<pdtstoch<<endl;
						pdt_order += pdtstoch;
						
					}
					cout<<"pdt_order is "<<pdt_order<<endl;
			
					// Create a KineticLaw object inside the Reaction object 
					ostringstream rate_law,kfparm,kbparm;
			
					if (kf != 0.0 ){
						kfparm<<rtnName<<"_"<<"kf";
						kbparm<<rtnName<<"_"<<"kb";
						/*if (kb != 0.0 )
							rate_law <<comptName<<"*"<<"("<<kfparm.str();
							
						else
							rate_law <<comptName<<"*"<<kfparm.str();*/
						rate_law <<kfparm.str();
						double rstoch,r_order;
						for(ri = rctUniq.begin(); ri != rctUniq.end(); ri++)
						{
							rstoch = count(rct.begin(),rct.end(),*ri);
							r_order += rstoch;
							string riName = nameString((*ri).name());	
							if (rstoch == 1)
								rate_law <<"*"<<riName;
							else
								rate_law <<"*"<<riName<<"^"<<rstoch;
					
						} 
				
					}
					if (kb != 0.0){
						rate_law <<"-"<<kbparm.str();
						int pdtcount;
						std::set < Eref > ::iterator i;
						pdtcount=pdtUniq.size();
						for(pi = pdtUniq.begin(); pi != pdtUniq.end(); pi++)
						{	
							pdtcount -= 1 ;
							//cout<<"pdt size"<<pdtUniq.size()<<endl;
							pdtstoch=count(pdt.begin(),pdt.end(),*pi);
							string piName = nameString((*pi).name());	
							if (pdtstoch == 1)
								rate_law <<"*"<<piName;
							else
								rate_law <<"*"<<piName<<"^"<<pdtstoch;
							/*if (pdtcount == 0)
								rate_law <<")";*/
		
						} 
					}
					cout<<"rate_law "<<rate_law.str()<<endl; 
			
					kl  = reaction->createKineticLaw();
					kl->setFormula(rate_law.str());

					// Create local Parameter objects inside the KineticLaw object. 
		
					para = kl->createParameter();
					para->setId(kfparm.str());
					string unit=parmUnit(rct_order-1);
					para->setUnits(unit);
					double rvalue,pvalue;
			
					if (rct_order == 1)
						//rvalue = kf/size;
						rvalue = kf * size; 
			
					else{ 
						//double m = pow(6e26,rct_order-1);
						double m = 1.66053873e-21;
						//double NA = 6.02214199e23; //Avogardo's number	
						//rvalue = kf * pow((m * size),rct_order-1)/(NA * size);
						// recent : rvalue = kf * pow((m * size),rct_order-1) * size;
						rvalue = (kf * size)/(pow((m / size),rct_order-1));
						
					}
					para->setValue(rvalue*1e3); //to convert m^3 to litre

					if (kb != 0.0){
						if (pdt_order == 1)
							//pvalue = kb/size;
							pvalue = kb * size;
									
						else{
							//double m = pow(6e26,pdt_order-1);
							double m = 1.66053873e-21; // 1uM=6.023e17/1e-3 m^3==>6.023e20 .so M is 1/6.023e20
							//double NA = 6.02214199e23; //Avogardo's number	
							//pvalue = kb * pow((m * size),pdt_order-1)/(NA * size);
							// recent :pvalue = kb * pow((m * size),pdt_order-1)* size;
							pvalue = (kb * size)/(pow((m / size),pdt_order-1));
							
						}
						para = kl->createParameter();
						para->setId(kbparm.str());
						string unit=parmUnit(pdt_order-1);
						para->setUnits(unit);
						para->setValue(pvalue*1e3); //to convert m^3 to litre	
					}
				} //reaction
		
		
 	} //compartment
 	
	return sbmlDoc;

}
void SbmlWriter::printParameters(KineticLaw* kl,string k,double kvalue,string unit)
{
	Parameter* para = kl->createParameter();
	para->setId(k);
	para->setValue(kvalue);
	para->setUnits(unit);
}
void SbmlWriter::printReactants(Reaction* reaction,vector< Eref > enz,ostringstream& rlaw)
{
	for (int i=0; i<enz.size();i++)
	{	
		string enzName = enz[ i ].name();		
		cout<<"name :"<<enzName<<endl;
		SpeciesReference* spr = reaction->createReactant();
		spr->setSpecies(enzName);
		rlaw<<"*"<<enzName;
				
	}
}
void SbmlWriter::printProducts(Reaction* reaction,vector< Eref > cplx,ostringstream& rlaw)
{
	for (int i=0; i<cplx.size();i++)
	{	
		string cplxName = cplx[ i ].name();		
		cout<<"complex name :"<<cplxName<<endl;
		SpeciesReference* spr = reaction->createProduct();
		spr->setSpecies(cplxName);
		bool rev=reaction->getReversible();
		if (rev)
			rlaw<<"*"<<cplxName;
	}
}
/* Enzyme */
void SbmlWriter::printEnzymes(vector< Id > enzms,string parentCompt,double size,Model* model)
{	
	Eref enzEl;	
	vector< Id >::iterator ezitr;
	vector< Eref > enz;
	vector< Eref > sub;
	vector< Eref > cplx;
	ostringstream rlaw;
	//string molecPath = moleEl.id().path();
	//cout<<"molecpath:"<<molecPath<<endl;	
	for (ezitr = enzms.begin(); ezitr != enzms.end(); ezitr++)
	{		
		enzEl = ( *ezitr )();
		string enzName = (enzEl)->name();
		enzName = nameString(enzName);
		cout<<"enzyme name :"<<enzName<<endl;	
		Id enzParent=Neutral::getParent(enzEl);
		string parentMole=enzParent()->name();
		parentMole = nameString(parentMole);
		//ModifierSpeciesReference * mspr = model->createModifier();
		//mspr->setSpecies(parentMole);	
		cout<<"parent mole :"<<parentMole<<endl;
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
	  	cout<<"k1 = "<<k1<<endl;
		cout<<"k2 = "<<k2<<endl;
		cout<<"k3 = "<<k3<<endl;
		cout<<"km = "<<Km<<endl;
		cout<<"kcat = "<<kcat<<endl;
		cout<<"mode = "<<emode<<endl;
		if (emode ==  0){
			Reaction* react;
			SpeciesReference* spr;
			KineticLaw* kl;
			//rlaw<<parentCompt<<"*"<<"("<<"k1";
			rlaw<<"k1";
			react = model->createReaction();
			react->setId("Complex_formation");
			react->setMetaId("reaccomplex");
			react->setReversible(true);
			string reaction_notes = "<body xmlns:moose=\"http://www.moose.ncbs.res.in\">\n\t\t";
			reaction_notes += "<moose:EnzymaticReaction> \n";
			reaction_notes += "<moose:enzyme>E</moose:enzyme> \n";
			reaction_notes += "<moose:substrates>S</moose:substrates> \n";
			reaction_notes += "<moose:complex>kenz_cplx</moose:complex> \n";
			reaction_notes += "<moose:groupName>1</moose:groupName> \n";
			reaction_notes += "<moose:stage>1</moose:stage> \n";
			reaction_notes += "</moose:EnzymaticReaction> \n";
			reaction_notes += "</body>";
			//cout<<"reaction notes :"<<reaction_notes<<endl;
			SBase *sb = react;
			//sb->setNotes(reaction_notes);
			XMLNode* xnode =XMLNode::convertStringToXMLNode(reaction_notes);
			react->setAnnotation(xnode);
			enz.clear();
			targets(enzEl,"enz",enz);
			printReactants(react,enz,rlaw); //invoke function createReactants
			//cout<<"no of enzyme :"<<enz.size()<<endl;
			sub.clear();
			targets(enzEl,"sub",sub);
			printReactants(react,sub,rlaw); //invoke function createReactants
			//cout<<"no of substrates :"<<sub.size()<<endl;
			rlaw<<"-"<<"k2";			
			cplx.clear();
			targets(enzEl,"cplx",cplx);
			string cplxName = cplx[ 0 ].name();					
			Species *sp = model->createSpecies();
			sp->setId(cplxName);
			//sp->setMetaId("complex");
			//SBase *sb = sp;
			//sb->setSBOTerm(0000014);
			//initconc = 0.0;
			sp->setCompartment(parentCompt);
			//get< double >(moleEl,concInitFinfo,initconc); 
			sp->setInitialConcentration(0.0);
			/*get< double >(cplx[0],nInitFinfo,initamt); 
			initamt /= 6e26 ;
			sp->setInitialAmount(initamt);*/
			//sp->setNotes("<body xmlns=\"http://www.w3.org/1999/xhtml\">\n\t\t this is a complex species \n\t    </body>");
			printProducts(react,cplx,rlaw); //invoke function createProducts
			//cout<<"no of complex :"<<cplx.size()<<endl;
			//rlaw<<")";
			cout<<"rate law of complex formation :"<<rlaw.str()<<endl; 
			kl  = react->createKineticLaw();
			kl->setFormula(rlaw.str());
			//kl->setNotes("<body xmlns=\"http://www.w3.org/1999/xhtml\">\n\t\t" + rlaw.str() + "\n\t    </body>");
			double f = 1.66053873e-21;
			k1 = (k1 * size)/(pow((f / size),0));
			k1 = k1 * 1e3;
			k2 *= size;
			k2 = k2 * 1e3; //to convert m^3 to litre
			printParameters(kl,"k1",k1,"litre_sq_per_mole_per_second"); //set the parameters
			printParameters(kl,"k2",k2,"litre_per_second"); 	
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
			react->setId("Product_formation");
			react->setReversible(false);
			string reaction1_notes = "<body xmlns:moose=\"http://www.moose.ncbs.res.in\">\n\t\t";
			reaction1_notes += "<moose:EnzymaticReaction> \n";
			reaction1_notes += "<moose:enzyme>E</moose:enzyme> \n";
			reaction1_notes += "<moose:complex>kenz_cplx</moose:complex> \n";
			reaction1_notes += "<moose:products>P</moose:products> \n";
			reaction1_notes += "<moose:groupName>1</moose:groupName> \n";
			reaction1_notes += "<moose:stage>2</moose:stage> \n";
			reaction1_notes += "</moose:EnzymaticReaction> \n";
			reaction1_notes += "</body>";
			//cout<<"reaction notes :"<<reaction1_notes<<endl;
			//SBase *sb = react;
			//sb->setNotes(reaction_notes);
			XMLNode* x1node =XMLNode::convertStringToXMLNode(reaction1_notes);
			react->setAnnotation(x1node);
			wildcardFind(enzPath + "/#[TYPE=Molecule]", ezmole);
			for (emitr = ezmole.begin(); emitr != ezmole.end(); emitr++)
			{
				ezmoleEl = ( *emitr )();
				string ezMole = (ezmoleEl)->name();
				ezMole = nameString(ezMole);
				//law<<parentCompt<<"*"<<"k3";
				law<<"k3";
				cplx.clear();
				targets(enzEl,"cplx",cplx);
				printReactants(react,cplx,law); //invoke function createReactants
				//cout<<"no of complex :"<<cplx.size()<<endl;
				//cout<<"molecule name :"<<ezMole<<endl;
				Id ezMParent=Neutral::getParent(ezmoleEl);
				string parentEMole=ezMParent()->name();
				parentEMole = nameString(parentEMole);
				ezm.clear();
				targets(ezMParent(),"enz",ezm);
				printProducts(react,ezm,law); //invoke function createProducts
				//cout<<"no of enz :"<<ezm.size()<<endl;
				ezprd.clear();
				targets(ezMParent(),"prd",ezprd);
				printProducts(react,ezprd,law); //invoke function createProducts
				//cout<<"no of prds :"<<ezprd.size()<<endl;
				cout<<"rate law of product formation :"<<law.str()<<endl; 
				kl  = react->createKineticLaw();
				kl->setFormula(law.str());
				//kl->setNotes("<body xmlns=\"http://www.w3.org/1999/xhtml\">\n\t\t" + law.str() + "\n\t    </body>");
				k3 *= size;
				k3 = k3 * 1e3;
				printParameters(kl,"k3",k3,"litre_per_second"); //set the parameters
			} //enzyme molecule
		} //if
		//else if (emode == 0){
					
		//}
					
	} //enzyme	
}					

string SbmlWriter::parmUnit(double rct_order)
{
	ostringstream unit;
	int order = (int)rct_order;	
	switch(order)
	{
		case 0:
			unit<<"litre_per_second";
			break;		
		case 1:
			unit<<"litre_sq_per_mole_per_second";
			break;
		case 2:
			unit<<"litre_cube_per_mole_per_second";
			break;
		
		default:
			unit<<"litre_"<<rct_order+1<<"_per_mole_per_second";
			break;
		
	}
	return unit.str();

}
string SbmlWriter::nameString(string str)
{	
		
	string str1;
	int len = str.length();
	int i= 0;
	//cout<<"string is : "<<str<<" and str length is : "<<len<<endl;
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


