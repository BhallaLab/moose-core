/*******************************************************************
 * File:            SbmlReader.cpp
 * Description:
 * Author: HarshaRani
 * E-mail: hrani@ncbs.res.in
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
/****************************
* Change log:

* Originally created by Siji for l2v4 for 'trunk ' branch
* Modified / adapted to 'asyn13' branch Harsharani for both l2v4 and l3v1
***************/


#ifdef USE_SBML

#include <cmath>
#include <stdexcept>
#include <sbml/SBMLTypes.h>
#include <sbml/UnitDefinition.h>
#include <sbml/units/UnitFormulaFormatter.h>
#include <sbml/units/FormulaUnitsData.h>
#include <string>
#include <stdlib.h>
#include "header.h"
#include "../shell/Shell.h"
#include "../shell/Wildcard.h"
//#include "../manager/SimManager.h"
#include "SbmlReader.h"
#include "../kinetics/FuncPool.h"
#include "../external/debug/simple_logger.hpp"

using namespace std;
map< string,double > parmValueMap;
map< string,double> :: iterator pvm_iter;

/*  Harsha : TODO in
    -Compartment
      --Need to add group
      --Need to deal with compartment outside
    -Molecule
      -- Need to add group
      -- Func pool and its math calculation need to be added.
    -Loading Model from SBML
      --Tested 1-30 testcase example model provided by l3v1 and l2v4 std.
        ---These are the models that worked (sbml testcase)1-6,10,14-15,17-21,23-25,34,35,58
	---Need to check
	 ----what to do when boundarycondition is true i.e.,
             differential equation derived from the reaction definitions
             should not be calculated for the species(7-9,11-13,16)
         ----kineticsLaw, Math fun has fraction,ceiling,reminder,power 28etc.
         ----Events to be added 26
	 ----initial Assisgment for compartment 27
         ----when stoichiometry is rational number 22
	 ---- For Michaelis¡VMenten kinetics km is not defined which is most of the case
	      need to calculate
              */

/**
 * @brief Reads a given SBML file and loads it into MOOSE.
 *
 * @param filename Name of file, std::string.
 * @param location 
 * @param solverClass
 *
 * @return  Id on success. Some expcetion on failure.
 */
Id SbmlReader::read( string filename, string location, string solverClass) 
{
    FILE * fp = fopen( filename.c_str(), "r" );
    if ( fp == NULL) {
        stringstream ss;
        ss << "File " << filename << " does not exist." << endl;
        throw runtime_error(ss.str());
    }

    document_ = readSBML( filename.c_str() );
    unsigned num_errors = document_->getNumErrors();
    if ( num_errors > 0 ) {
        cerr << "Errors encountered while reading" << endl;
        document_->printErrors( cerr );
        errorFlag_ = true;
        return baseId;
    }

    model_= document_->getModel();
    if ( model_ == 0 ) {
        cout << "SBML: Error: No model present." << endl;
        errorFlag_ = true;
        return baseId;
    }
    if ( !errorFlag_ )
        getGlobalParameter();

    if ( !errorFlag_ ) {
        string modelName;
        Id parentId;
        findModelParent ( Id(), location, parentId, modelName ) ;
        Id parentId2 = parentId;
        Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
        /*  As a policy model is created under /model and all the graphs are created under /data, if this file is invoke from Gui the /model is already created and path is passed, but if readSbml is called then only modelName is taken so making sure /model is the model path                                                                    */
        //Id modelPath_ = s->doCreate("Neutral",parentId,"model",1,MooseGlobal);
        if (parentId == Id()) {
            Id parentId1 = s->doCreate("Neutral",parentId,"model",1,MooseGlobal);
            parentId = parentId1;
        }
        Id base_ = s->doCreate( "Neutral", parentId, modelName, 1, MooseGlobal);
            //Id base_ = s->doCreate( "SimManager", parentId, modelName, dims, true );
        assert( base_ != Id() );
        //Map Compartment's SBML id to Moose ID
        map< string,Id > comptSidMIdMap;
        // Map between Molecule's SBML id to which it belongs compartment Moose Id
        map< string, Id > molSidcmptMIdMap;

        if ( !errorFlag_ )
            comptSidMIdMap = createCompartment(location, parentId, modelName, base_);

        if ( !errorFlag_ )
            molSidcmptMIdMap = createMolecule( comptSidMIdMap);

        if ( !errorFlag_ )
            getRules();

        if ( !errorFlag_ )
            createReaction( molSidcmptMIdMap );
        // or we get
        //createReaction (result);

        if ( errorFlag_ )
            return baseId;
        else {
            // SimManager* sm = reinterpret_cast< SimManager* >(baseId.eref().data());
            //Shell* s = reinterpret_cast< Shell* >(baseId.eref().data());
            XMLNode * annotationNode = model_->getAnnotation();
            if( annotationNode != NULL ) {
                unsigned int num_children = annotationNode->getNumChildren();
                for( unsigned int child_no = 0; child_no < num_children; child_no++ ) {
                    XMLNode childNode = annotationNode->getChild( child_no );
                    if ( childNode.getPrefix() == "moose" && childNode.getName() == "ModelAnnotation" ) {
                        unsigned int num_gchildren = childNode.getNumChildren();
                        for( unsigned int gchild_no = 0; gchild_no < num_gchildren; gchild_no++ ) {
                            XMLNode &grandChildNode = childNode.getChild( gchild_no );
                            string nodeName = grandChildNode.getName();
                            if (grandChildNode.getNumChildren() == 1 ) {
                                string plotValue;
                                //double nodeValue;
                                if(nodeName == "plots") {
                                    // if plots exist then will be placing at "/data"
                                    Id graphs;
                                    //Id dataId;
                                    Id dataIdTest;
                                    if (parentId2 == Id())
                                        graphs = s->doCreate( "Neutral", parentId2, "data", 1);
                                    else
                                        // need to check how to put / while coming from gui as the path is /model/modelName??? 27 jun 2014
                                        findModelParent ( Id(), modelName, dataIdTest, modelName ) ;
                                        string test = "/data";
                                        Id tgraphs(test);
                                        graphs=tgraphs;
                                        //graphs = s->doCreate("Neutral",parentId,"data",1);
                                        //Id dataId;
                                        //if (dataId == Id())
                                        //    cout << "Id " << dataId;
                                        //    graphs = s->doCreate( "Neutral",dataId, "data", 1);
                                        assert( graphs != Id() );
                                    plotValue = (grandChildNode.getChild(0).toXMLString()).c_str();
                                    istringstream pltVal(plotValue);
                                    string pltClean;
                                    while (getline(pltVal,pltClean, ';')) {
                                        pltClean.erase( remove( pltClean.begin(), pltClean.end(), ' ' ), pltClean.end() );
                                        //string plotPath = location+pltClean;
                                        string plotPath = base_.path()+pltClean;
                                        Id plotSId(plotPath);
                                        unsigned pos = pltClean.find('/');
                                        if (pos != std::string::npos)
                                            pltClean = pltClean.substr(pos+1,pltClean.length());
                                        replace(pltClean.begin(),pltClean.end(),'/','_');
                                        string plotName =  pltClean + ".conc";
                                        Id pltPath(graphs.path());
                                        Id tab = s->doCreate( "Table", pltPath, plotName, 1 );
                                        if (tab != Id())
                                            s->doAddMsg("Single",tab,"requestOut",plotSId,"getConc");
                                    }//while
                                    /* passing /model and /data         */
                                    string comptPath =base_.path()+"/##";
                                    s->doUseClock(comptPath,"process",4);

                                    string tablePath = graphs.path()+"/##[TYPE=Table]";
                                    s->doUseClock( tablePath, "process",8 );
                                }//plots
                                /*else
                                  nodeValue = atof ((grandChildNode.getChild(0).toXMLString()).c_str());

                                  if (nodeName == "runTime")
                                  sm->setRunTime(nodeValue);
                                  else if (nodeName == "simdt")
                                  sm->setSimDt(nodeValue);
                                  else if(nodeName == "plotdt")
                                  sm->setPlotDt(nodeValue);
                                  */

                            } //grandChild
                            else
                                cout << "Warning: expected exactly ONE child of " << nodeName << " but none found "<<endl;
                        } //gchild
                    } //moose and modelAnnotation
                }
            }//annotation Node
            else {
                //4 for simdt and 8 for plotdt
                s->doUseClock(base_.path()+"/##","process",4);
                s->doUseClock(+"/data/##[TYPE=Table]","process",8);
                s->doSetClock(4,0.1);
                s->doSetClock(8,0.1);

            }
            vector< ObjId > compts;
            string comptpath = base_.path()+"/##[ISA=ChemCompt]";
            wildcardFind( comptpath, compts );
            vector< ObjId >::iterator i = compts.begin();
            string comptName = nameString(Field<string> :: get(ObjId(*i),"name"));
            string simpath = base_.path() + "/##";
            s->doUseClock( simpath, "process", 4 );

            //wildcardFind( plotpath, plots );
            //Id pathexist(base_.path()+"/kinetics");
            /*
               if (solverClass.empty())
               {
               if( pathexist != Id())
               sm->build(base_.eref(),&q,"rk5");
               else
               sm->buildForSBML(base_.eref(),&q,"rk5");
               }
               else
               { if(pathexist != Id())
               sm->build(base_.eref(),&q,solverClass);
               else
               sm->buildForSBML(base_.eref(),&q,solverClass);
               }
               */
            return base_;
        }

    } else
        return baseId;
}

/**
 * @brief Map SBML compartments to MOOSE.
 *
 * @param location 
 * @param parentId string. Name of parent compartment.
 * @param modelName string. Name of model.
 * @param base_ Id, Id of parent.
 *
 * @return std::map<string, Id>.
 */
map< string,Id > SbmlReader::createCompartment(string location, Id parentId, string modelName, Id base_) 
{
    /* In compartment: pending to add
       -- the group
       -- outside     -- units of volume
    */
    Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
    map< string,Id > comptSidMIdMap;
    map< string,string > outsideMap;
    map< string,string > ::iterator iter;
    double msize = 0.0, size = 0.0;

    ::Compartment* compt;
    unsigned int num_compts = model_->getNumCompartments();

    if (num_compts == 0) {
        errorFlag_ = true;
        stringstream ss;
        ss << "No compartment found in given model: " << endl;
        logger.dump("SBML", ss.str());
        return comptSidMIdMap;
    }

    baseId = base_;
    for ( unsigned int i = 0; i < num_compts; i++ ) {
        compt = model_->getCompartment(i);
        std::string id = "";
        if ( compt->isSetId() ) {
            id = compt->getId();
        }

        std::string name = "";
        if ( compt->isSetName() ) {
            name = compt->getName();
            name = nameString(name);
        }

        std::string outside = "";
        if ( compt->isSetOutside() ) {
            outside = compt->getOutside ();
        }
        if ( compt->isSetSize() ) {
            msize = compt->getSize();
        }

        UnitDefinition * ud = compt->getDerivedUnitDefinition();
        size = transformUnits( msize,ud , "compartment",0);
        unsigned int dimension = compt->getSpatialDimensions();

        if (dimension < 3)
            cout << "\n ###### Spatial Dimension is " << dimension <<" volume should not be converted from liter to cubicmeter which is happening as default check \n";

        ASSERT_FALSE(name.empty() && id.empty()
                , "Compartment name and id are empty"
                );

        if (name.empty()) {
            if(! id.empty() )
                name = id;
        }

//        cerr << "Creating under " << base_.path() << " : " << name << endl;
        Id compt = s->doCreate( "CubeMesh", base_, name,  1);
        comptSidMIdMap[id] = compt;
        if (size != 0.0)
            Field< double >::set( compt, "volume", size );
        if (dimension != 0)
            continue;
        //Field < int > :: set(compt, "numDimensions", dimension);
    }
    return comptSidMIdMap;
}

/* create MOLECULE  */
const SbmlReader::sbmlStr_mooseId SbmlReader::createMolecule( map< string,Id > &comptSidMIdMap) {
    Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
    map< string, Id >molSidcmptMIdMap;
    double transvalue = 0.0;
    int num_species = model_->getNumSpecies();

    if (num_species == 0) {
        baseId = Id();
        errorFlag_ = true;
        return molSidcmptMIdMap;
    }

    for ( int sindex = 0; sindex < num_species; sindex++ ) {
        Species* spe = model_->getSpecies(sindex);
        if (!spe) {
            continue;
        }
        std::string compt = "";
        if ( spe->isSetCompartment() ) {
            compt = spe->getCompartment();
        }
        if (compt.length()< 1) {
            //cout << "compt is empty for species "<< sindex << endl;
            continue;
        }
        string id = spe->getId();
        if (id.length() < 1) {
            continue;
        }
        std::string name = "";
        if ( spe->isSetName() ) {
            name = spe->getName();
            name = nameString(name);
        }
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
            for ( unsigned int r = 0; r < nr; r++ ) {
                Rule * rule = model_->getRule(r);
                bool assignRule = rule->isAssignment();
                if ( assignRule ) {
                    string rule_variable = rule->getVariable();
                    if (rule_variable.compare(id) == 0) {
                        found = true;
                        break;
                    }
                }
            }
            if (found == false) {
                cout << "Invalid SBML: Either initialConcentration or initialAmount must be set or it should be found in assignmentRule but non happening for " << spe->getName() <<endl;
                return molSidcmptMIdMap;
            }
        }
        Id comptEl = comptSidMIdMap[compt];
        Id meshEntry = Neutral::child( comptEl.eref(), "mesh" );
        bool constant = spe->getConstant();
        bool boundaryCondition = spe->getBoundaryCondition();
        if (boundaryCondition == true)
            cout << "Pools having BoundaryCondition true " << name <<endl;
        Id pool;
        //If constant is true then its equivalent to BuffPool in moose
        if (constant == true)
            //if( (boundaryCondition == true) && (constant==false))
            pool = shell->doCreate("BufPool",comptEl,name,1);
        else
            pool = shell->doCreate("Pool", comptEl, name ,1);
        molSidcmptMIdMap[id] = comptEl;
        //Map to Molecule SBML id to Moose Id
        molSidMIdMap_[id] = pool;

        //shell->doAddMsg( "OneToOne",pool, "mesh", meshEntry, "mesh" );
        bool bcondition = spe->getBoundaryCondition();
        if ( constant == true && bcondition == false)
            cout <<"The species "<< name << " should not appear in reactant or product as per sbml Rules"<< endl;

        unsigned int spatialDimen =Field< unsigned int >::get( comptEl, "numDimensions");

        UnitDefinition * ud = spe->getDerivedUnitDefinition();
        assert(ud != NULL);
        bool hasonlySubUnit = spe->getHasOnlySubstanceUnits();

        //double v = Field< double >::get( comptEl.path(), "volume" );
        transvalue = transformUnits(1,ud,"substance",hasonlySubUnit);
        if (hasonlySubUnit) {
            // In Moose, no. of molecules (nInit) and unit is "item"
            if (spatialDimen > 0 && spe->isSetInitialAmount() ) {
                //transvalue *= initvalue;
                initvalue *=transvalue;
                Field < double> :: set( pool, "nInit", initvalue);
            }
        } else {
            //transvalue *=initvalue;
            initvalue *=transvalue;
            Field <double> :: set(pool, "concInit",initvalue);
        }
        //cout << " poolMap_ "<< name << " " <<transvalue << " "<< hasonlySubUnit;
        //poolMap_.insert(make_pair(id,make_tuple(name,transvalue,hasonlySubUnit)));
    }
    return molSidcmptMIdMap;

}

/* Assignment Rule */

void SbmlReader::getRules() {
    unsigned int nr = model_->getNumRules();
    //if (nr > 0)
    //  cout << "\n ##### Need to populate funcpool and sumtotal which is pending due to equations \n";
    Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
    for ( unsigned int r = 0; r < nr; r++ ) {

        Rule * rule = model_->getRule(r);
        bool assignRule = rule->isAssignment();
        if ( assignRule ) {
            string rule_variable = rule->getVariable();
            map< string,Id >::iterator v_iter;
            map< string,Id >::iterator m_iter;
            v_iter = molSidMIdMap_.find( rule_variable );
            if (v_iter != molSidMIdMap_.end()) {
                Id rVariable = molSidMIdMap_.find(rule_variable)->second;
                string rstring =molSidMIdMap_.find(rule_variable)->first;
                Id sumId = shell->doCreate( "SumFunc", rVariable, "func", 1 );
                rVariable.element()->zombieSwap( FuncPool::initCinfo() );
                ObjId ret = shell->doAddMsg( "single",
                                             ObjId( sumId, 0 ), "output",
                                             ObjId( rVariable, 0 ), "input" );
                assert( ret != ObjId() );
                const ASTNode * ast = rule->getMath();
                vector< string > ruleMembers;
                ruleMembers.clear();
                printMembers( ast,ruleMembers );
                for ( unsigned int rm = 0; rm < ruleMembers.size(); rm++ ) {
                    m_iter = molSidMIdMap_.find( ruleMembers[rm] );
                    if ( m_iter != molSidMIdMap_.end() ) {
                        Id rMember = molSidMIdMap_.find(ruleMembers[rm])->second;
                        ObjId ret = shell->doAddMsg( "single",
                                                     ObjId( rMember, 0 ), "nOut",
                                                     ObjId( sumId, 0 ), "input" );
                        string test = molSidMIdMap_.find(ruleMembers[rm])->first;
                    } else {
                        cerr << "SbmlReader::getRules: Assignment rule member is not a species" << endl;
                        // In assignment rule there are constants instead of molecule which is yet to deal in moose.
                        errorFlag_ = true;
                    }
                }
            }
        }
        bool rateRule = rule->isRate();
        if ( rateRule ) {
            cout << "warning : for now Rate Rule is not handled " << endl;
            errorFlag_ = true;
        }
        bool  algebRule = rule->isAlgebraic();
        if ( algebRule ) {
            cout << "warning: for now Algebraic Rule is not handled" << endl;
            errorFlag_ = true;
        }
    }
}

//REACTION

void SbmlReader::createReaction(const map< string, Id > &molSidcmptMIdMap ) {
    Reaction* reac;

    map< string,double > rctMap;
    map< string,double >::iterator rctMap_iter;
    map< string,double >prdMap;
    map< string,double >::iterator prdMap_iter;
    map< string,EnzymeInfo >enzInfoMap;

    for ( unsigned int r = 0; r < model_->getNumReactions(); r++ ) {
        Id reaction_;
        reac = model_->getReaction( r );
        noOfsub_ = 0;
        noOfprd_ = 0;
        std:: string id; //=reac->getId();
        if ( reac->isSetId() )
            id = reac->getId();

        std::string name;
        if ( reac->isSetName() ) {
            name = reac->getName();
            name = nameString(name);
        }
        if (name.empty()) {
            if (id.empty())
                assert("Reaction id and name is empty");
            else
                name = id;
        }
        string grpname = getAnnotation( reac,enzInfoMap );
        if ( (grpname != "") && (enzInfoMap[grpname].stage == 3) ) {
            setupEnzymaticReaction( enzInfoMap[grpname],grpname ,molSidcmptMIdMap,name);
        }
        //if (grpname != "")
        // {
        //cout << "\n enz matic reaction " << enzInfoMap[grpname].stage;
        //setupEnzymaticReaction( enzInfoMap[grpname],grpname ,molSidcmptMIdMap);
        //}

        else if ( grpname == "" ) {
            if (reac->getNumModifiers() > 0)
                setupMMEnzymeReaction( reac,id,name ,molSidcmptMIdMap);
            else {
                bool rev=reac->getReversible();
                bool fast=reac->getFast();
                if ( fast ) {
                    cout<<"warning: for now fast attribute is not handled"<<endl;
                    errorFlag_ = true;
                }
                int numRcts = reac->getNumReactants();
                int numPdts = reac->getNumProducts();
                if ( numRcts == 0 && numPdts != 0 ) {
                    cout << "Reaction with zero Substrate is not possible but exist in this model";
                    const SpeciesReference* pdt = reac->getProduct( 0 );
                    std::string spName = pdt->getSpecies();
                    Id parent = molSidcmptMIdMap.find( spName )->second; //gives compartment of spName
                    cout << " \n \t ################################# Sub = 0 and prd != 0 need to the reac ############### ";
                    const SpeciesReference* rect=reac->getReactant(0);
                    std::string sp=rect->getSpecies();
                    Id comptRef = molSidcmptMIdMap.find(sp)->second; //gives compartment of sp
                    Id meshEntry = Neutral::child( comptRef.eref(), "mesh" );
                    Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
                    reaction_ = shell->doCreate("Reac", meshEntry, name, 1);
                    //shell->doAddMsg( "Single", meshEntry, "remeshReacs", reaction_, "remesh");
                    //Get Substrate
                    addSubPrd(reac,reaction_,"prd");
                } //if numRcts == 0
                else {
                    const SpeciesReference* rect=reac->getReactant(0);
                    std::string sp=rect->getSpecies();
                    Id comptRef = molSidcmptMIdMap.find(sp)->second; //gives compartment of sp
                    Id meshEntry = Neutral::child( comptRef.eref(), "mesh" );
                    Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );

                    reaction_ = shell->doCreate("Reac", comptRef, name, 1);
                    //shell->doAddMsg( "Single", meshEntry, "remeshReacs", reaction_, "remesh");
                    //Get Substrate
                    addSubPrd(reac,reaction_,"sub");

                    //Get Product
                    addSubPrd(reac,reaction_,"prd");
                }
                if ( reac->isSetKineticLaw() ) {
                    KineticLaw * klaw=reac->getKineticLaw();

                    //vector< double > rate = getKLaw( klaw,rev );
                    vector< double > rate;
                    rate.clear();
                    getKLaw( klaw,rev,rate );
                    if ( errorFlag_ )
                        return;
                    else if ( !errorFlag_ ) {
                        //cout << " Reaction name " << name << " kf " << rate[0] << " kb " << rate[1]<<endl;

                        Field < double > :: set( reaction_, "Kf", rate[0] );
                        Field < double > :: set( reaction_, "Kb", rate[1] );
                        /*if (numRcts > 1)
                        rate[0] = rate[0]*pow(1e3,1.0);
                             cout << "Reaction " << id << " " << name << " " << rate[0] << "  " << rate[1]<<endl;
                             Field < double > :: set( reaction_, "Kf", rate[0] );
                             Field < double > :: set( reaction_, "Kb", rate[1] );
                             */
                    }
                } //issetKineticLaw

            } //else
        } // else grpname == ""
    }//for unsigned
} //reaction

/* Enzymatic Reaction  */
void SbmlReader::setupEnzymaticReaction( const EnzymeInfo & einfo,string enzname, const map< string, Id > &molSidcmptMIdMap,string name1) {
    string enzPool = einfo.enzyme;

    Id comptRef = molSidcmptMIdMap.find(enzPool)->second; //gives compartment of sp
    Id meshEntry = Neutral::child( comptRef.eref(), "mesh" );
    Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );

    //Creating enz pool to enzyme site
    Id enzPoolId = molSidMIdMap_.find(enzPool)->second;
    Id enzyme_ = shell->doCreate("Enz", enzPoolId, name1, 1);
    //shell->doAddMsg( "Single", meshEntry, "remeshReacs", enzyme_, "remesh");

    Id complex = einfo.complex;
    //Moving enzyme site under enzyme
    shell->doMove(complex,enzyme_);
    shell->doAddMsg("OneToAll",enzyme_,"cplx",complex,"reac");

    shell->doAddMsg("OneToOne",enzyme_,"enz",enzPoolId,"reac");

    vector< Id >::const_iterator sub_itr;
    for ( sub_itr = einfo.substrates.begin(); sub_itr != einfo.substrates.end(); sub_itr++ ) {
        Id S = (*sub_itr);
        Id b = shell->doAddMsg( "OneToOne", enzyme_, "sub" ,S , "reac" );
    }

    vector< Id >::const_iterator prd_itr;
    for ( prd_itr = einfo.products.begin(); prd_itr != einfo.products.end(); prd_itr++ ) {
        Id P = (*prd_itr);
        shell->doAddMsg ("OneToOne",enzyme_,"prd", P,"reac");
    }
    // populate k3,k2,k1 in this order only.
    Field < double > :: set( enzyme_, "k3", einfo.k3 );
    Field < double > :: set( enzyme_, "k2", einfo.k2 );
    Field < double > :: set( enzyme_, "k1", einfo.k1 );
}

/*  get annotation  */
string SbmlReader::getAnnotation( Reaction* reaction,map<string,EnzymeInfo> &enzInfoMap ) {
    XMLNode * annotationNode = reaction->getAnnotation();
    EnzymeInfo einfo;
    string grpname = "",stage;

    if( annotationNode != NULL ) {
        unsigned int num_children = annotationNode->getNumChildren();
        for( unsigned int child_no = 0; child_no < num_children; child_no++ ) {
            XMLNode childNode = annotationNode->getChild( child_no );
            if ( childNode.getPrefix() == "moose" && childNode.getName() == "EnzymaticReaction" ) {
                unsigned int num_gchildren = childNode.getNumChildren();
                for( unsigned int gchild_no = 0; gchild_no < num_gchildren; gchild_no++ ) {
                    XMLNode &grandChildNode = childNode.getChild( gchild_no );
                    string nodeName = grandChildNode.getName();
                    string nodeValue;
                    if (grandChildNode.getNumChildren() == 1 ) {
                        nodeValue = grandChildNode.getChild(0).toXMLString();

                    } else {
                        cout << "Error: expected exactly ONE child of " << nodeName << endl;
                    }
                    if ( nodeName == "enzyme" )
                        einfo.enzyme = molSidMIdMap_.find(nodeValue)->first;

                    else if ( nodeName == "complex" )
                        einfo.complex=molSidMIdMap_.find(nodeValue)->second;

                    else if ( nodeName == "substrates") {
                        Id elem = molSidMIdMap_.find(nodeValue)->second;
                        einfo.substrates.push_back(elem);
                    } else if ( nodeName == "product" ) {
                        Id elem = molSidMIdMap_.find(nodeValue)->second;
                        einfo.products.push_back(elem);
                    } else if ( nodeName == "groupName" )
                        grpname = nodeValue;
                    else if ( nodeName == "stage" )
                        stage = nodeValue;
                }
                if ( stage == "1" ) {
                    enzInfoMap[grpname].substrates = einfo.substrates;
                    enzInfoMap[grpname].enzyme = einfo.enzyme;
                    einfo.stage = 1;
                    enzInfoMap[grpname].stage = einfo.stage;
                    KineticLaw * klaw=reaction->getKineticLaw();
                    vector< double > rate ;
                    rate.clear();
                    getKLaw( klaw,true,rate );
                    if ( errorFlag_ )
                        exit(0);
                    else if ( !errorFlag_ ) {
                        enzInfoMap[grpname].k1 = rate[0];
                        enzInfoMap[grpname].k2 = rate[1];
                    }
                }
                //Stage =='2' means ES* -> E+P;
                else if ( stage == "2" ) {
                    enzInfoMap[grpname].complex = einfo.complex;
                    enzInfoMap[grpname].products = einfo.products;
                    einfo.stage = 2;
                    enzInfoMap[grpname].stage += einfo.stage;
                    KineticLaw * klaw=reaction->getKineticLaw();
                    vector< double > rate;
                    rate.clear();
                    getKLaw( klaw,false,rate );
                    if ( errorFlag_ )
                        exit(0);
                    else if ( !errorFlag_ )
                        enzInfoMap[grpname].k3 = rate[0];
                }
            }
        }
    }
    return grpname;
}

/*    set up Michalies Menten reaction  */
void SbmlReader::setupMMEnzymeReaction( Reaction * reac,string rid,string rname,const map< string, Id > &molSidcmptMIdMap ) {
    string::size_type loc = rid.find( "_MM_Reaction_" );
    if( loc != string::npos ) {
        int strlen = rid.length();
        rid.erase( loc,strlen-loc );
    }
    unsigned int num_modifr = reac->getNumModifiers();
    for ( unsigned int m = 0; m < num_modifr; m++ ) {
        const ModifierSpeciesReference* modifr=reac->getModifier( m );
        std::string sp = modifr->getSpecies();
        Id enzyme_;
        Id E = molSidMIdMap_.find(sp)->second;

        Id comptRef = molSidcmptMIdMap.find(sp)->second; //gives compartment of sp
        Id meshEntry = Neutral::child( comptRef.eref(), "mesh" );
        Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
        enzyme_ = shell->doCreate("MMenz",E,rname,1);
        //shell->doAddMsg( "Single", meshEntry, "remeshReacs", enzyme_, "remesh");
        shell->doAddMsg("Single",E,"nOut",enzyme_,"enzDest");

        KineticLaw * klaw=reac->getKineticLaw();
        vector< double > rate;
        rate.clear();
        getKLaw( klaw,true,rate );
        if ( errorFlag_ )
            return;
        else if ( !errorFlag_ ) {
            for ( unsigned int rt = 0; rt < reac->getNumReactants(); rt++ ) {
                const SpeciesReference* rct = reac->getReactant( rt );
                sp=rct->getSpecies();
                Id S = molSidMIdMap_.find(sp)->second;
                shell->doAddMsg( "OneToOne", enzyme_, "sub" ,S , "reac" );
            }
            for ( unsigned int pt = 0; pt < reac->getNumProducts(); pt++ ) {
                const SpeciesReference* pdt = reac->getProduct(pt);
                sp = pdt->getSpecies();
                Id P = molSidMIdMap_.find(sp)->second;
                shell->doAddMsg( "OneToOne", enzyme_, "prd" ,P, "reac" );
            }
            Field < double > :: set( enzyme_, "kcat", rate[0] );
            Field < double > :: set( enzyme_, "numKm", rate[1] );
        }

    }
}

/*    get Parameters from Kinetic Law  */
void SbmlReader::getParameters( const ASTNode* node,vector <string> & parameters ) {
    assert( parameters.empty() );
    //cout << " parameter type " <<node->getType();

    if ( node->getType() == AST_MINUS ) {
        const ASTNode* lchild = node->getLeftChild();
        pushParmstoVector( lchild,parameters );

        if ( parameters.size() == 1 ) {
            const ASTNode* rchild = node->getRightChild();
            pushParmstoVector( rchild,parameters );
        }
    } else if ( node->getType() == AST_DIVIDE ) {
        const ASTNode* lchild = node->getLeftChild();
        pushParmstoVector( lchild,parameters );
        if (( parameters.size() == 1 ) || ( parameters.size() == 0 )) {
            const ASTNode* rchild = node->getRightChild();
            pushParmstoVector( rchild,parameters );
        }
    } else if ( node->getType() == AST_TIMES ) {
        //cout << " time " <<endl;
        pushParmstoVector( node,parameters );
    } else if ( node->getType() == AST_PLUS )
        pushParmstoVector( node,parameters );
    else if ( node->getType() == AST_NAME )
        pushParmstoVector( node,parameters );
    if ( parameters.size() > 2 ) {
        cout<<"Sorry! for now MOOSE cannot handle more than 2 parameters ."<<endl;
        errorFlag_ = true;
    }

}

/*   push the Parameters used in Kinetic law to a vector  */

void SbmlReader::pushParmstoVector(const ASTNode* p,vector <string> & parameters) {
    string parm = "";
    //cout << "\n there " << p->getType();
    //cout << "_NAME" << " = " <<p->getName();
    if ( p->getType() == AST_NAME ) {
        pvm_iter = parmValueMap.find( std::string(p->getName()) );
        if ( pvm_iter != parmValueMap.end() ) {
            parm = pvm_iter->first;
            parameters.push_back( parm );
        }
    }
    int num = p->getNumChildren();
    for( int i = 0; i < num; ++i ) {
        const ASTNode* child = p->getChild(i);
        pushParmstoVector( child,parameters );
    }
}

/*     get Kinetic Law  */
void SbmlReader::getKLaw( KineticLaw * klaw,bool rev,vector< double > & rate ) {
    std::string id;
    double value = 0.0;
    UnitDefinition * kfud;
    UnitDefinition * kbud;
    int np = klaw->getNumParameters();
    bool flag = true;
    for ( int pi = 0; pi < np; pi++ ) {
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
    else if ( !errorFlag_ ) {
        if ( parameters.size() == 1 ) {
            kfparm = parameters[0];
            kbparm = parameters[0];
        } else if ( parameters.size() == 2 ) {
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
        if ( flag ) {
            kfp = model_->getParameter( kfparm );
            kbp = model_->getParameter( kbparm );
        } else {
            kfp = klaw->getParameter( kfparm );
            kbp = klaw->getParameter( kbparm );
        }
        //cout << "\t \n \n" << kfp << " " <<kbp;

        if ( kfp->isSetUnits() ) {
            kfud = kfp->getDerivedUnitDefinition();
            //cout << "parameter unit :" << UnitDefinition::printUnits(kfp->getDerivedUnitDefinition())<< endl;
            double transkf = transformUnits( 1,kfud ,"substance",true);
            kf = kfvalue * transkf;
            kb = 0.0;
        } else if (! kfp->isSetUnits() ) {
            double lvalue =1.0;
            /* If rate units are not defined then trying to get substanceUnit*/
            if (model_->getNumUnitDefinitions() > 0)
                lvalue = unitsforRates();
            /* If neither RateUnits nor substanceUnit is defined, then assuming SubstanceUnit are in mole,
               so converting mole to millimole
            */
            if (noOfsub_ >1)
                lvalue /= pow(1e+3,(noOfsub_-1));
            kf = kfvalue*lvalue;
        }// !kfp is notset
        if ( ( kbp->isSetUnits() ) && ( rev ) ) {
            kbud = kbp->getDerivedUnitDefinition();
            double transkb = transformUnits( 1,kbud,"substance",true );
            kb = kbvalue * transkb;
        }
        if ( (! kbp->isSetUnits() ) && ( rev ) ) {
            double lvalue =1.0;
            /* If rate units are not defined then,trying to get substanceUnit*/
            if (model_->getNumUnitDefinitions() > 0)
                lvalue = unitsforRates();
            /* If neither RateUnits nor substanceUnit is defined, then assuming SubstanceUnit are in mole (concentration)
               and hasOnlySubstanceUnit =false so converting mole to millimole
            */
            if (noOfprd_ >1)
                lvalue /= pow(1e+3,(noOfprd_-1));
            kf = kfvalue*lvalue;

        }
        rate.push_back( kf );
        rate.push_back( kb );
    }
}
double SbmlReader::unitsforRates() {
    double lvalue =1;
    for (unsigned int n=0; n < model_->getNumUnitDefinitions(); n++) {
        UnitDefinition * ud = model_->getUnitDefinition(n);
        for (unsigned int ut=0; ut <ud->getNumUnits(); ut++) {
            Unit * unit = ud->getUnit(ut);
            if (ud->getId() == "substance") {
                if ( unit->isMole() ) {
                    double exponent = unit->getExponent();
                    double multiplier = unit->getMultiplier();
                    int scale = unit->getScale();
                    double offset = unit->getOffset();
                    lvalue *= pow( multiplier * pow(10.0,scale), exponent ) + offset;
                    return lvalue;
                }
            }
        }
    }
    return lvalue;
}//unitforRates
void SbmlReader::addSubPrd(Reaction * reac,Id reaction_,string type) {
    map< string,double > rctMap;
    map< string,double >::iterator rctMap_iter;
    double rctcount=0.0;
    Shell * shell = reinterpret_cast< Shell* >( Id().eref().data() );
    rctMap.clear();
    unsigned int nosubprd;
    const SpeciesReference* rct;
    if (type == "sub") {
        nosubprd = reac->getNumReactants();
    } else
        nosubprd = reac->getNumProducts();
    for ( unsigned int rt=0; rt<nosubprd; rt++ ) {
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
        if (type =="sub")
            noOfsub_ +=rctcount;
        for ( int i=0; (int)i<rct->getStoichiometry(); i++ )
            shell->doAddMsg( "OneToOne", reaction_, type ,molSidMIdMap_[sp] , "reac" );
    }
}
/* Transform units from SBML to MOOSE
   MOOSE units for
   volume -- cubic meter
*/

double SbmlReader::transformUnits( double mvalue,UnitDefinition * ud,string type, bool hasonlySubUnit ) {
    double lvalue = mvalue;
    if (type == "compartment") {
        for ( unsigned int ut = 0; ut < ud->getNumUnits(); ut++ ) {
            Unit * unit = ud->getUnit(ut);
            double exponent = unit->getExponent();
            double multiplier = unit->getMultiplier();
            int scale = unit->getScale();
            double offset = unit->getOffset();
            lvalue *= pow( multiplier * pow(10.0,scale), exponent ) + offset;
            // Need to check if spatial dimension is less than 3 then,
            // then volume conversion e-3 to convert cubicmeter shd not be done.
            if ( unit->isLitre() ) {
                lvalue *= pow(1e-3,exponent);
                return lvalue;
            }
        }
    } else if(type == "substance") {
        for ( unsigned int ut = 0; ut < ud->getNumUnits(); ut++ ) {
            Unit * unit = ud->getUnit(ut);
            //cout << " :) " << UnitKind_toString(unit->getKind());
            if ( unit->isMole() ) {
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
                    // if hasonlySubstanceUnit is false,
                    //Then convert mole to milli Molar for moose as concentration units are in milliMolar
                    // Molar = mole/size
                    lvalue *= pow(1e+3,exponent);
                return lvalue;
            } else if(unit->isItem())
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
void SbmlReader::printMembers( const ASTNode* p,vector <string> & ruleMembers ) {
    if ( p->getType() == AST_NAME ) {
        //cout << "_NAME" << " = " << p->getName() << endl;
        ruleMembers.push_back( p->getName() );
    }
    int num = p->getNumChildren();
    for( int i = 0; i < num; ++i ) {
        const ASTNode* child = p->getChild(i);
        printMembers( child,ruleMembers );
    }
}

void SbmlReader ::findModelParent( Id cwe, const string& path, Id& parentId, string& modelName ) {
    //Taken from LoadModels.cpp
    //If path exist example /model when come from GUI it creates model under /model/filename
    // i.e. because by default we creat genesis,sbml models under '/model', which is created before and path exist
    // at the time it comes to SbmlReader.cpp
    //When run directly (command line readSBML() )it ignores the path and creates under '/' and filename takes as "SBMLtoMoose"
    //modelName = "test";
    cout << "here " << path;
    string fullPath = path;
    if ( path.length() == 0 )
        parentId = cwe;

    if ( path == "/" )
        parentId = Id();

    if ( path[0] != '/' ) {
        string temp = cwe.path();
        if ( temp[temp.length() - 1] == '/' )
            fullPath = temp + path;
        else
            fullPath = temp + "/" + path;
    }
    Id paId( fullPath );
    if ( paId == Id() ) { // Path includes new model name
        string::size_type pos = fullPath.find_last_of( "/" );
        assert( pos != string::npos );
        string head = fullPath.substr( 0, pos );
        Id ret( head );
        // When head = "" it means paId should be root.
        if ( ret == Id() && head != "" && head != "/root" )
            ;//return 0;
        parentId = ret;
        modelName = fullPath.substr( pos + 1 );
    }

    else { // Path is an existing element.
        parentId = paId;

    }
}

/**
 * @brief Populate parmValueMap; a member variable with keeps all the gloabals
 * parameters of SBML model.
 */
void SbmlReader::getGlobalParameter() {
    for ( unsigned int pm = 0; pm < model_->getNumParameters(); pm++ ) {
        Parameter* prm = model_->getParameter( pm );
        std::string id,unit;
        if ( prm->isSetId() ) {
            id = prm->getId();
        }
        double value = 0.0;
        if ( prm->isSetValue() ) {
            value=prm->getValue();
        }
        parmValueMap[id] = value;
    }
}
#endif // USE_SBML

string SbmlReader::nameString( string str ) {
    string str1;

    int len = str.length();
    int i= 0;
    do {
        switch( str.at(i) ) {
        case ' ':
            str1 = "_space_";
            str.replace( i,1,str1 );
            len += str1.length()-1;
            break;
        }
        i++;
    } while ( i < len );
    return str;
}

