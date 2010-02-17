#include <vector>
#include <string>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <libxml/xmlreader.h>
#include <iostream>
#include <sstream>
#include <cassert>
#include "NCell.h"
#include "IonPool.h"

using namespace std;
map< std::string,PoolDefinition >IonPool::lookup_;
void IonPool::setName( const std::string& value )
{
	name = value;
	pooldefinition_ = lookupDefinition( value );
}
void IonPool::unsetName( )
{
	name.erase();
}
bool IonPool::isSetName () const
{
   return (name.empty() == false);
}
const std::string& IonPool::getName() const
{
	return name;
}
void IonPool::setScalingFactor( double value )
{
	B = value;
}
const double IonPool::getScalingFactor() const
{
	return B;
}
void IonPool::setGroups( string value )
{
	groups.push_back( value );
}
vector < string > IonPool::getGroups() const
{
	return groups;
}
void IonPool::unsetGroups()
{
	groups.clear();
}
void IonPool::setScaling( const std::string& value )
{
	scaling = value;
	
}
const std::string& IonPool::getScaling() const
{
	return scaling;
}
const std::string& IonPool::getIon() const
{
	return pooldefinition_->getIon();
}
void IonPool::setIon(const std::string& ion)
{
	pooldefinition_->setIon( ion );
}
double IonPool::getCharge()
{
	return pooldefinition_->getCharge();
}
void IonPool::setCharge(double value )
{
	pooldefinition_->setCharge( value );
}
const std::string& IonPool::getPoolName() const
{
	return pooldefinition_->getPoolName();
}
void IonPool::setPoolName(const std::string& name)
{
	pooldefinition_->setPoolName( name );
}
const std::string& IonPool::getStatus() const
{
	return pooldefinition_->getStatus();
}
void IonPool::setStatus(const std::string& status)
{
	pooldefinition_->setStatus( status );
}
double IonPool::getResting_conc()
{
	return pooldefinition_->getResting_conc();
}
void IonPool::setResting_conc(double value )
{
	pooldefinition_->setResting_conc( value );
}
double IonPool::getDecay_constant()
{
	return pooldefinition_->getDecay_constant();
}
void IonPool::setDecay_constant(double value )
{
	pooldefinition_->setDecay_constant( value );
}
double IonPool::getShell_thickness()
{
	return pooldefinition_->getShell_thickness();
}
void IonPool::setShell_thickness(double value )
{
	pooldefinition_->setShell_thickness( value );
}
vector< string > IonPool::getSegGroups() const
{
	return segGroups;
}
void IonPool::setSegGroups( string seg )
{
	segGroups.push_back( seg );
}
void IonPool::unsetSegGroups()
{
	segGroups.clear();
}
const std::string& PoolDefinition::getIon() const
{
	return ion;
}
void PoolDefinition::setIon(const std::string& name)
{
	ion = name;
}
double PoolDefinition::getCharge()
{
	return charge;
}
void PoolDefinition::setCharge(double value )
{
	charge = value;
}
const std::string& PoolDefinition::getPoolName() const
{
	return pool_name;
}
void PoolDefinition::setPoolName(const std::string& name)
{
	pool_name = name;
}
const std::string& PoolDefinition::getStatus() const
{
	return status;
}
void PoolDefinition::setStatus(const std::string& value)
{
	status = value;
}
double PoolDefinition::getResting_conc()
{
	return resting_conc;
}
void PoolDefinition::setResting_conc(double value )
{
	resting_conc = value;
}
double PoolDefinition::getDecay_constant()
{
	return decay_constant;
}
void PoolDefinition::setDecay_constant(double value )
{
	decay_constant = value;
}
double PoolDefinition::getShell_thickness()
{
	return shell_thickness;
}
void PoolDefinition::setShell_thickness(double value )
{
	shell_thickness = value;
}
PoolDefinition* IonPool::lookupDefinition( const std::string& name )
{	std::string filename = name + ".xml";
	static map< string,PoolDefinition >::iterator iter;
	iter = lookup_.find( name );
	if( iter == lookup_.end() ){
		lookup_[name].readDefinition( filename ); 
	}
	return &lookup_[name];
}
/* register namespaces in the IonPool */
int PoolDefinition::register_PoolNamespaces() 
{
    
     int ret,r;
     if (rdrptr != NULL) {
     	ret = xmlTextReaderRead(rdrptr);
	if (ret == 1) {
	   int tot = xmlTextReaderAttributeCount(rdrptr);
	   for( int  i = 0; i < tot-1; i++ )
	   {
	 	if (xmlTextReaderMoveToAttributeNo(rdrptr,i)){
		   const xmlChar * readername = xmlTextReaderName(rdrptr);
		   const xmlChar * value = xmlTextReaderValue(rdrptr);
		   const xmlChar * prefix = xmlTextReaderPrefix(rdrptr);
		   xmlNodePtr root = xmlDocGetRootElement(docptr);
                   if (prefix == NULL){ 
			if ((!xmlStrcmp(root->name, (const xmlChar *) "channelml")))
				readername = (xmlChar * )" channelml" ;
			prefix = (xmlChar *)"";
		   }
		   int	namelen = xmlStrlen(readername);
		   int	pfxlen = xmlStrlen(prefix);
		   const xmlChar * name = xmlStrsub(readername,pfxlen+1,namelen); 
		   r = xmlXPathRegisterNs(cxtptr,name,value);
	           if ( r != 0 ){
		      cerr << "Error: unable to register NS with prefix= " 
				 << name << " and href= " << value;    
		      return(-1);
		   }
		   else setpoolnamespace = true;
	    	}
	    }
	    
	  }
	     
	}
	return(r);
}
/* returns xmlXPathObjectPtr for ionpool nodes */
xmlXPathObjectPtr PoolDefinition::getnodeset (xmlChar *xpath) 
{
       xmlXPathObjectPtr result;
        
        if (cxtptr == NULL) {
                cout << "Error in xmlXPathNewContext" <<endl;
                return NULL;
        }
	result = xmlXPathEvalExpression(xpath, cxtptr);
        if (result == NULL) {
                cout << "Error in xmlXPathEvalExpression" <<endl;
                return NULL;
        }
        /*if(xmlXPathNodeSetIsEmpty(result->nodesetval)){
                xmlXPathFreeObject(result);
                cout << "No result" <<endl;
                return NULL;
        }*/
        return result;
}

void PoolDefinition::readDefinition( std::string& filename )
{
        
        docptr = xmlParseFile(&filename[0]);
  	cxtptr = xmlXPathNewContext(docptr);
  	rdrptr = xmlReaderForFile(filename.c_str(), NULL, 0);
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
	xmlNodePtr cur;
	//xmlChar *density;
	int size;
	char *endp, *ion, *charge, *pool_name, *status;
	char * resting_conc, *decay_constant, *shell_thickness;
	int r = register_PoolNamespaces();
	if (r == 0){	
	   result = getnodeset((xmlChar *)"//channelml:ion");
   	   nodeset = result->nodesetval;
	   unsigned int num = (nodeset) ? nodeset->nodeNr : 0;
   	   for(unsigned int i = 0; i < num; ++i) {
     		assert(nodeset->nodeTab[i]);
     		if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
        	   cur = nodeset->nodeTab[i];  
        	    if(cur->ns) { 
          		ion = (char *) xmlGetProp(cur,(const xmlChar *) "name");
	  	        setIon( ion );
          		charge = (char *) xmlGetProp(cur,(const xmlChar *) "charge");
			double charge_d = strtod( charge,&endp );
	  		setCharge( charge_d );
	  	    }
		}
	   }
	   result = getnodeset((xmlChar *)"//channelml:ion_concentration");
           nodeset = result->nodesetval;
           size = (nodeset) ? nodeset->nodeNr : 0;
           for(int i = 0; i < size; i++) {
           	assert(nodeset->nodeTab[i]);
           	if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
              	   cur = nodeset->nodeTab[i];  
              	   if(cur->ns){ 
                	pool_name = (char *) xmlGetProp(cur,(const xmlChar *) "name");
			setPoolName( pool_name );
		   }
		   result = getnodeset((xmlChar *)"//channelml:status");
		   nodeset = result->nodesetval;
		   size = (nodeset) ? nodeset->nodeNr : 0;
		   for(int v = 0; v < size; v++) {
           	    assert(nodeset->nodeTab[v]);
           	    if(nodeset->nodeTab[v]->type == XML_ELEMENT_NODE) {
             	            cur = nodeset->nodeTab[v];  
		            if(cur->ns){ 
		        	status = (char *) xmlGetProp(cur,(const xmlChar *) "value");
				setStatus( status );
		     	    }
	             }
     		   }
		   result = getnodeset((xmlChar *)"//channelml:decaying_pool_model");
       		   nodeset = result->nodesetval;
       		   size = (nodeset) ? nodeset->nodeNr : 0;
       		   for(int i = 0; i < size; i++) {
           	       assert(nodeset->nodeTab[i]);
          	       if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
              		cur = nodeset->nodeTab[i];  
              		if(cur->ns){ 
                	  resting_conc = (char *) xmlGetProp(cur,(const xmlChar *) "resting_conc");
			  decay_constant = (char *) xmlGetProp(cur,(const xmlChar *) "decay_constant");
			}
			result = getnodeset((xmlChar *)"//channelml:pool_volume_info");
			nodeset = result->nodesetval;
       		   	size = (nodeset) ? nodeset->nodeNr : 0;
       		   	for(int i = 0; i < size; i++) {
           	            assert(nodeset->nodeTab[i]);
          	            if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
              		      cur = nodeset->nodeTab[i];  
              		      if(cur->ns){ 
		                shell_thickness = (char *) xmlGetProp(cur,(const xmlChar *) "shell_thickness");
			      }
			    }
			}
			double restConc = strtod( resting_conc, &endp );
			double decayconst = strtod( decay_constant, &endp );
			double thickness = strtod( shell_thickness, &endp );
			setResting_conc( restConc );
			setDecay_constant( decayconst );
			setShell_thickness( thickness );
	   	     }
           	  }
       	       }
	    }
	}
}


