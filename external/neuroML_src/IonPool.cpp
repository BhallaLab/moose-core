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
/**
 * Sets the value of the "name" attribute of this IonPool. 
 */
void IonPool::setName( const std::string& value )
{
	name = value;
	pooldefinition_ = lookupDefinition( value );
}
/**
 * Unsets the value of the "name" attribute of this IonPool. 
 */
void IonPool::unsetName( )
{
	name.erase();
}
bool IonPool::isSetName () const
{
   return (name.empty() == false);
}
/**
 * Returns the value of the "name" attribute of this IonPool. 
 * 
 */
const std::string& IonPool::getName() const
{
	return name;
}
/**
 * Sets the value of the "B" attribute  of this IonPool where 
 * B  is the scaling factor.
 */
void IonPool::setScalingFactor( double value )
{
	B = value;
}
/**
 * Returns the value of the "B" attribute of this IonPool where 
 * B  is the scaling factor.
 */
const double IonPool::getScalingFactor() const
{
	return B;
}
/**
 * Sets the value of the "groups" attribute of this IonPool. 
 */
void IonPool::setGroups( string value )
{
	groups.push_back( value );
}
/**
 * Returns the value of the "groups" attribute of this IonPool. 
 * 
 */
vector < string > IonPool::getGroups() const
{
	return groups;
}
/**
 * Unsets the value of the "groups" attribute of this IonPool. 
 */
void IonPool::unsetGroups()
{
	groups.clear();
}
/**
 * Sets the value of the "scaling" attribute of this IonPool. 
 */
void IonPool::setScaling( const std::string& value )
{
	scaling = value;
	
}
/**
 * Returns the value of the "scaling" attribute of this IonPool. 
 * 
 */
const std::string& IonPool::getScaling() const
{
	return scaling;
}
/**
 * Returns the value of the "ion" attribute of this IonPool. 
 * 
 */
const std::string& IonPool::getIon() const
{
	return pooldefinition_->getIon();
}
/**
 * Sets the value of the "ion" attribute of this IonPool. 
 */
void IonPool::setIon(const std::string& ion)
{
	pooldefinition_->setIon( ion );
}
/**
 * Returns the value of the "charge" attribute of this IonPool. 
 * 
 */
double IonPool::getCharge()
{
	return pooldefinition_->getCharge();
}
/**
 * Sets the value of the "charge" attribute of this IonPool. 
 */
void IonPool::setCharge(double value )
{
	pooldefinition_->setCharge( value );
}
/**
 * Returns the value of the "pool_name" attribute of this IonPool. 
 * 
 */
const std::string& IonPool::getPoolName() const
{
	return pooldefinition_->getPoolName();
}
/**
 * Sets the value of the "pool_name" attribute of this IonPool. 
 */
void IonPool::setPoolName(const std::string& name)
{
	pooldefinition_->setPoolName( name );
}
/**
 * Returns the value of the "status" attribute of this IonPool. 
 * 
 */
const std::string& IonPool::getStatus() const
{
	return pooldefinition_->getStatus();
}
/**
 * Sets the value of the "status" attribute of this IonPool. 
 */
void IonPool::setStatus(const std::string& status)
{
	pooldefinition_->setStatus( status );
}
/**
 * Returns the value of the "resting_conc" attribute of this IonPool. 
 * 
 */
double IonPool::getResting_conc()
{
	return pooldefinition_->getResting_conc();
}
/**
 * Sets the value of the "resting_conc" attribute of this IonPool. 
 */
void IonPool::setResting_conc(double value )
{
	pooldefinition_->setResting_conc( value );
}
/**
 * Returns the value of the "decay_constant" attribute of this IonPool. 
 * 
 */
double IonPool::getDecay_constant()
{
	return pooldefinition_->getDecay_constant();
}
/**
 * Sets the value of the "decay_constant" attribute of this IonPool. 
 */
void IonPool::setDecay_constant(double value )
{
	pooldefinition_->setDecay_constant( value );
}
/**
 * Returns the value of the "shell_thickness" attribute of this IonPool. 
 * 
 */
double IonPool::getShell_thickness()
{
	return pooldefinition_->getShell_thickness();
}
/**
 * Sets the value of the "shell_thickness" attribute of this IonPool. 
 */
void IonPool::setShell_thickness(double value )
{
	pooldefinition_->setShell_thickness( value );
}
/**
 * Returns the value of the "segGroups" attribute of this IonPool. 
 * 
 */
vector< string > IonPool::getSegGroups() const
{
	return segGroups;
}
/**
 * Sets the value of the "segGroups" attribute of this IonPool. 
 */
void IonPool::setSegGroups( string seg )
{
	segGroups.push_back( seg );
}
/**
 * Unsets the value of the "segGroups" attribute of this IonPool. 
 */
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
/**
 * Returns the PoolDefinition object corresponding to the IonPool.
 * For now it is assumed that filename is poolname.xml 
 */
PoolDefinition* IonPool::lookupDefinition( const std::string& name )
{	std::string filename = name + ".xml";
	static map< string,PoolDefinition >::iterator iter;
	iter = lookup_.find( name );
	if( iter == lookup_.end() ){
		lookup_[name].readDefinition( filename ); 
	}
	return &lookup_[name];
}
/**
 * Register the namespaces used in channelML file and returns 0 if success.
 */
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
/**
 * Returns xmlXPathObjectPtr to channelML node for the given path.
 * The xmlPathObjectPtr returned by the function contains a set of nodes and 
 * other information needed to iterate through the set and act on the results.
*/

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
/**
 *  Reads in the IonPool definition for the given channelML file. 
 */
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


