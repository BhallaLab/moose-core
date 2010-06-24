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
#include "Segment.h"
#include "NCable.h"
#include "Channel.h"
#include "IonPool.h"
#include "NCell.h"
using namespace std;
/**
 * Sets xmlDocPtr of the containing document.  
 */
void NCell::setXmldoc(xmlDocPtr& doc)
{
	xmlDoc = doc;
}
/**
 * Returns the xmlDocPtr of the containing document. 
 */
xmlDocPtr& NCell::getXmldoc()
{
	return xmlDoc;
}
/**
 * Sets the xmlXPathContextPtr.
 */
void NCell::setContext(xmlXPathContextPtr& cnxt)
{
	context = cnxt;
}
/**
 * Returns the xmlXPathContextPtr. 
 */
xmlXPathContextPtr& NCell::getContext()
{
	return context;
}
/**
 * Sets the xmlTextReaderPtr. 
 * 
 */
void NCell::setReaderptr(xmlTextReaderPtr& reader)
{
	readerPtr = reader;
}
/**
 * Returns the xmlTextReaderPtr.
 * 
 */
xmlTextReaderPtr& NCell::getReaderptr()
{
	return readerPtr;
}
/**
 * Returns the value of the "lengthunits" attribute of this NCell. 
 * LengthUnits which is specified in MorphML file.
 */
const std::string& NCell::getLengthUnits()
{
	return lengthunits;
}
/**
 * Sets the value of the "lengthunits" attribute of this NCell. 
 * LengthUnits which is specified in MorphML file.
 */
void NCell::setLengthUnits(const std::string& value )
{
	lengthunits = value;
}
/**
 * Returns the "BiophysicsUnits" specified in MorphML file.
 */
string NCell::getBiophysicsUnits()
{
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
	xmlNodePtr cur;
	char *unit;
  	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:setBiophysicsUnits() unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
}
	}
	result = getnodeset((xmlChar *)"//neuroml:biophysics");
  	nodeset = result->nodesetval;
  	unsigned int size = (nodeset) ? nodeset->nodeNr : 0;
 	for(unsigned int i = 0; i < size; ++i) {
     	   assert(nodeset->nodeTab[i]);
     	   if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
              cur = nodeset->nodeTab[i];  
              if(cur->ns) { 
                unit = (char *)xmlGetProp(cur, (const xmlChar *) "units"); 
	      }
	    }
	}	
	return unit;
}
/**
 * Returns xmlXPathObjectPtr to morphML node for the given path.
 * The xmlPathObjectPtr returned by the function contains a set of nodes and 
 * other information needed to iterate through the set and act on the results.
*/
xmlXPathObjectPtr NCell::getnodeset (xmlChar *xpath)
{
        xmlXPathObjectPtr result;
        
        if (context == NULL) {
                cout << "Error in xmlXPathNewContext" <<endl;
                return NULL;
        }
	result = xmlXPathEvalExpression(xpath, context);
       // xmlXPathFreeContext(context);
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
 * Predicate returning true or false depending on whether this NCell's  
 * "setnamespaces" attribute has been set. 
 */
bool NCell::isSetNamespaces()
{
	return setnamespaces;
}
/* bool NCell::hasSynapse()
{
	return synapticChl;
} */

/**
 *  Register the namespaces used in morphML file and returns 0 if success. 
 */
int NCell::register_namespaces()
{
     int ret,r;
     if (readerPtr != NULL) {
     	ret = xmlTextReaderRead(readerPtr);
	if (ret == 1) {
	   int tot = xmlTextReaderAttributeCount(readerPtr);
	   for( int  i = 0; i < tot-1; i++ )
	   {
	 	if (xmlTextReaderMoveToAttributeNo(readerPtr,i)){
		   const xmlChar * readername = xmlTextReaderName(readerPtr);
		   const xmlChar * value = xmlTextReaderValue(readerPtr);
		   const xmlChar * prefix = xmlTextReaderPrefix(readerPtr);
		   xmlNodePtr root = xmlDocGetRootElement(xmlDoc);
                   if (prefix == NULL){ 
			if ((!xmlStrcmp(root->name, (const xmlChar *) "neuroml")))
				readername = (xmlChar * )" neuroml" ;
			prefix = (xmlChar *)"";
			}
		   int	namelen = xmlStrlen(readername);
		   int	pfxlen = xmlStrlen(prefix);
		   const xmlChar * name = xmlStrsub(readername,pfxlen+1,namelen); 
		   r = xmlXPathRegisterNs(context,name,value);
	           if ( r != 0 ){
		      cerr << "Error: unable to register NS with prefix= " 
				 << name << " and href= " << value;    
		      return(-1);
		   }
		   else setnamespaces = true;
	       }
	   }
	   if (xmlTextReaderMoveToAttributeNo(readerPtr,tot-1)){
		   const xmlChar * readername = xmlTextReaderName(readerPtr);
		   if  ((!xmlStrcmp(readername,(const xmlChar *) "length_units")) ||(!xmlStrcmp(readername,(const xmlChar *)"lengthUnits")))
		   {
			  char* lengthunit = (char *)xmlTextReaderValue(readerPtr);
			  cout << "length units " << lengthunit <<endl;
			  setLengthUnits( lengthunit );
		   }
	   }
	}
     }
     return(r);
}
/**
 * Returns the number of segments in the NCell.
 */
unsigned int NCell::getNumSegments() 
{
	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:getNumSegments() unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
	}
	xmlXPathObjectPtr result = getnodeset((xmlChar *)"//mml:segment");
  	xmlNodeSetPtr nodeset = result->nodesetval;
  	unsigned int numSegments = (nodeset) ? nodeset->nodeNr : 0;
 	return numSegments;
	
}
/**
 * Returns the number of cables in the NCell.
 * 
 */
unsigned int NCell::getNumCables() 
{
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
  	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:getNumCables() unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
	}
	result = getnodeset((xmlChar *)"//mml:cable");
  	nodeset = result->nodesetval;
  	unsigned int numCables = (nodeset) ? nodeset->nodeNr : 0;
 	return numCables;
	
}
/**
 * Returns the number of channels in the NCell.
 * 
 */
/* function to get the number of channels */
unsigned int NCell::getNumChannels() 
{
	unsigned int numChannels = 0;	
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
	xmlNodePtr cur,parmNode,grpNode;
	char *endp;
	ostringstream xpath;
  	xmlChar *pname;
  	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:getNumChannels() unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
	}
	result = getnodeset((xmlChar *)"//bio:mechanism");
  	nodeset = result->nodesetval;
  	unsigned int num = (nodeset) ? nodeset->nodeNr : 0;
	for(unsigned int i = 0; i < num; ++i) {
     		assert(nodeset->nodeTab[i]);
     		if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       		  cur = nodeset->nodeTab[i];  
       		  if(cur->ns) { 
		    char *name, *value, *passivecond, *group, *seg;
	  	    double gmax;
	  	    bool pc;
          	    xmlChar * type;
	  	    type = (xmlChar *)xmlGetProp(cur, (const xmlChar *) "type"); 
	            if ((!xmlStrcmp(type, (xmlChar *) "Channel Mechanism"))){
	      		numChannels++;
			name = (char *)xmlGetProp(cur, (const xmlChar *) "name"); 
	                chl_.setName( name );
	                passivecond = (char *)xmlGetProp(cur, (const xmlChar *) "passive_conductance");
		        if (passivecond)
			    pc = true;
		        else pc = false;
			    chl_.setPassivecond(pc);
			parmNode = cur->xmlChildrenNode;
			while (parmNode != NULL){
            		     if((!xmlStrcmp(parmNode->name, (const xmlChar *) "parameter"))) {
		               pname = (xmlChar *)xmlGetProp(parmNode,(const xmlChar *) "name");
			       value = (char *)xmlGetProp(parmNode, (const xmlChar *) "value");  
			       if ((!xmlStrcmp(pname, (xmlChar *) "gmax"))){
				gmax = strtod(value, &endp);
				chl_.setGmax( gmax ); 
			       }	
			       xmlFree(pname);
			       xmlFree(value);
			       grpNode = parmNode->xmlChildrenNode;	
			       chl_.unsetGroups();
			       chl_.unsetSegGroups();
			       while (grpNode != NULL) {
            		     	if((!xmlStrcmp(grpNode->name, (const xmlChar *) "group"))) {
		               	  group = (char *)xmlNodeGetContent(grpNode);
				  if ( group != '\0' ){
                       	       	     std::string group_str((char *) group);
			       	     chl_.setGroups( group_str );
			       	     xmlFree(group);
				  }
			     	}
				else if((!xmlStrcmp(grpNode->name, (const xmlChar *) "seg"))) {
				   seg = (char *)xmlNodeGetContent(grpNode);
				   if ( seg != '\0' ){
                       	       	      std::string group_str((char *) seg);
			       	      chl_.setSegGroups( group_str );
			       	      xmlFree(seg);
				   }
				}
			     	grpNode = grpNode->next;
			       }
			      }
			      parmNode = parmNode->next;			
			}
			vec_channel.push_back( chl_ );	
		    }
	      	}
	    }
	
	}
 	return numChannels;
}
/**
 * Returns the number of calciumpools in the NCell. 
 * 
 */
unsigned int NCell::getNumPools() 
{
	unsigned int numPool = 0;	
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
	xmlNodePtr cur,parmNode,grpNode;
	char *name, *value, *endp, *group, *seg;
	double sf;
	xmlChar  *pname, *scaling;
	ostringstream xpath;
  	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:getNumPools() unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
	}
	result = getnodeset((xmlChar *)"//bio:mechanism");
  	nodeset = result->nodesetval;
  	unsigned int num = (nodeset) ? nodeset->nodeNr : 0;
	for(unsigned int i = 0; i < num; ++i) {
     		assert(nodeset->nodeTab[i]);
     		if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       		  cur = nodeset->nodeTab[i];  
       		  if(cur->ns) { 
  		    xmlChar * type;
  		    type = (xmlChar *)xmlGetProp(cur, (const xmlChar *) "type"); 
		    if ((!xmlStrcmp(type, (xmlChar *) "Ion Concentration"))){
			numPool++;
		     	name = (char *)xmlGetProp(cur, (const xmlChar *) "name"); 
	      		ion_.setName( name );
			parmNode = cur->xmlChildrenNode;
			while (parmNode != NULL){
            		     if((!xmlStrcmp(parmNode->name, (const xmlChar *) "parameter"))) {
		               pname = (xmlChar *)xmlGetProp(parmNode,(const xmlChar *) "name");
			       value = (char *)xmlGetProp(parmNode, (const xmlChar *) "value");  
			       /*scaling = (xmlChar *)xmlGetProp(parmNode, (const xmlChar *) "scaling"); 
			       std::string scale((char*)scaling);
			       ion_.setScaling( scale ); */
			       if ((!xmlStrcmp(pname, (xmlChar *) "specific_current_scaling_factor"))){
				 ion_.setScaling("specific_current_scaling_factor");
				 sf = strtod(value, &endp);
				 ion_.setScalingFactor( sf ); 
		    	       }
			       if ((!xmlStrcmp(pname, (xmlChar *) "fixed_current_scaling_factor"))){
				 ion_.setScaling("fixed_current_scaling_factor");
				 sf = strtod(value, &endp);
				 ion_.setScalingFactor( sf ); 
		    	       }
			       xmlFree(pname);
			       xmlFree(value);
			     //  xmlFree(scaling);
			       grpNode = parmNode->xmlChildrenNode;	
			       ion_.unsetGroups();
			       ion_.unsetSegGroups();
			       while (grpNode != NULL) {
            		     	if((!xmlStrcmp(grpNode->name, (const xmlChar *) "group"))) {
		               	  group = (char *)xmlNodeGetContent(grpNode);
				  if ( group != '\0' ){
                       	       		std::string group_str((char *) group);
			       		ion_.setGroups( group_str );
			       		xmlFree(group);
				  }
			     	}
				else if((!xmlStrcmp(grpNode->name, (const xmlChar *) "seg"))) {
				   seg = (char *)xmlNodeGetContent(grpNode);
				   if ( seg != '\0' ){
                       	       	      std::string group_str((char *) seg);
			       	      ion_.setSegGroups( group_str );
			       	      xmlFree(seg);
				   }
				}				
			     	grpNode = grpNode->next;
			       }
			      }
			      parmNode = parmNode->next;			
			}
	      		vec_pool.push_back(ion_);   
	         }
	       }	 
	    }
	    
	}
	return numPool;
}
/**
 * Returns the number of synaptic channels in the NCell.
 * 
 */
unsigned int NCell::getNumSynChannels() 
{
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
  	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:getNumSynChannels():unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
	}
	result = getnodeset((xmlChar *)"//net:potential_syn_loc");
  	nodeset = result->nodesetval;
  	unsigned int numSynChannels = (nodeset) ? nodeset->nodeNr : 0;
	return numSynChannels;
	
}
/**
 * Returns the init membrane potential of the NCell.
 * 
 */
double NCell::getInit_memb_potential() 
{
	double initVm;	
  	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
  	xmlNodePtr cur;
  	
  	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:from getInit_memb_potential() unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
	}
	result = getnodeset((xmlChar *)"//bio:init_memb_potential/bio:parameter");
  	nodeset = result->nodesetval;
  	int size= (nodeset) ? nodeset->nodeNr : 0;
 	for(int i = 0; i < size; ++i) {
     	    assert(nodeset->nodeTab[i]);
     	    if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
               cur = nodeset->nodeTab[i];  
               if(cur->ns) { 
                 char *value, *endp;
    	         value = (char *)xmlGetProp(cur, (const xmlChar *) "value"); 
	         initVm = strtod(value, &endp);
	       }
	     }
	 }
	 xmlXPathFreeObject(result); 
	 return initVm;
	
}
/**
 * Returns the specific capacitance of the NCell.
 * 
 */
double NCell::getSpec_capacitance() 
{
	double CM;	
  	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
  	xmlNodePtr cur;
  	
  	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:from getSpec_capacitance():unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
	}
	result = getnodeset((xmlChar *)"//bio:spec_capacitance/bio:parameter");
  	nodeset = result->nodesetval;
  	int size= (nodeset) ? nodeset->nodeNr : 0;
 	for(int i = 0; i < size; ++i) {
     	   assert(nodeset->nodeTab[i]);
     	   if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
              cur = nodeset->nodeTab[i];  
              if(cur->ns) { 
                char *value, *endp;
    	        value = (char *)xmlGetProp(cur, (const xmlChar *) "value"); 
	        CM = strtod(value, &endp);
	       }
	    }
	}
	xmlXPathFreeObject(result); 
	return CM;
	
}
/**
 * Returns the specific axial resistance of the NCell.
 * 
 */
double NCell::getSpec_axial_resistance() 
{
	double RA;
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
  	xmlNodePtr cur;
  	
  	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:from getSpec_axial_resistance(): unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
	}
	result = getnodeset((xmlChar *)"//bio:spec_axial_resistance/bio:parameter");
  	nodeset = result->nodesetval;
  	int size= (nodeset) ? nodeset->nodeNr : 0;
 	for(int i = 0; i < size; ++i) {
     	       assert(nodeset->nodeTab[i]);
     	       if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
                 cur = nodeset->nodeTab[i];  
                 if(cur->ns) { 
                   char *value, *endp;
    	           value = (char *)xmlGetProp(cur, (const xmlChar *) "value"); 
	           RA = strtod(value, &endp);
	         }
	       }
	   } 
	   xmlXPathFreeObject(result); 
	   return RA;
	
}
/**
 * Returns the object to nth segment. 
 * 
 */
const Segment* NCell::getSegment(int n)
{
  xmlXPathContextPtr context;
  context = xmlXPathNewContext(xmlDoc);
  xmlXPathObjectPtr result;
  xmlNodeSetPtr nodeset;
  xmlNodePtr cur;
  if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:from getSegment(): unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
   }
  ostringstream xpath;
  xpath<<"//mml:segment["<<n<<"]";
  result = getnodeset((xmlChar *)(xpath.str().c_str()));
  nodeset = result->nodesetval;
  int size = (nodeset) ? nodeset->nodeNr : 0;
  for(int i = 0; i < size; ++i) {
     assert(nodeset->nodeTab[i]);
     if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       cur = nodeset->nodeTab[i];  
       if(cur->ns) { 
         char *id, *name, *parent, *cable;
	  char *x, *y, *z, *diameter, *endp;
    	  id = ( char * )( xmlGetProp(cur, (const xmlChar *) "id") ); 
	  name = ( char * )( xmlGetProp(cur, (const xmlChar *) "name") ); 
	  parent = ( char * )( xmlGetProp(cur, (const xmlChar *) "parent") ); 
	  cable = ( char * )( xmlGetProp(cur, (const xmlChar *) "cable") ); 
	  std::string id_str = (char *) id;
	  seg_.setId( id_str );
	  std::string name_str((char*)name);
	  seg_.setName( name_str );
	  if( parent ){
	   std::string parent_str((char*)parent);
	   seg_.setParent( parent_str );
	  }
	  std::string cable_str((char*)cable);
	  seg_.setCable( cable_str );
	  xpath.str( "" ); 
	  xpath<<"//mml:segment["<<n<<"]/mml:proximal";	   
	  result = getnodeset((xmlChar *)xpath.str().c_str());
	  nodeset = result->nodesetval;
	  int size1 = (nodeset) ? nodeset->nodeNr : 0;
	  if ( size1 == 0 )
		seg_.setproximal = false;
	  else
		seg_.setproximal = true;
	  for(int i = 0; i < size1; ++i) {
	     assert(nodeset->nodeTab[i]);
	     if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
	        cur = nodeset->nodeTab[i];  
	        if(cur->ns) { 
	    	   x = (char *) xmlGetProp(cur,(const xmlChar *) "x");
		   double x0 = strtod(x, &endp);
		   seg_.proximal.setX( x0 );
		   y = (char *) xmlGetProp(cur,(const xmlChar *) "y");
		   double y0 = strtod(y, &endp);
		   seg_.proximal.setY( y0 );
		   z = (char *) xmlGetProp(cur,(const xmlChar *) "z");
		   double z0 = strtod(z, &endp);
	    	   seg_.proximal.setZ( z0 );
		   diameter = (char *) xmlGetProp(cur,(const xmlChar *) "diameter");
		   double dia0 = strtod(diameter, &endp);
		   seg_.proximal.setDiameter( dia0 );
		}  	
	      }
	    }
	     xpath.str( "" ); 
	     xpath<<"//mml:segment["<<n<<"]/mml:distal";	 
            result = getnodeset((xmlChar *)xpath.str().c_str());
	    nodeset = result->nodesetval;
   	    int size2 = (nodeset) ? nodeset->nodeNr : 0;
   	    for(int i = 0; i < size2; ++i) {
	        assert(nodeset->nodeTab[i]);
		if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
           	   cur = nodeset->nodeTab[i];  
           	   if(cur->ns) { 
		     x = (char *) xmlGetProp(cur,(const xmlChar *) "x");
		     double x1 = strtod(x, &endp);
		     seg_.distal.setX( x1 );
		     y = (char *) xmlGetProp(cur,(const xmlChar *) "y");
		     double y1 = strtod(y, &endp);
		     seg_.distal.setY( y1 );
		     z = (char *) xmlGetProp(cur,(const xmlChar *) "z");
		     double z1 = strtod(z, &endp);
		     seg_.distal.setZ( z1 );
		     diameter = (char *) xmlGetProp(cur,(const xmlChar *) "diameter");
		     double dia1 = strtod(diameter, &endp);
		     seg_.distal.setDiameter( dia1 );
		  }  	
               }
           }
       }

   } 
  }
  xmlXPathFreeObject(result); 
  return &seg_;
}
/**
 * Returns the object to Segment with the given id. 
 * 
 */
const Segment* NCell::getSegment(string id)
{
  xmlXPathObjectPtr result;
  xmlNodeSetPtr nodeset;
  xmlNodePtr cur;
  if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:from getSegment(): unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
   }
  ostringstream xpath;
  xpath<<"//mml:segment[@id='"<<id<<"']";
  result = getnodeset((xmlChar *)xpath.str().c_str());
  nodeset = result->nodesetval;
  int size = (nodeset) ? nodeset->nodeNr : 0;
  for(int i = 0; i < size; ++i) {
     assert(nodeset->nodeTab[i]);
     if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       cur = nodeset->nodeTab[i];  
       if(cur->ns) { 
          char *id, *name, *parent, *cable, *endp, *x, *y, *z, *diameter;
    	  id = (char *)xmlGetProp(cur, (const xmlChar *) "id"); 
	  name = (char *)xmlGetProp(cur, (const xmlChar *) "name"); 
	  parent = (char *)xmlGetProp(cur, (const xmlChar *) "parent"); 
	  cable = (char *)xmlGetProp(cur, (const xmlChar *) "cable"); 
	  seg_.setId( id );
	  seg_.setName( name );
	  if( parent )
	   seg_.setParent( parent );
	  seg_.setCable( cable );
	  xmlFree(id);
	  xmlFree(name);
	  xmlFree(parent);
	  xmlFree(cable);
	  xpath.str( "" ); 
	  xpath<<"//mml:segment[@id='"<<id<<"']/mml:proximal";
	  result = getnodeset((xmlChar *)xpath.str().c_str());
          nodeset = result->nodesetval;
	  int size1 = (nodeset) ? nodeset->nodeNr : 0;
	  for(int i = 0; i < size1; ++i) {
	     assert(nodeset->nodeTab[i]);
	     if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
	        cur = nodeset->nodeTab[i];  
	        if(cur->ns) { 
	    	   x = (char *) xmlGetProp(cur,(const xmlChar *) "x");
		   double x0 = strtod(x, &endp);
		   seg_.proximal.setX( x0 );
		   y = (char *) xmlGetProp(cur,(const xmlChar *) "y");
		   double y0 = strtod(y, &endp);
		   seg_.proximal.setY( y0 );
		   z = (char *) xmlGetProp(cur,(const xmlChar *) "z");
		   double z0 = strtod(z, &endp);
	    	   seg_.proximal.setZ( z0 );
		   diameter = (char *) xmlGetProp(cur,(const xmlChar *) "diameter");
		   double dia0 = strtod(diameter, &endp);
		   seg_.proximal.setDiameter( dia0 );
		   xmlFree(x);
		   xmlFree(y);
		   xmlFree(z);
		   xmlFree(diameter); 
		}  	
	      }
	    }
	    xpath.str( "" ); 
	    xpath<<"//mml:segment[@id='"<<id<<"']/mml:distal";
            result = getnodeset((xmlChar *)xpath.str().c_str());
   	    nodeset = result->nodesetval;
   	    int size2 = (nodeset) ? nodeset->nodeNr : 0;
   	    for(int i = 0; i < size2; ++i) {
	        assert(nodeset->nodeTab[i]);
		if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
           	   cur = nodeset->nodeTab[i];  
           	   if(cur->ns) { 
		     x = (char *) xmlGetProp(cur,(const xmlChar *) "x");
		     double x1 = strtod(x, &endp);
		     seg_.distal.setX( x1 );
		     y = (char *) xmlGetProp(cur,(const xmlChar *) "y");
		     double y1 = strtod(y, &endp);
		     seg_.distal.setY( y1 );
		     z = (char *) xmlGetProp(cur,(const xmlChar *) "z");
		     double z1 = strtod(z, &endp);
		     seg_.distal.setZ( z1 );
		     diameter = (char *) xmlGetProp(cur,(const xmlChar *) "diameter");
		     double dia1 = strtod(diameter, &endp);
		     seg_.distal.setDiameter( dia1 );
		     xmlFree(x);
		     xmlFree(y);
		     xmlFree(z);
		     xmlFree(diameter); 
	          }  	
               }
           }
       }

   } 
  }
  xmlXPathFreeObject(result); 
  return &seg_;
}
/**
 * Returns the object to nth NCable. 
 * 
 */
const NCable* NCell::getCable(int n)
{
	xmlNodeSetPtr nodeset;
        xmlXPathObjectPtr result;
	char *group,*name, *id;
	xmlNodePtr cur;
	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error:from getCable(): unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
        }
	ostringstream xpath;
  	xpath<<"//mml:cable["<<n<<"]";
  	result = getnodeset((xmlChar *)xpath.str().c_str());
	nodeset = result->nodesetval;
        int size = (nodeset) ? nodeset->nodeNr : 0;
        for(int i = 0; i < size; ++i) {
           assert(nodeset->nodeTab[i]);
           if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
             cur = nodeset->nodeTab[i];  
             if(cur->ns) {
	       id = (char *)xmlGetProp(cur, (const xmlChar *) "id");
	       std::string id_str((char*)id);	
	       cab_.setId( id_str ); 
               name = (char *)xmlGetProp(cur, (const xmlChar *) "name"); 
	       std::string name_str = (char *) name;
	       cab_.setName( name_str );
	       
	       xmlFree(id);
	       xmlFree(name);	 
	       xpath.str( "" ); 
	       xpath<<"//mml:cable["<<n<<"]/meta:group";	
	       result = getnodeset ((xmlChar*)xpath.str().c_str());
	       if(result) {
                  nodeset = result->nodesetval;
		  int gsize;
		  gsize = (nodeset) ? nodeset->nodeNr : 0;
		  cab_.unsetGroups();
	   	  for(int i = 0; i < gsize; ++i) {
     		    assert(nodeset->nodeTab[i]);
     		    if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       		       cur = nodeset->nodeTab[i];  
       		       if(cur->ns) {
		         group = (char *)xmlNodeGetContent(cur);
                       	 std::string group_str = (char *) group;
			 cab_.setGroups( group_str );
			 xmlFree(group);
		       }
		    }
                  }
                  xmlXPathFreeObject (result);
              }	
	    }
	}
     }
     return &cab_;

}
/**
 * Returns the object to nth Channel. 
 * 
 */
Channel* NCell::getChannel(int n)
{
	//chl_ = vec_channel[n];
	return &vec_channel[n];
}
/**
 * Returns the object to nth IonPool. 
 * 
 */
IonPool* NCell::getPool(int n)
{
	return &vec_pool[n];
}

/*Channel* NCell::getChannel(int n)
{
  //xmlNodePtr root = xmlDocGetRootElement(xmlDoc);
  NCell c;	
  xmlXPathObjectPtr result,result1,result2;
  xmlNodeSetPtr nodeset;
  xmlNodePtr cur;
  int chlsize,psize,grpsize;
  char *endp;
  xmlChar * type, *pname;
  if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error: from getChannel(): unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
   }
   ostringstream xpath;
   xpath<<"//bio:mechanism["<<n<<"]";
   result = getnodeset((xmlChar *)(xpath.str().c_str()));
   nodeset = result->nodesetval;
   chlsize = (nodeset) ? nodeset->nodeNr : 0;
   for(int i = 0; i < chlsize; ++i) {
     assert(nodeset->nodeTab[i]);
     if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       cur = nodeset->nodeTab[i];  
       if(cur->ns) { 
          char *name, *value, *group, *passivecond;
	  double gmax;
	  bool pc;
	  type = (xmlChar *)xmlGetProp(cur, (const xmlChar *) "type"); 
	  if ((!xmlStrcmp(type, (xmlChar *) "Channel Mechanism"))){
	      //std::string chltype((char*)type);
	      //chl_.setType( chltype );
	      name = (char *)xmlGetProp(cur, (const xmlChar *) "name"); 
	      chl_.setName( name );
	      passivecond = (char *)xmlGetProp(cur, (const xmlChar *) "passive_conductance");
	      if (passivecond)
		pc = true;
	      else pc = false;
		chl_.setPassivecond(pc);
	      xpath.str( "" ); 
	      xpath<<"//bio:mechanism["<<n<<"]/bio:parameter";	   
	      result1 = getnodeset((xmlChar *)xpath.str().c_str());
	      nodeset = result1->nodesetval;
  	      psize = (nodeset) ? nodeset->nodeNr : 0;
	      for(int i = 0; i < psize; ++i) {
     		assert(nodeset->nodeTab[i]);
     		if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       		   cur = nodeset->nodeTab[i];  
       		   if(cur->ns) { 
	             pname = (xmlChar *)xmlGetProp(cur, (const xmlChar *) "name"); 
		    // std::string parmName((char*)pname);
		    // chl_.setParameterName( parmName );
		     value  = (char *)xmlGetProp(cur, (const xmlChar *) "value"); 
		   }
	            if ((!xmlStrcmp(pname, (xmlChar *) "gmax"))){
	                gmax = strtod(value, &endp);
			 chl_.setGmax( gmax ); 
		    }	
	          }
		}
		xpath.str( "" ); 
	        xpath<<"//bio:mechanism["<<n<<"]/bio:parameter/bio:group";	   
	        result2 = getnodeset((xmlChar *)xpath.str().c_str());
	        nodeset = result2->nodesetval;
  	      	grpsize = (nodeset) ? nodeset->nodeNr : 0;
		chl_.unsetGroups();
	   	for(int i = 0; i < grpsize; ++i) {
     		   assert(nodeset->nodeTab[i]);
     		   if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       		     cur = nodeset->nodeTab[i];  
       		     if(cur->ns) {
		        group = (char *)xmlNodeGetContent(cur);
                       	std::string group_str = (char *) group;
			chl_.setGroups( group_str );
			xmlFree(group);
		     }
		   }
		}
	     }
	}	 

      }
   }
 return &chl_;
}*/
/**
 * Returns the object to nth SynChannel. 
 * 
 */
SynChannel* NCell::getSynChannel(int n)
{
	NCell c;	
  	xmlXPathObjectPtr result,result1;
  	xmlNodeSetPtr nodeset;
  	xmlNodePtr cur;
  	int synsize,grpsize;
  	char *group;
  	//xmlChar * type;
  	if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error: from getSynChannel(): unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
   	}
   	ostringstream xpath;
   	xpath<<"//net:potential_syn_loc["<<n<<"]";
   	result = getnodeset((xmlChar *)(xpath.str().c_str()));
   	nodeset = result->nodesetval;
   	synsize = (nodeset) ? nodeset->nodeNr : 0;
	for(int i = 0; i < synsize; ++i) {
	     assert(nodeset->nodeTab[i]);
	     if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
	       cur = nodeset->nodeTab[i];  
	       if(cur->ns) { 
		  char *type;
		  type = (char *)xmlGetProp(cur, (const xmlChar *) "synapse_type"); 
		  //cout <<"syn type" << type << endl;
		  synchl_.setSynType( type );
	       }
	     }
	     xpath.str( "" ); 
	        xpath<<"//net:potential_syn_loc["<<n<<"]/net:group";	   
	        result1 = getnodeset((xmlChar *)xpath.str().c_str());
	        nodeset = result1->nodesetval;
  	      	grpsize = (nodeset) ? nodeset->nodeNr : 0;
		synchl_.unsetGroups();
	   	for(int i = 0; i < grpsize; ++i) {
     		   assert(nodeset->nodeTab[i]);
     		   if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       		     cur = nodeset->nodeTab[i];  
       		     if(cur->ns) {
		        group = (char *)xmlNodeGetContent(cur);
			std::string group_str = (char *) group;
			synchl_.setGroups( group_str );
			xmlFree(group);
		     }
		   }
		}

	}
	return &synchl_;
}


/* find nth Pool */
/* IonPool* NCell::getPool(int n)
{
  //xmlNodePtr root = xmlDocGetRootElement(xmlDoc);
  NCell c;	
  xmlXPathObjectPtr result,result1,result2;
  xmlNodeSetPtr nodeset;
  xmlNodePtr cur;
  int size,psize,grpsize;
  char *endp;
  xmlChar * type, *pname, *scaling;
  if (! isSetNamespaces() ){	
	   if (register_namespaces() < 0){	
    	    	cerr << "Error: from getPool(): unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	   }
   }
   ostringstream xpath;
   xpath<<"//bio:mechanism["<<n<<"]";
   result = getnodeset((xmlChar *)(xpath.str().c_str()));
   nodeset = result->nodesetval;
   size = (nodeset) ? nodeset->nodeNr : 0;
   for(int i = 0; i < size; ++i) {
     assert(nodeset->nodeTab[i]);
     if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       cur = nodeset->nodeTab[i];  
       if(cur->ns) { 
          char *name, *value, *group, *passivecond;
	  double B;
	  bool pc;
	  type = (xmlChar *)xmlGetProp(cur, (const xmlChar *) "type"); 
	  if ((!xmlStrcmp(type, (xmlChar *) "Ion Pool"))){
	     // std::string iontype((char*)type);
	    //  ion_.setType( iontype );
	      name = (char *)xmlGetProp(cur, (const xmlChar *) "name"); 
	      ion_.setName( name );
	      xpath.str( "" ); 
	      xpath<<"//bio:mechanism["<<n<<"]/bio:parameter";	   
	      result1 = getnodeset((xmlChar *)xpath.str().c_str());
	      nodeset = result1->nodesetval;
  	      psize = (nodeset) ? nodeset->nodeNr : 0;
	      for(int i = 0; i < psize; ++i) {
     		assert(nodeset->nodeTab[i]);
     		if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       		   cur = nodeset->nodeTab[i];  
       		   if(cur->ns) { 
	             pname = (xmlChar *)xmlGetProp(cur, (const xmlChar *) "name"); 
		     //std::string parmName((char*)pname);
		    // ion_.setParameterName( parmName );
		     value  = (char *)xmlGetProp(cur, (const xmlChar *) "value"); 
		     scaling = (xmlChar *)xmlGetProp(cur, (const xmlChar *) "scaling"); 
		     std::string scale((char*)scaling);
		     ion_.setScaling( scale );
		   }
	            if ((!xmlStrcmp(pname, (xmlChar *) "B"))){
	                B = strtod(value, &endp);
			ion_.setB( B ); 
		    }	
	          }
		}
		xpath.str( "" ); 
	        xpath<<"//bio:mechanism["<<n<<"]/bio:parameter/bio:group";	   
	        result2 = getnodeset((xmlChar *)xpath.str().c_str());
	        nodeset = result2->nodesetval;
  	      	grpsize = (nodeset) ? nodeset->nodeNr : 0;
		ion_.unsetGroups();
	   	for(int i = 0; i < grpsize; ++i) {
     		   assert(nodeset->nodeTab[i]);
     		   if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       		     cur = nodeset->nodeTab[i];  
       		     if(cur->ns) {
		        group = (char *)xmlNodeGetContent(cur);
                       	std::string group_str = (char *) group;
			ion_.setGroups( group_str );
			xmlFree(group);
		     }
		   }
		}
	     }
	}	 

      }
   }
 return &ion_;
}*/

