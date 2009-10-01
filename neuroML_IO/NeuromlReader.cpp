#include "moose.h"
#include "element/Neutral.h"
#include <iostream>
#include "neuroML_IO/Cable.h"
#include <neuroML/NCell.h>
#include <neuroML/NBase.h>
#include "Cable.h"
#include "NeuromlReader.h"
using namespace std;
const double NeuromlReader::PI = 3.14159;
void NeuromlReader::readModel( string filename,Id location )
{
   static const Cinfo* compartmentCinfo = initCompartmentCinfo();
   static const Finfo* lengthFinfo = compartmentCinfo->findFinfo( "length" );
   static const Finfo* diameterFinfo = compartmentCinfo->findFinfo( "diameter" );
   static const Finfo* x0Finfo = compartmentCinfo->findFinfo( "x0" );
   static const Finfo* y0Finfo = compartmentCinfo->findFinfo( "y0" );
   static const Finfo* z0Finfo = compartmentCinfo->findFinfo( "z0" );
   static const Finfo* xFinfo = compartmentCinfo->findFinfo( "x" );
   static const Finfo* yFinfo = compartmentCinfo->findFinfo( "y" );
   static const Finfo* zFinfo = compartmentCinfo->findFinfo( "z" );
   static const Finfo* RaFinfo = compartmentCinfo->findFinfo( "Ra" );
   static const Finfo* initVmFinfo = compartmentCinfo->findFinfo( "initVm" );
   static const Finfo* CmFinfo = compartmentCinfo->findFinfo( "Cm" );
   static const Finfo* EmFinfo = compartmentCinfo->findFinfo( "Em" );
   static const Finfo* RmFinfo = compartmentCinfo->findFinfo( "Rm" );
   static const Cinfo* cableCinfo = initCableCinfo();
   NBase nb;   
   NCell* c;
   c = nb.readNeuroML (filename);
   c->register_namespaces();
   double x,y,z,x1,y1,z1,diam1,diam2,diameter,length,initVm,r,ca,Cm,Ra,sa;
   initVm = c->getInit_memb_potential();
   ca = c->getSpec_capacitance();
   r = c->getSpec_axial_resistance();
   static const Finfo* nameFinfo = cableCinfo->findFinfo( "name" );
   static const Finfo* groupsFinfo = cableCinfo->findFinfo( "groups" );
   unsigned int num_cables = c->getNumCables();
   unsigned int num_segments = c->getNumSegments();
   Id loc("/cables");  
   if ( loc.good() ) 
      cerr << "Warning: cables already exists. You are trying to overwrite it" << endl;	
   Element* locE = Neutral::create( "Neutral", "cables", Id(), Id::scratchId() );
   loc = locE->id();
   map< string,vector<string> > groupcableMap; 
   map< string,Id > cabMap;	
   for( int cb = 1; cb <= num_cables; cb++ )
   {
	const NCable * cab;
	cab = c->getCable(cb);
	std::string cid = "", cname = "";
	cid = cab->getId();
	cname = cab->getName(); 
	//cout << "id " << cid << " name " << cname << endl;
	cable_ = Neutral::create( "Cable",cid,loc,Id::scratchId() ); 
	cabMap[cid] = cable_->id();
	set< string >( cable_,nameFinfo,cid );
	std::vector< std::string > groups;
	groups = cab->getGroups();
	set< vector< string > >( cable_,groupsFinfo,groups );
	for ( int itr = 0; itr < groups.size(); itr++ )
	     groupcableMap[groups[itr]].push_back(cid);
   }
   map< string,vector< string > > cablesegMap;
   map< string,Id > segMap;	
   for( int i = 1; i<= num_segments; i ++ )
   {
		
	const Segment * seg ;
	seg = c->getSegment(i);
	std::string id = "", name = "", cable = "", parent = ""; 
	id = seg->getId();
	name = seg->getName();   
	cable = seg->getCable();
	parent = seg->getParent();
	//cout << "id " << id << " name " << name << " cable " << cable << endl; 
	compt_ = Neutral::create( "Compartment",id,location,Id::scratchId() );
	segMap[id] = compt_->id(); 
	if (parent != "")
		Eref(compt_).add("raxial",segMap[parent](),"axial",ConnTainer::Default ); 
	Eref(cabMap[cable]()).add("compartment",Eref(compt_),"cable",ConnTainer::Default ); 
	cablesegMap[cable].push_back(id);
	x = seg->proximal.getX();
	y = seg->proximal.getY();
	z = seg->proximal.getZ();
	x1 = seg->distal.getX();
	y1 = seg->distal.getY();
	z1 = seg->distal.getZ();
	diam1 = seg->proximal.getDiameter();
	diam2 = seg->distal.getDiameter();
    	set< double >( compt_, x0Finfo, x );
	set< double >( compt_, y0Finfo, y );
 	set< double >( compt_, z0Finfo, z );
 	set< double >( compt_, xFinfo, x1  );
	set< double >( compt_, yFinfo, y1 );
	set< double >( compt_, zFinfo, z1 );
	diameter = (diam1 + diam2 )/2;
	diameter *= 1e-4;
	length = sqrt(pow((x1-x),2) + pow((y1-y),2) + pow((z1-z),2));
	length *= 1e-4;
	set< double >( compt_, diameterFinfo, diameter );
	set< double >( compt_, lengthFinfo, length );
	if ( length == 0 ){
		Cm = ca * PI * diameter * diameter;
		Ra = 13.50 * r /( diameter * PI );
		sa = 4 * PI * diameter * diameter/4;
	}
	else{
        	Cm = ca * length * PI * diameter;
        	Ra = (r * length )/(PI * diameter * diameter/4);
		sa = (2 * PI * (diameter * diameter)/4)+(2 * PI * diameter/2 * length);
	}
        set< double >( compt_, CmFinfo, Cm );
        set< double >( compt_, RaFinfo, Ra ); 
        set< double >( compt_, initVmFinfo, initVm );
	set< double >( compt_, EmFinfo, initVm );
    }
    
    static const Cinfo* channelCinfo = initHHChannelCinfo();
    static const Finfo* gbarFinfo = channelCinfo->findFinfo( "Gbar" );
    static const Finfo* ekFinfo = channelCinfo->findFinfo( "Ek" );
    static const Finfo* xpowerFinfo = channelCinfo->findFinfo( "Xpower" );
    static const Finfo* ypowerFinfo = channelCinfo->findFinfo( "Ypower" );
    static const Finfo* zpowerFinfo = channelCinfo->findFinfo( "Zpower" );
    unsigned int num_channels = c->getNumChannels();
    map< string,vector<string> >::iterator cmap_iter;
    map< string,vector<string> >::iterator smap_iter; 
    for( int ch = 1; ch <= num_channels; ch++ )
    {
    	Channel * chl ;
	chl = c->getChannel( ch );
	string name = "", group = "";
	double gmax,ek;
	Id loc( "/library" );
	name = chl->getName();
 	 channel_ = Neutral::create( "HHChannel",name,loc,Id::scratchId() );
	 gmax = chl->getParameterValue();
	 double gbar = gmax * sa;
	 set< double >( channel_, gbarFinfo, gbar );
	 ek = chl->getDefault_erev();
	 set< double >( channel_, ekFinfo, ek );
	 double xmin,xmax;
	 int xdivs;
	 string units = "Physiological Units";
	 if ( units == "Physiological Units" ){
	 	xmin = -100;
	 	xmax = 50;
	 }
	 else if ( units == "SI Units" ){
	 	xmin = -0.1;
	 	xmax = 0.05;
	 }
	 xdivs = 3000;
	 string path = Eref(channel_).id().path();
	 vector< string > gatename;
	 gatename.push_back("/xGate");
	 gatename.push_back("/yGate");
	 gatename.push_back("/zGate");
	 unsigned int  num = chl->getNumGates();
	 for( int i=1;i<=num;i++ )
	 {	
		Gate* gat ;		
		gat = chl->getGate(i);
		string name = gat->getGateName();
		double power = gat->getInstances();
		if ( i == 1 )
			set< double >( channel_, xpowerFinfo, power );
		else if ( i == 2 )
			set< double >( channel_, ypowerFinfo, power );
		else if ( i == 3 )
			set< double >( channel_, zpowerFinfo, power );
		string gatepath = path + gatename[i-1];
		//cout << " gate path : " << gatepath << endl;
	 	Id gateid(gatepath);
		vector< double > result;
		result.clear();
		string Aexpr_form = gat->alpha.expr_form;
		double Arate = gat->alpha.rate;
		double Ascale = gat->alpha.scale;
		double Amidpoint = gat->alpha.midpoint;
		pushtoVector(result,Aexpr_form,Arate,Ascale,Amidpoint);
		string Bexpr_form = gat->beta.expr_form;
		double Brate = gat->beta.rate;
		double Bscale = gat->beta.scale;
		double Bmidpoint = gat->beta.midpoint;
		pushtoVector(result,Bexpr_form,Brate,Bscale,Bmidpoint);
		result.push_back(xdivs);
		result.push_back(xmin);
		result.push_back(xmax);
		//cout << "result size : "<< result.size() << endl;
		set< vector< double > >( Eref(gateid()),"setupAlpha",result );
	 }
	 bool passive = chl->getPassivecond();
	 group = chl->getGroup();
	// cout<< " group: " << group ;
	 vector< string > cId ;
	 vector< string > sId;
	 cmap_iter = groupcableMap.find( group );
	 if ( cmap_iter != groupcableMap.end() )
	    cId = cmap_iter->second;
	 else {
	   cout << "Error: CableId is empty" << endl;
	   return;
	 }
	 for( int i = 0; i < cId.size(); i++ )
	 {
	   //cout << " cable id:  " << cId[i];
	    smap_iter = cablesegMap.find( cId[i] );
	    if ( smap_iter != cablesegMap.end() )
		sId = smap_iter->second;
	    else{
		cout << "Error:SegmentId is empty" << endl;
		return;
	    }
	    for( int j = 0; j < sId.size(); j++ ){
		//cout << " seg Id:  " << sId[j] << endl; 
		Id comptEl = segMap[sId[j]];
		if (passive){
			
	   		double Rm = 1/(gmax*sa);
	   		set< double >( comptEl(), RmFinfo, Rm );
			set< double >( comptEl(), EmFinfo, ek );
	 	}
		else{
			Element* copyEl = channel_->copy(comptEl(),channel_->name());
			Eref(comptEl()).add("channel",copyEl,"channel",ConnTainer::Default ); 
		}
           }
	 
     }
}   
}
void NeuromlReader::pushtoVector(vector< double > &result,string expr_form,double r,double s,double m)
{
	double A,B,C,D,F;
	if( expr_form == "exp_linear" ){
		A = r*m/s;
		B = -r/s;
		C = -1;
		D = -m;
		F = -s;	
	}
	if( expr_form == "sigmoid" ){
		A = r;
		B = 0;
		C = 1;
		D = -m;
		F = s;	
	}
	if( expr_form == "exponential" ){
		A = r;
		B = 0;
		C = 0;
		D = -m;
		F = -s;
	}
	result.push_back(A);
	result.push_back(B);
	result.push_back(C);
	result.push_back(D);
	result.push_back(F);	
}



  

  

