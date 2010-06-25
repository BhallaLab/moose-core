#ifndef _SEGMENT_H
#define _SEGMENT_H

using namespace std;

class NPoint
{
	
	   	
	public:
		void setX( double x );
		double getX() const ;
		void setY( double y );
		double getY() const ;
		void setZ( double z );
		double getZ() const ;
		void setDiameter( double diameter );
		double getDiameter() const ;
	private:
		double x,y,z,diameter;
};

class Segment
{
	public:
		Segment():
			id( "" ), name( "" ), parent( "" ), cable( "" ),setparent(false),setproximal(false) 
		{;}
		~Segment() {;}
		bool setproximal;
		const std::string& getId() const;
		const std::string& getName() const;
		bool isSetId() const;
		bool isSetName() const;
		void setId(const std::string& id);
		void setName(const std::string& name);
		void unsetId();
		void unsetName();
		void setCable(const std::string& cable );
		const std::string& getCable( ) const;
		void setParent(const std::string& parent );
		const std::string& getParent( ) const;
		bool isSetParent() const;
		bool isSetProximal() const;
		const Segment* getSegment(string filename,string id);
		NPoint proximal,distal;
	protected:
		std::string id;
		std::string name;		
		std::string parent;	
		std::string cable;
		bool setparent;
			
		
		
		

};
#endif // _SEGMENT_H
