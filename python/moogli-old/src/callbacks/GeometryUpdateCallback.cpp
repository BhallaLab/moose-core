#include "callbacks/GeometryUpdateCallback.hpp"

GeometryUpdateCallback::GeometryUpdateCallback(Compartment * compartment, unsigned int points)
{
    this -> compartment = compartment;
    this -> points      = points;
}

void
GeometryUpdateCallback::update(osg::NodeVisitor * nv, osg::Drawable * drawable)
{
    osg::Geometry * geometry = static_cast<osg::Geometry *>(drawable);

    if(compartment -> _proximal_d_updated || compartment -> _distal_d_updated)
    {
        compartment -> _center   = (compartment -> _distal + compartment -> _proximal) / 2.0;
        compartment -> _direction = compartment -> _distal - compartment -> _proximal;
        compartment -> _height = compartment -> _direction.normalize();
        double _radius = (compartment -> _proximal_d + compartment -> _distal_d) / 4.0;

        //Floating point comparison is being done with compare_double function.
        if((compare_double(compartment -> _height, _radius, 0.01f) == 0)
        || (compare_double(compartment -> _height, 0.0f, 0.01f) == 0)
        )
        {
            sphere( geometry
                  , compartment -> _center
                  , _radius
                  , points
                  );
        }
        else
        {
            cylinder( geometry
                    , compartment -> _center
                    , compartment -> _distal_d   / 2.0
                    , compartment -> _proximal_d / 2.0
                    , compartment -> _height
                    , compartment -> _direction
                    , points
                    );
        }
        compartment -> _proximal_d_updated = false;
        compartment -> _distal_d_updated = false;
    }
}
