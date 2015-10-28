#include "utility/includes.hpp"
#include "core/Compartment.hpp"

class GeometryUpdateCallback : public osg::Drawable::UpdateCallback
{
public:
    Compartment * compartment;
    unsigned int points;

    GeometryUpdateCallback(Compartment * compartment, unsigned int points);

    virtual void
    update(osg::NodeVisitor * nv, osg::Drawable * drawable);
};

