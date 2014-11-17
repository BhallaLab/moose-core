#ifndef _GLOBALS_HPP_
#define _GLOBALS_HPP_ value

#include "mesh/CylinderMesh.hpp"
#include "mesh/SphereMesh.hpp"

/* Global variables should be avoided for they increase program’s complexity immensely and because their values can be changed by any function that is called. But it is a necessity in this case.
 * CylinderGeometry and SphereGeometry implement a caching scheme which is effective iff there is one instance of these objects. Hence they are defined as global variables.
 * Another approach would be to define these as singleton classes but that is mostly frowned upon and does not provide us any benefit in comparison to declaring global variables.
*/

extern CylinderMesh cylinder;
extern SphereMesh   sphere;

    // light -> setAmbient( osg::);
    // light -> setDiffuse( osg::);
    // light -> setSpecular( osg::);



#endif
