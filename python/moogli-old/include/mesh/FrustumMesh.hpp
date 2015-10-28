#ifndef __FRUSTUM_MESH__
#define __FRUSTUM_MESH__

#include "utility/libraries.hpp"
#include "mesh/Mesh.hpp"

class FrustumMesh : public Mesh
{
public:

    FrustumMesh(const string & id);

    FrustumMesh( const string & id
               , const osg::Vec3f & center
               , const osg::Vec3f & axis
               , float length
               , float base_radius
               , float apex_radius
               , unsigned int vertices
               , const osg::Vec4f & base_color
               , const osg::Vec4f & apex_color
               );

    FrustumMesh( const string & id
               , const osg::Vec3f & base
               , const osg::Vec3f & apex
               , float base_radius
               , float apex_radius
               , unsigned int vertices
               , const osg::Vec4f & base_color
               , const osg::Vec4f & apex_color
               );
    void
    set_geometry( const osg::Vec3f & center
                , const osg::Vec3f & axis
                , float length
                , float base_radius
                , float apex_radius
                , unsigned int vertices
                );

    void
    set_geometry( const osg::Vec3f & base
                , const osg::Vec3f & apex
                , float base_radius
                , float apex_radius
                , unsigned int vertices
                );

    void
    set( const osg::Vec3f & center
       , const osg::Vec3f & axis
       , float length
       , float base_radius
       , float apex_radius
       , unsigned int vertices
       , const osg::Vec4f & base_color
       , const osg::Vec4f & apex_color
       );

    void
    set( const osg::Vec3f & base
       , const osg::Vec3f & apex
       , float base_radius
       , float apex_radius
       , unsigned int vertices
       , const osg::Vec4f & base_color
       , const osg::Vec4f & apex_color
       );

    void
    set_vertices(unsigned int vertices);

    unsigned int
    get_vertices();

    void
    set_axis(const osg::Vec3f & axis);

    osg::Vec3f
    get_axis() const;

    void
    set_length(const float & length);

    float
    get_length() const;

    void
    set_center(const osg::Vec3f & center);

    osg::Vec3f
    get_center() const;

    void
    set_apex(const osg::Vec3f & apex);

    osg::Vec3f
    get_apex() const;

    void
    set_base(const osg::Vec3f & base);

    osg::Vec3f
    get_base() const;

    void
    set_base_radius(const float base_radius);

    float
    get_base_radius() const;

    void
    set_apex_radius(const float apex_radius);

    float
    get_apex_radius() const;

    void
    set_radii(float base_radius, float apex_radius);

    void
    set_radii(float radius);

    void
    set_base_color(const osg::Vec4f & base_color);

    void
    set_apex_color(const osg::Vec4f & apex_color);

    const osg::Vec4f &
    get_base_color() const;

    const osg::Vec4f &
    get_apex_color() const;

    void
    set_colors(const osg::Vec4f & color);

    void
    set_colors(const osg::Vec4f & base_color, const osg::Vec4f & apex_color);

    void
    move_apex_by(float dl);

    void
    move_base_by(float dl);

    void
    move_apex_along(const osg::Vec3f & direction, float dl);

    void
    move_base_along(const osg::Vec3f & direction, float dl);

    void
    move_center_by(const osg::Vec3f & displacement);

private:
    osg::Vec3f _axis;
    osg::Vec3f _base;
    osg::Vec3f _apex;
    osg::Vec3f _center;
    float _base_radius;
    float _apex_radius;
    float _length;
    unsigned int _vertices;
    osg::Vec4f _base_color;
    osg::Vec4f _apex_color;
};





// void
// allocate()
// {
//     setVertexArray(new osg::Vec3Array(_vertices * 4));
//     setNormalArray( new osg::Vec3Array(_vertices * 4)
//                   , osg::Array::BIND_PER_VERTEX
//                   );
//     addPrimitiveSet( new DrawElementsUShort( GL_TRIANGLES
//                                            , 12 * _vertices - 12
//                                            )
//                    );
//     setColorArray( new Vec4Array(4 * _vertices)
//                  , osg::Array::BIND_PER_VERTEX
//                  );
// }

// void
// _construct_vertices()
// {
//     osg::Vec3Array * V = getVertexArray();
//     osg::Quat rotation( osg::Vec3f(0.0f, 0.0f, 1.0f)
//                       , direction
//                       );
//     create_base_and_apex_polygons( V
//                                  , 0
//                                  , _vertices
//                                  , _vertices
//                                  , _apex_radius
//                                  , _base_radius
//                                  , rotation
//                                  , _apex
//                                  , _base
//                                  );
//     std::copy( V
//              , V -> begin()
//              , V -> begin() + _vertices
//              , V -> begin() + 2 * _vertices
//              );

//     std::copy( V
//              , V -> begin() + _vertices
//              , V -> begin() + 2 * _vertices
//              , V -> begin() + 3 * _vertices
//              );
// }

// void
// create_polygon( osg::Vec3f * V
//               , uint start_index
//               , uint vertices
//               , float z
//               )
// {
//     V -> 
// }

// void
// _construct_normals()
// {
//     float radii_difference = _base_radius - _apex_radius;
//     float cos_phi = radii_difference
//                   / sqrt( radii_difference * radii_difference
//                         + _length * _length
//                         );
//     create_polygon( N
//                   , 0
//                   , _vertices
//                   , cos_phi
//                   );
//     std::copy( N
//              , N -> begin()
//              , N -> begin() + _vertices
//              , N -> begin() + _vertices
//              );
//     std::fill(N -> begin() + 2 * _vertices, direction);
//     std::fill(N -> begin() + 3 * _vertices, -direction);
// }

// void
// construct()
// {
//     _construct_vertices();
//     _construct_normals();
// }

// void
// color()
// {
//     C = getColorArray();
//     std::fill( C -> begin()
//              , C -> begin() + _vertices
//              , _apex_color
//              );
//     std::fill( C -> begin() + _vertices
//              , C -> begin() + 2 * _vertices
//              , _base_color
//              );
//     std::fill( C -> begin() + 2 * _vertices
//              , C -> begin() + 3 * _vertices
//              , _apex_color
//              );
//     std::fill( C -> begin() + 3 * _vertices
//              , C -> begin() + 4 * _vertices
//              , _base_color
//              );
// }
#endif /* __FRUSTUM_MESH__ */
