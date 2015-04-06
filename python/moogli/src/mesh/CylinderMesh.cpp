#include "mesh/CylinderMesh.hpp"
#include <osgUtil/SmoothingVisitor>

CylinderMesh::CylinderMesh()
{ }

const tuple< const Vec3Array * const
           , const DrawElementsUShort * const
           , const Vec3Array * const
           >
CylinderMesh::unit(unsigned int points)
{
    //points should always be even

    // First check if cylinder with the required number of points already exists.
    auto result = cylinders.find(points);
    if(result != cylinders.end()) { return result -> second; }

    // If a cylinder is not found then create, cache and return it.

    unsigned int vertex_count   = 2 * points;
    unsigned int triangle_count = 12 * points - 12;

    unsigned int triangle_index = 0;

    unsigned int i, j;
    float theta;

    float theta_delta = 2.0 * M_PI / points;
    float z;
    float radius;

    Vec3Array          * vertices = new Vec3Array(vertex_count);
    Vec3Array          * normals  = new Vec3Array(vertex_count);
    DrawElementsUShort * indices  = new DrawElementsUShort(GL_TRIANGLES, triangle_count);

    const float COS_DELTA_THETA         = cos(theta_delta);
    const float COS_DELTA_THETA_HALF    = cos(theta_delta / 2);
    const float NORMAL_MAGNITUDE        = sqrt(COS_DELTA_THETA + 3);
    float x, y, nx, ny;
    const float NZ = M_SQRT2 / NORMAL_MAGNITUDE;
    const float NC = M_SQRT2 * COS_DELTA_THETA_HALF / NORMAL_MAGNITUDE;

    for(i = 0; i < points; ++i)
    {
        theta = i * theta_delta;
        x = cos(theta);
        y = sin(theta);

        nx = NC * x;
        ny = NC * y;

        (*vertices)[i]          = Vec3f(x, y,  0.5f);
        (*vertices)[i + points] = Vec3f(x, y, -0.5f);

        // (*normals)[i]           = Vec3f(nx, ny,  NZ);
        // (*normals)[i + points]  = Vec3f(nx, ny, -NZ);
        (*normals)[i]           = Vec3f(nx, ny,  0);
        (*normals)[i + points]  = Vec3f(nx, ny,  0);

    }

    for(i = 0; i < points; ++i)
    {
        (*indices)[triangle_index    ] = i;
        (*indices)[triangle_index + 1] = i + points;
        (*indices)[triangle_index + 2] = i + 1;

        (*indices)[triangle_index + 3] = i + points;
        (*indices)[triangle_index + 4] = i + 1 + points;
        (*indices)[triangle_index + 5] = i + 1;

        triangle_index += 6;
    }

    (*indices)[triangle_index - 1]     = 0;
    (*indices)[triangle_index - 2]     = points;
    (*indices)[triangle_index - 4]     = 0;

    for(i = 1; i < points - 1; ++i)
    {
        (*indices)[triangle_index    ] = 0;
        (*indices)[triangle_index + 1] = i;
        (*indices)[triangle_index + 2] = i + 1;
        triangle_index += 3;
    }

    for(i = points + 1; i < 2 * points - 1; ++i)
    {
        (*indices)[triangle_index    ] = i + 1;
        (*indices)[triangle_index + 1] = i;
        (*indices)[triangle_index + 2] = points;
        triangle_index += 3;
    }

    auto insert_position =
        cylinders.insert( make_pair( points
                                   , make_tuple( vertices
                                               , indices
                                               , normals
                                               )
                                   )
                        );
    return insert_position.first -> second;
}

Geometry *
CylinderMesh::operator()( Vec3f        center
                        , float        upper_radius
                        , float        lower_radius
                        , float        height
                        , Vec3f        direction
                        , unsigned int points
                        , const Vec4& color
                        )
{
    ref_ptr<Geometry> cylinder_geometry(new Geometry());

    const auto arrays = unit(points);

    const auto unit_vertices = get<0>(arrays);
    const auto unit_indices  = get<1>(arrays);
    const auto unit_normals  = get<2>(arrays);

    ref_ptr<Vec3Array> vertices(
        new Vec3Array(unit_vertices -> size())
                               );
    ref_ptr<Vec3Array> normals(
        new Vec3Array(unit_normals -> size())
                              );
    ref_ptr<DrawElementsUShort> indices(
        new DrawElementsUShort(*unit_indices)
                                       );

    unsigned int i;
    Vec3f temp_vertex;

    Quat rotate;
    rotate.makeRotate(osg::Vec3f(0.0f, 0.0f, 1.0f), direction);

    for(i = 0; i < unit_vertices -> size() / 2; ++i)
    {
        temp_vertex.set( upper_radius * (*unit_vertices)[i][0]
                       , upper_radius * (*unit_vertices)[i][1]
                       , height/2.0f
                       );
        (*vertices)[i] = rotate * temp_vertex + center;

        (*normals)[i]  = rotate * (*unit_normals)[i];
    }

    for(; i < unit_vertices -> size(); ++i)
    {
        temp_vertex.set( lower_radius * (*unit_vertices)[i][0]
                       , lower_radius * (*unit_vertices)[i][1]
                       , -height/2.0f
                       );
        (*vertices)[i] = rotate * temp_vertex + center;

        (*normals)[i]  = rotate * (*unit_normals)[i];
    }

    Vec4Array * colors = new Vec4Array();
    colors -> push_back(osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f)); //color);
    // for(i = 0; i < vertices -> size(); ++i)
    // {
    // }
    // cylinder_geometry -> setNormalArray( normals.get()
    //                                    , Array::BIND_PER_VERTEX
    //                                    );
    // polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,6));
    cylinder_geometry -> addPrimitiveSet(indices.get());
    cylinder_geometry -> setVertexArray(vertices.get());
    cylinder_geometry -> setColorArray(colors);
    cylinder_geometry -> setColorBinding( osg::Geometry::BIND_OVERALL );
    osgUtil::SmoothingVisitor::smooth(*cylinder_geometry);
    return cylinder_geometry.release();
}
