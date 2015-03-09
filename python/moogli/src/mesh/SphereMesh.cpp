#include "mesh/SphereMesh.hpp"

using namespace std;
using namespace osg;

SphereMesh::SphereMesh()
{ }

const tuple< const Vec3Array * const
           , const DrawElementsUShort * const
           , const Vec3Array * const
           >
SphereMesh::unit(unsigned int points)
{
    //points should always be even

    // First check if sphere with the required number of points already exists.
    auto result = spheres.find(points);
    if(result != spheres.end()) { return result -> second; }

    // If a sphere is not found then create, cache and return it.

    unsigned int vertex_count   = (points * points) / 2 - points + 2;
    unsigned int triangle_count = 3 * points * points - 6 * points;

    unsigned int vertex_index   = 1;
    unsigned int triangle_index = 0;

    unsigned int i, j;
    unsigned int alpha, beta;

    float phi_delta   = 2.0 * M_PI / points;
    float theta_delta = 2.0 * M_PI / points;
    float z;
    float radius;

    Vec3Array          * vertices = new Vec3Array(vertex_count);
    Vec3Array          * normals  = new Vec3Array(vertex_count);
    DrawElementsUShort * indices  = new DrawElementsUShort(GL_TRIANGLES, triangle_count);

    // Excluding the topmost and bottommost vertices there will be n - 2 vertices
    // So there will be (n - 2)/2 vertices per side
    // There will be a angle delta of n/2 between each vertex.

    (*vertices)[0] = (*normals)[0] = Vec3f(0.0f, 0.0f, 1.0f);

    for(i = 1; i < points / 2; ++i)
    {
        z = cos( i * phi_delta);
        radius = sin( i * phi_delta );
        for(j = 0; j < points; ++j)
        {
            (*vertices)[vertex_index] =
            (*normals)[vertex_index] =
                Vec3f( radius * cos(j * theta_delta)
                     , radius * sin(j * theta_delta)
                     , z
                     );
            ++vertex_index;
        }
    }

    (*vertices)[vertex_index] =
    (*normals)[vertex_index] = Vec3f(0.0f, 0.0f, -1.0f);

    for(i = 0; i < points; ++i)
    {
        (*indices)[triangle_index]     = 0;
        (*indices)[triangle_index + 1] = i + 1;
        (*indices)[triangle_index + 2] = i + 2;
        triangle_index += 3;
    }

    (*indices)[triangle_index - 1]     = 1;

    for(i = 1; i < points / 2 - 1; ++i)
    {
        alpha = 1 + points * (i - 1);
        beta  = 1 + points * i;

        for(j = 0; j < points; ++j)
        {
            (*indices)[triangle_index    ] = alpha + j;
            (*indices)[triangle_index + 1] = beta  + j;
            (*indices)[triangle_index + 2] = beta  + j + 1;

            (*indices)[triangle_index + 3] = alpha + j + 1;
            (*indices)[triangle_index + 4] = alpha + j;
            (*indices)[triangle_index + 5] = beta  + j + 1;
            triangle_index += 6;
        }
        (*indices)[triangle_index - 4] = beta;
        (*indices)[triangle_index - 3] = alpha;
        (*indices)[triangle_index - 1] = beta;
    }

    for(i = 0; i < points; ++i)
    {
        (*indices)[triangle_index    ] = vertex_count - 1;
        (*indices)[triangle_index + 1] = beta + i + 1;
        (*indices)[triangle_index + 2] = beta + i;
        triangle_index += 3;
    }

    (*indices)[triangle_index - 2]     = beta;

    auto insert_position = spheres.insert(make_pair( points
                            , make_tuple( vertices
                                        , indices
                                        , normals
                                        )
                            )
                  );
    return insert_position.first -> second;
}

Geometry*
SphereMesh::operator()( Vec3f        center
                      , float        radius
                      , unsigned int points
                      )
{
    Geometry* sphere_geometry = new Geometry();

    const auto arrays = unit(points);

    const auto unit_vertices = get<0>(arrays);
    const auto unit_indices  = get<1>(arrays);
    const auto unit_normals  = get<2>(arrays);

    Vec3Array          * vertices = new Vec3Array(unit_vertices -> size());
    Vec3Array          * normals  = new Vec3Array(*unit_normals);
    DrawElementsUShort * indices  = new DrawElementsUShort(*unit_indices);

    std::transform( unit_vertices -> begin()
                  , unit_vertices -> end()
                  , vertices -> begin()
                  , [&](const Vec3f& vertex)
                    {
                        return vertex * radius + center;
                    }
                  );

    sphere_geometry -> setVertexArray(vertices);
    sphere_geometry -> setNormalArray(normals);
    sphere_geometry -> setNormalBinding(Geometry::BIND_PER_VERTEX);
    sphere_geometry -> addPrimitiveSet(indices);
    Vec4Array * colors = new Vec4Array();
    colors -> push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f)); //color);
    sphere_geometry -> setColorArray(colors);
    sphere_geometry -> setColorBinding( osg::Geometry::BIND_OVERALL );

    return sphere_geometry;
}
