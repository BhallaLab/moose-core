#include "mesh/CylinderMesh.hpp"
#include <osgUtil/SmoothingVisitor>
#include "utility/record.hpp"


// http://gamedev.stackexchange.com/questions/55803/smooth-shading-vs-flat-shading-whats-the-difference-in-the-models
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

    Vec3Array          * vertices = new Vec3Array(3 * vertex_count);
    Vec3Array          * normals  = new Vec3Array(3 * vertex_count);
    DrawElementsUShort * indices  = new DrawElementsUShort(GL_TRIANGLES, triangle_count);

    float x, y;
    // const float COS_DELTA_THETA         = cos(theta_delta);
    // const float COS_DELTA_THETA_HALF    = cos(theta_delta / 2);
    // const float NORMAL_MAGNITUDE        = sqrt(COS_DELTA_THETA + 3);
    // float x, y, nx, ny;
    // const float NZ = M_SQRT2 / NORMAL_MAGNITUDE;
    // const float NC = M_SQRT2 * COS_DELTA_THETA_HALF / NORMAL_MAGNITUDE;
    float root_2 = sqrt(2.0);
    for(i = 0; i < points; ++i)
    {
        theta = i * theta_delta;
        x = cos(theta);
        y = sin(theta);

        // nx = NC * x;
        // ny = NC * y;

        (*vertices)[i + 4 * points] =
        (*vertices)[i + 2 * points] =
        (*vertices)[i]              = Vec3f(x, y,  0.5f);

        (*vertices)[i + 5 * points] =
        (*vertices)[i + 3 * points] =
        (*vertices)[i + points]     = Vec3f(x, y,  0.5f);



        // (*normals)[i]           = Vec3f(nx, ny,  NZ);
        // (*normals)[i + points]  = Vec3f(nx, ny, -NZ);
        (*normals)[i]              = Vec3f(x,y,0.0f);
        (*normals)[i + points]     = Vec3f(x,y,0.0f);
        (*normals)[i + 2 * points] = Vec3f(0.0f, 0.0f, 1.0f);
        (*normals)[i + 3 * points] = Vec3f(0.0f, 0.0f, -1.0f);
        (*normals)[i + 4 * points] = Vec3f(x, y, 1.0f);
        (*normals)[i + 5 * points] = Vec3f(x, y, -1.0f);
        // (*normals)[i + points] -> normalize();
        // (*normals)[i]           = Vec3f(nx, ny,  0);
        // (*normals)[i + points]  = Vec3f(nx, ny,  0);
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

    for(i = 2 * points + 1; i < 3 * points - 1; ++i)
    {
        (*indices)[triangle_index    ] = 2 * points;
        (*indices)[triangle_index + 1] = i;
        (*indices)[triangle_index + 2] = i + 1;
        triangle_index += 3;
    }

    for(i = 3 * points + 1; i < 4 * points - 1; ++i)
    {
        (*indices)[triangle_index    ] = i + 1;
        (*indices)[triangle_index + 1] = i;
        (*indices)[triangle_index + 2] = 3 * points;
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
    osg::Geometry * cylinder_geometry = new Geometry();
    cylinder_geometry -> setUseDisplayList(false);
    cylinder_geometry -> setUseVertexBufferObjects(true);
    const auto arrays = unit(points);

    const auto unit_vertices = get<0>(arrays);
    const auto unit_indices  = get<1>(arrays);
    const auto unit_normals  = get<2>(arrays);

    Vec3Array * vertices = new Vec3Array(unit_vertices -> size());
    Vec3Array * normals = new Vec3Array(unit_normals -> size());
    DrawElementsUShort * indices = new DrawElementsUShort(*unit_indices);
    cylinder_geometry -> addPrimitiveSet(indices);
    cylinder_geometry -> setVertexArray(vertices);
    cylinder_geometry -> setNormalArray(normals, Array::BIND_PER_VERTEX);
    Vec4Array * colors = new Vec4Array();
    colors -> push_back(color); //color);
    cylinder_geometry -> setColorArray(colors);
    cylinder_geometry -> setColorBinding( osg::Geometry::BIND_OVERALL);

    operator()( cylinder_geometry
              , center
              , upper_radius
              , lower_radius
              , height
              , direction
              , points
              , color
              );
    return cylinder_geometry;
}

void
CylinderMesh::operator()( osg::Geometry * cylinder_geometry
                        , Vec3f        center
                        , float        upper_radius
                        , float        lower_radius
                        , float        height
                        , Vec3f        direction
                        , unsigned int points
                        , const Vec4& color
                        )
{
    const auto arrays = unit(points);

    const auto unit_vertices = get<0>(arrays);
    const auto unit_indices  = get<1>(arrays);
    const auto unit_normals  = get<2>(arrays);

    // std::cerr << "Vertex Array" << cylinder_geometry -> getVertexArray() << std::endl;
    Vec3Array * vertices = static_cast<Vec3Array *>(cylinder_geometry -> getVertexArray());
    Vec3Array * normals = static_cast<Vec3Array *>(cylinder_geometry -> getNormalArray());
    DrawElementsUShort * indices = static_cast<DrawElementsUShort *>(cylinder_geometry -> getPrimitiveSet(0));

    unsigned int i, j;
    Vec3f temp_vertex;
    // RECORD_ERROR("Reaching Here!");
    Quat rotate;
    rotate.makeRotate(osg::Vec3f(0.0f, 0.0f, 1.0f), direction);

    for(i = 0; i < points; ++i)
    {
        temp_vertex.set( upper_radius * (*unit_vertices)[i][0]
                       , upper_radius * (*unit_vertices)[i][1]
                       , height/2.0f
                       );
        (*vertices)[i + 4 * points] =
        (*vertices)[i + 2 * points] =
        (*vertices)[i]              = rotate * temp_vertex + center;
        temp_vertex.set( lower_radius * (*unit_vertices)[i + points][0]
                       , lower_radius * (*unit_vertices)[i + points][1]
                       , -height/2.0f
                       );
        (*vertices)[i + 5 * points] =
        (*vertices)[i + 3 * points] =
        (*vertices)[i +     points] = rotate * temp_vertex + center;
 
        (*normals)[i + points]   =
        (*normals)[i]            = rotate * (*unit_normals)[i];
        (*normals)[i + 2 * points] = direction;
        (*normals)[i + 3 * points] = -direction;
        (*normals)[i + 4 * points] = rotate * (*unit_normals)[i + 4 * points];
        (*normals)[i + 5 * points] = rotate * (*unit_normals)[i + 5 * points];
    }

    // RECORD_ERROR("Reaching Here!");
    // for(i = 0; i < vertices -> size(); ++i)
    // {
    // }
    // cylinder_geometry -> setNormalArray( normals.get()
    //                                    , Array::BIND_PER_VERTEX
    //                                    );
    // polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,6));

    // RECORD_ERROR("Reaching Here!");
    // vertices -> dirty();
    // normals -> dirty();
    // osgUtil::SmoothingVisitor::smooth(*cylinder_geometry);
    // RECORD_ERROR("Reaching Here!");
}
