#include "core/Morphology.hpp"

using namespace std;
using namespace osg;


// http://webee.technion.ac.il/~cgcourse/classes/class5.pdf
// http://www.cuboslocos.com/tutorials/OSG-BasicLighting
// http://trac.openscenegraph.org/projects/osg//wiki/Support/Tutorials
// http://hi.baidu.com/brianlanbo/item/411aad00ff89216fd45a1198
// http://merlin.fit.vutbr.cz/wiki/index.php/OSG_knowledge_base

Morphology::Morphology( const string&       name
                      , const unsigned int  lod_resolution
                      , const float         lod_distance_delta
                      , const unsigned int  min_points
                      , const unsigned int  points_delta
                      ) : name(name)
                        , lod_resolution(lod_resolution)
                        , lod_distance_delta(lod_distance_delta)
                        , min_points(min_points)
                        , points_delta(points_delta)
                        , _matrix_transform(new MatrixTransform())
                        , _state_set(new StateSet())
                        , _lightcount(0)
{
    _initialize();
}


Morphology::Morphology( const Morphology & morphology
                      ) : name(morphology.name)
                        , lod_resolution(morphology.lod_resolution)
                        , lod_distance_delta(morphology.lod_distance_delta)
                        , min_points(morphology.min_points)
                        , points_delta(morphology.points_delta)
                        , _matrix_transform(new MatrixTransform())
                        , _state_set(new StateSet())
                        , _lightcount(0)
{
    _initialize();
}

Morphology::Morphology( const char *        name
                      , const unsigned int  lod_resolution
                      , const float         lod_distance_delta
                      , const unsigned int  min_points
                      , const unsigned int  points_delta
                      ) : name(name)
                        , lod_resolution(lod_resolution)
                        , lod_distance_delta(lod_distance_delta)
                        , min_points(min_points)
                        , points_delta(points_delta)
                        , _matrix_transform(new MatrixTransform())
                        , _state_set(new StateSet())
                        , _lightcount(0)
{
    _initialize();
}
// Morphology::Morphology( const char * name
//                       , const unsigned int  lod_resolution
//                       , const float         lod_distance_delta
//                       , const unsigned int  min_points
//                       ) : Morphology::Morphology(string(name))
// { }

void
Morphology::_initialize()
{
    _initialize_state_set();
    // _initialize_lights();
}

void
Morphology::_initialize_state_set()
{

    //1 _state_set -> setMode( GL_RESCALE_NORMAL, StateAttribute::ON );
    //3 _state_set -> setMode( GL_BLEND, StateAttribute::ON );
    //4 _state_set -> setRenderingHint( StateSet::TRANSPARENT_BIN );

    // Enable depth test so that an opaque polygon will occlude a transparent one behind it.
    _state_set->setMode( GL_DEPTH_TEST, osg::StateAttribute::ON );

    // Conversely, disable writing to depth buffer so that
    // a transparent polygon will allow polygons behind it to shine thru.
    // OSG renders transparent polygons after opaque ones.
    Depth * depth = new Depth();
    depth -> setWriteMask( true );
    _state_set->setAttributeAndModes( depth, StateAttribute::ON );

    // _state_set -> setAttribute(_create_material(), StateAttribute::ON);

    // _state_set -> setMode(GL_LIGHTING, StateAttribute::ON);

    // _state_set -> setAttribute(_create_shade_model());


    // _matrix_transform -> setStateSet(_state_set.get());
}

Material *
Morphology::_create_material( const Vec4& diffuse
                            , const Vec4& specular
                            , const Vec4& emission
                            , float shininess
                            , float alpha
                            )
{
    Material * material = new Material();
    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 64.0f);
    // material -> setColorMode(Material::DIFFUSE);
    // material -> setDiffuse(  Material::FRONT_AND_BACK, diffuse);
    // material -> setSpecular( Material::FRONT_AND_BACK, specular);
    // material -> setEmission( Material::FRONT_AND_BACK, emission);
    // material -> setShininess(Material::FRONT_AND_BACK, shininess);
    // material -> setAlpha (   Material::FRONT_AND_BACK, alpha);
    return material;
}

ShadeModel *
Morphology::_create_shade_model()
{
    ShadeModel * shade_model = new ShadeModel();
    shade_model -> setMode(osg::ShadeModel::FLAT);
    return shade_model;
}

void
Morphology::_initialize_lights()
{
    //http://sjbaker.org/steve/omniv/opengl_lighting.html
    LightSource * light_source = _create_light_source();
    // light_source -> setLight(_create_light( Vec4(0.0f, 0.0f, 0.0f, 0.0f)
    //                                       , X_AXIS
    //                                       )
    //                         );
    // light_source -> setLight(_create_light( Vec4(0.0f, 0.0f, 0.0f, 0.0f)
    //                                       , Y_AXIS
    //                                       )
    //                         );
    // light_source -> setLight(_create_light( Vec4(0.0f, 0.0f, 0.0f, 0.0f)
    //                                       , Z_AXIS
    //                                       )
    //                         );

    _matrix_transform -> addChild(light_source);
    _state_set -> setMode( GL_LIGHTING      , StateAttribute::ON );

}

LightSource *
Morphology::_create_light_source()
{
    LightSource * light_source = new LightSource;
    light_source -> setReferenceFrame( LightSource::ABSOLUTE_RF );
    return light_source;
}

Light *
Morphology::_create_light( const Vec4& position
                         , const Vec3& direction
                         , const Vec4& ambient
                         , const Vec4& diffuse
                         , const Vec4& specular
                         , float constant_attenuation
                         , float linear_attenuation
                         , float quadratic_attenuation
                         , float spot_exponent
                         , float spot_cutoff
                         )
{
    Light * light = new Light();
    light -> setLightNum(_lightcount);
    light -> setAmbient(ambient);
    // light -> setDiffuse(diffuse);
    // light -> setSpecular(specular);
    // light -> setPosition(position);
    // light -> setDirection(direction);
    // light -> setSpotCutoff(spot_cutoff);
    // light -> setConstantAttenuation(constant_attenuation);
    // light -> setLinearAttenuation(linear_attenuation);
    // light -> setQuadraticAttenuation(quadratic_attenuation);
    // light -> setSpotExponent(spot_exponent);
    // light -> setSpotCutoff(spot_cutoff);
    // https://www.khronos.org/opengles/sdk/1.1/docs/man/glLight.xml
    // The operation (GL_LIGHTi = GL_LIGHT0 + i) is mandated by the standard.
    _state_set -> setMode( GL_LIGHT0 + _lightcount , StateAttribute::ON );
    ++_lightcount;
    return light;
}



// Morphology::create_view( const unsigned int  lod_resolution
//                         , const float         lod_distance_delta
//                         , const unsigned int  min_points
//                         )
// {
//     MatrixTransform * morphology_node = new MatrixTransform();
//     for(auto& neuron : _neurons)
//     {
//         LOD * neuron_node = new LOD();
//         neuron_node -> setName(neuron.first);
//         for (int lod_index = 0; lod_index < lod_resolution; ++lod_index)
//         {
//             neuron_node -> addChild( new Geode()
//                                    , range
//                                    , range + lod_distance_delta
//                                    );
//             range += lod_distance_delta;
//         }

//         // Model visibility should be across the entire range of floats.
//         // The geode added in the end should be the worst quality and visible
//         // no matter how far the model is from the viewer's eyes.
//         // Hence we need to reset the max range of this geode.
//         neuron_node -> setRange( lod_resolution - 1
//                                , range - lod_distance_delta
//                                , FLT_MAX
//                                );

//         for (auto& compartment : neuron.second)
//         {
//             auto& geometries = compartment -> create_view( lod_resolution
//                                                          , lod_distance_delta
//                                                          , min_points
//                                                          );
//             for (int lod_index = 0; lod_index < lod_resolution; ++lod_index)
//             {
//                 Geode * geode = neuron_node -> getChild(lod_index)
//                 geode -> addChild(geometries[lod_index].get());
//             }
//         }
//         morphology_node -> addChild(neuron_node);
//     }
//     return morphology_node;
// }

bool
Morphology::add_compartment( const string &  compartment_id
                           , const string &  neuron_id
                           , double          proximal_x
                           , double          proximal_y
                           , double          proximal_z
                           , double          proximal_d
                           , double          distal_x
                           , double          distal_y
                           , double          distal_z
                           , double          distal_d
                           )
{
    // RECORD_INFO( "Compartment => " + compartment_id
    //            + "Neuron      => " + neuron_id
    //            );

    _add_neuron(neuron_id);

    Compartment * compartment = new Compartment( compartment_id
                                               , neuron_id
                                               );

    compartment -> set_proximal_parameters( proximal_x
                                          , proximal_y
                                          , proximal_z
                                          , proximal_d
                                          );

    compartment -> set_distal_parameters( distal_x
                                        , distal_y
                                        , distal_z
                                        , distal_d
                                        );

    compartment -> create_geometry( lod_resolution
                                  , lod_distance_delta
                                  , min_points
                                  , points_delta
                                  , _state_set.get()
                                  );

    pair<compartment_map_t::iterator ,bool> result =
        _compartments.insert(make_pair(compartment_id, compartment));

    if(! result.second)
    {
        RECORD_ERROR(compartment_id + " already exists!");
        delete compartment;
    }
    else
    {
        _neurons_compartments[neuron_id].insert(compartment);
        _attach_compartment_geometry(compartment);
    }

    return result.second;
}

int
Morphology::add_compartment( const char *    compartment_id
               , const char *    neuron_id
               , double          proximal_x
               , double          proximal_y
               , double          proximal_z
               , double          proximal_d
               , double          distal_x
               , double          distal_y
               , double          distal_z
               , double          distal_d
               )
{
    return add_compartment( string(compartment_id)
                         , string(neuron_id)
                         , proximal_x
                         , proximal_y
                         , proximal_z
                         , proximal_d
                         , distal_x
                         , distal_y
                         , distal_z
                         , distal_d
                         );
}


bool
Morphology::_add_neuron(const string &  neuron_id)
{
    if(_neurons.find(neuron_id) != _neurons.end())
    {
        return false;
    }

    LOD * neuron_node = new LOD();
    neuron_node -> setName(neuron_id);
    // neuron_node->setNodeMask(0xffffffff);
    // neuron_node -> setStateSet(_state_set.get());

    float range = FLT_MIN;

    for(unsigned int i = 0; i < lod_resolution; ++i)
    {
        Geode * geode = new Geode();
        geode -> setName(neuron_id);
        // geode -> setNodeMask(0xffffffff);
        // geode -> setStateSet(_state_set.get());
        neuron_node -> addChild( geode
                               , range
                               , range + lod_distance_delta
                               );
        range += lod_distance_delta;
    }
    // Model visibility should be across the entire range of floats.
    // The geode added in the end should be the worst quality and visible
    // no matter how far the model is from the viewer's eyes.
    // Hence we need to reset the max range of this geode.
    neuron_node -> setRange( lod_resolution - 1
                           , range - lod_distance_delta
                           , FLT_MAX
                           );

    _neurons.insert(
        make_pair( neuron_id, ref_ptr<LOD>(neuron_node)));

    _neurons_compartments.insert(
        make_pair( neuron_id, unordered_set<Compartment *>()));

    _matrix_transform -> addChild(neuron_node);
    return true;
}

/*
source_list  -> list of python strings
spatial_data -> numpy array with 8 columns and same number of rows as the source list
 no error checking for data types or dimensions of array.
function logs if all elements are inserted or not. returns the current size of morphology in terms of the number of compartments.
*/


bool
Morphology::remove_compartment(const string & compartment_id)
{

    compartment_map_t::iterator result =
        _compartments.find(compartment_id);

    if(result == _compartments.end())
    {
        RECORD_ERROR(compartment_id + " does not exist.");
        return false;
    }

    Compartment * compartment = result -> second;

    _detach_compartment_geometry(compartment);

    delete compartment;

    _compartments.erase(compartment_id);
    _neurons_compartments[compartment -> neuron_id].erase(compartment);
    return true;
}


bool
Morphology::remove_neuron(const string & neuron_id)
{
    neuron_map_t::iterator iter =
        _neurons.find(neuron_id);

    if(iter == _neurons.end())
    {
        RECORD_ERROR(neuron_id + " is not a valid neuron id.");
        return false;
    }

    _detach_neuron_geometry(iter -> second);

    _neurons.erase(neuron_id);

    _neurons_compartments.erase(neuron_id);

    bool result = true;

    for(auto & compartment : _neurons_compartments[neuron_id])
    {
        result = result
               && remove_compartment(compartment -> id);
    }

    return result;
}

bool
Morphology::neuron_is_hidden(const string & neuron_id)
{
    neuron_map_t::iterator result =
        _neurons.find(neuron_id);

    if(result == _neurons.end())
    {
        RECORD_ERROR(neuron_id + " is not a valid neuron id.");
        return false;
    }

    ;
    return (result -> second -> getNodeMask() == 0);
}

bool
Morphology::compartment_is_hidden(const string & compartment_id)
{
    compartment_map_t::iterator result =
        _compartments.find(compartment_id);

    if(result == _compartments.end())
    {
        RECORD_ERROR(compartment_id + " is not a valid compartment id.");
        return false;
    }

    return (result -> second -> geometries[0] -> getNumParents() == 0);
}

void
Morphology::_detach_neuron_geometry(ref_ptr<LOD>& neuron_node)
{
    _matrix_transform -> removeChild(neuron_node.get());
}

void
Morphology::_attach_neuron_geometry(ref_ptr<LOD>& neuron_node)
{
    _matrix_transform -> addChild(neuron_node.get());
}

void
Morphology::_detach_compartment_geometry(Compartment * compartment)
{
    /* If the compartment being removed is the only one
       in the neuron then just detach the neuron itself
    */
    auto& neuron = _neurons[compartment -> neuron_id];
    for(unsigned int i = 0; i < lod_resolution; ++i)
    {
        Geode * geode = (Geode *)(neuron -> getChild(i));
        geode -> removeDrawable(compartment -> geometries[i].get());
    }
}

void
Morphology::_attach_compartment_geometry(Compartment * compartment)
{
    auto& neuron = _neurons[compartment -> neuron_id];
    for(unsigned int i = 0; i < lod_resolution; ++i)
    {
        Geode * geode = (Geode *)(neuron -> getChild(i));
        geode -> addDrawable(compartment -> geometries[i].get());
    }
}

bool
Morphology::hide_neuron(const string & neuron_id)
{
    neuron_map_t::iterator result =
        _neurons.find(neuron_id);

    if(result == _neurons.end())
    {
        RECORD_ERROR(neuron_id + " is not a valid neuron id.");
        return false;
    }

    result -> second -> setNodeMask(0);
    return true;
}

bool
Morphology::show_neuron(const string & neuron_id)
{
    neuron_map_t::iterator result =
        _neurons.find(neuron_id);

    if(result == _neurons.end())
    {
        RECORD_ERROR(neuron_id + " is not a valid neuron id.");
        return false;
    }

    result -> second -> setNodeMask(~0);
    return true;
}

bool
Morphology::show_compartment(const string & compartment_id)
{
    compartment_map_t::iterator result =
        _compartments.find(compartment_id);

    if(result == _compartments.end())
    {
        RECORD_ERROR(compartment_id + " is not a valid compartment id.");
        return false;
    }
    _attach_compartment_geometry(result -> second);
    return true;
}

bool
Morphology::hide_compartment(const string & compartment_id)
{
    compartment_map_t::iterator result =
        _compartments.find(compartment_id);

    if(result == _compartments.end())
    {
        RECORD_ERROR(compartment_id + " is not a valid compartment id.");
        return false;
    }
    _detach_compartment_geometry(result -> second);
    return true;
}

PyObject *
Morphology::set_compartment_order(PyObject * compartment_order)
{
    if(PySequence_Check(compartment_order) != 1)
    {
        RECORD_ERROR("Invalid data structure provided for compartment order.");
        Py_RETURN_FALSE;
    }
    if(static_cast<unsigned int>(PySequence_Length(compartment_order)) != _compartments.size())
    {
        RECORD_ERROR("Sequence doesn't include all compartments of the mode.");
        Py_RETURN_FALSE;
    }
    unsigned int i;
    _compartment_order.clear();
    _compartment_order.resize(_compartments.size());
    for(i = 0; i < _compartments.size();++i)
    {
        PyObject * object = PySequence_GetItem(compartment_order, i);
        _compartment_order[i] = _compartments[string(PyString_AsString(object))];
    }
    Py_RETURN_TRUE;
}

void
Morphology::set_initial_color(float r, float g, float b, float a)
{
    _initial_color.set(r, g, b, a);
}

void
Morphology::set_final_color(float r, float g, float b, float a)
{
    _final_color.set(r, g, b, a);
}

void
Morphology::set_base_membrane_voltage(double base_vm)
{
    _base_vm = base_vm;
}

void
Morphology::set_peak_membrane_voltage(double peak_vm)
{
    _peak_vm = peak_vm;
}

// PyObject *
// Morphology::set_final_color(float r, float g, float b, float a)
// {
//     if(PySequence_Check(color) != 1)
//     {
//         RECORD_ERROR("Invalid data structure provided for final color.");
//         Py_RETURN_FALSE;
//     }
//     if(PySequence_Length(color) != 4)
//     {
//         RECORD_ERROR("Sequence doesn't include all 4 components(r, g, b, a).");
//         Py_RETURN_FALSE;
//     }

//     _final_color.set( PyFloat_AS_DOUBLE(PySequence_GetItem(color, 0))
//                     , PyFloat_AS_DOUBLE(PySequence_GetItem(color, 1))
//                     , PyFloat_AS_DOUBLE(PySequence_GetItem(color, 2))
//                     , PyFloat_AS_DOUBLE(PySequence_GetItem(color, 3))
//                     );
//     Py_RETURN_TRUE;
// }


void
Morphology::set_membrane_voltages(PyObject * vms)
{
    unsigned int i;
    for(i = 0; i < PySequence_Length(vms); ++i)
    {
        _compartment_order[i] -> set_membrane_voltage( PyFloat_AS_DOUBLE(PySequence_GetItem(vms, i))
                                                     , _base_vm
                                                     , _peak_vm
                                                     , _initial_color
                                                     , _final_color
                                                     );
    }
}


void
Morphology::destroy_group(const char * group_id)
{
    _groups.erase(group_id);
}

void
Morphology::modify_group( const char * group_id
                        , PyObject * compartment_ids
                        , double base_value
                        , double peak_value
                        , PyObject * base_color
                        , PyObject * peak_color
                        )
{
    Vec4f base_color_vector( PyFloat_AS_DOUBLE(PySequence_GetItem(base_color, 0))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(base_color, 1))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(base_color, 2))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(base_color, 3))
                           );

    Vec4f peak_color_vector( PyFloat_AS_DOUBLE(PySequence_GetItem(peak_color, 0))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(peak_color, 1))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(peak_color, 2))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(peak_color, 3))
                           );

    unsigned int i;
    unsigned int size = PySequence_Length(compartment_ids);
    vector<Compartment *> compartments(size);

    for(i = 0; i < size; ++i)
    {
        PyObject * object = PySequence_GetItem(compartment_ids, i);
        compartments[i] = _compartments[string(PyString_AsString(object))];
    }


    _groups[group_id] = make_tuple( compartments
                                  , base_value
                                  , peak_value
                                  , base_color_vector
                                  , peak_color_vector
                                  );

}

void
Morphology::create_group( const char * group_id
                     , PyObject * compartment_ids
                     , double base_value
                     , double peak_value
                     , PyObject * base_color
                     , PyObject * peak_color
                     )
{

    group_map_t::iterator result = _groups.find(group_id);

    if(result != _groups.end())
    {
        RECORD_ERROR("Overwriting existing group with group id => " + string(group_id));
        return;
    }


    Vec4f base_color_vector( PyFloat_AS_DOUBLE(PySequence_GetItem(base_color, 0))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(base_color, 1))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(base_color, 2))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(base_color, 3))
                           );

    Vec4f peak_color_vector( PyFloat_AS_DOUBLE(PySequence_GetItem(peak_color, 0))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(peak_color, 1))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(peak_color, 2))
                           , PyFloat_AS_DOUBLE(PySequence_GetItem(peak_color, 3))
                           );
    unsigned int i;
    unsigned int size = PySequence_Length(compartment_ids);
    vector<Compartment *> compartments(size);

    for(i = 0; i < size; ++i)
    {
        PyObject * object = PySequence_GetItem(compartment_ids, i);
        compartments[i] = _compartments[string(PyString_AsString(object))];
    }


    _groups[group_id] = make_tuple( compartments
                                  , base_value
                                  , peak_value
                                  , base_color_vector
                                  , peak_color_vector
                                  );
}

void
Morphology::set_color( const char * group_id
                     , PyObject   * values
                     )
{
    auto compartments   = get<0>(_groups[group_id]);

    if(static_cast<unsigned int>(compartments.size()) != PySequence_Length(values))
    {
        RECORD_ERROR("Number of values not the same as the number of compartments in the group =>" + string(group_id));
        return;
    }
    for(int i = 0; i < PySequence_Length(values); ++i)
    {
        compartments[i] -> set_color( PyFloat_AS_DOUBLE(PySequence_GetItem(values, i))
                                    , get<1>(_groups[group_id])
                                    , get<2>(_groups[group_id])
                                    , get<3>(_groups[group_id])
                                    , get<4>(_groups[group_id])
                                    );
    }
}


Morphology::~Morphology()
{

}

ref_ptr<MatrixTransform>
Morphology::get_scene_graph()
{
    return _matrix_transform;
}
