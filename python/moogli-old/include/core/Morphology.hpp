#ifndef __MORPHOLOGY_HPP__
#define __MORPHOLOGY_HPP__

#include "includes.hpp"
#include "core/Compartment.hpp"
// #include "core/Neuron.hpp"

using namespace std;
using namespace osg;

class Morphology
{
private:

    typedef unordered_map<string, Compartment *> compartment_map_t;
    typedef unordered_map<string, ref_ptr<LOD> > neuron_map_t;
    typedef unordered_map<string, unordered_set<Compartment *> > neuron_compartment_map_t;
    typedef unordered_map<string, unordered_set<string> > ontology_collection_t;
    typedef unordered_map<string, tuple<vector<Compartment *>, double, double, Vec4f, Vec4f> > group_map_t;
    compartment_map_t           _compartments;
    vector<Compartment *>       _compartment_order;
    neuron_map_t                _neurons;
    ontology_collection_t       _compartment_ontology;
    ontology_collection_t       _neuron_ontology;
    neuron_compartment_map_t    _neurons_compartments;
    Vec4f                       _initial_color;
    Vec4f                       _final_color;
    double                      _base_vm;
    double                      _peak_vm;
    unsigned int _lightcount;
    ref_ptr<StateSet> _state_set;
    group_map_t                 _groups;

public:

    ref_ptr<MatrixTransform>    _matrix_transform;

    const unsigned int  lod_resolution;
    const float         lod_distance_delta;
    const unsigned int  min_points;
    const unsigned int  points_delta;
    const string        name;

    Morphology( const string &      name
              , const unsigned int  lod_resolution     = 3
              , const float         lod_distance_delta = 50.0f
              , const unsigned int  min_points         = 8
              , const unsigned int  points_delta       = 2
              );

    Morphology( const char *        name               = ""
              , const unsigned int  lod_resolution     = 3
              , const float         lod_distance_delta = 50.0f
              , const unsigned int  min_points         = 8
              , const unsigned int  points_delta       = 2
              );

    bool
    add_compartment( const string &  compartment_id
                   , const string &  neuron_id
                   , double          proximal_x
                   , double          proximal_y
                   , double          proximal_z
                   , double          proximal_d
                   , double          distal_x
                   , double          distal_y
                   , double          distal_z
                   , double          distal_d
                   );

    int
    add_compartment( const char *    compartment_id
                   , const char *    neuron_id
                   , double          proximal_x
                   , double          proximal_y
                   , double          proximal_z
                   , double          proximal_d
                   , double          distal_x
                   , double          distal_y
                   , double          distal_z
                   , double          distal_d
                   );

    bool
    remove_compartment(const string & compartment_id);

    bool
    remove_neuron(const string & neuron_id);

    bool
    neuron_is_hidden(const string & neuron_id);

    bool
    compartment_is_hidden(const string & compartment_id);

    bool
    hide_compartment(const string & compartment_id);

    bool
    show_compartment(const string & compartment_id);

    bool
    hide_neuron(const string & neuron_id);

    bool
    show_neuron(const string & neuron_id);

    ref_ptr<MatrixTransform>
    get_scene_graph();

    void
    frame();

    PyObject *
    set_compartment_order(PyObject * compartment_order);

    void
    set_initial_color(float r, float g, float b, float a);

    void
    set_final_color(float r, float g, float b, float a);

    void
    set_membrane_voltages(PyObject * vms);

    void
    set_base_membrane_voltage(double base_vm);

    void
    set_peak_membrane_voltage(double peak_vm);

    void
    destroy_group(const char * group_id);

    void
    modify_group( const char *  group_id
                , PyObject *    compartment_ids
                , double        base_value
                , double        peak_value
                , PyObject *    base_color
                , PyObject *    peak_color
                );

    void
    create_group( const char *  group_id
                , PyObject *    compartment_ids
                , double        base_value
                , double        peak_value
                , PyObject *    base_color
                , PyObject *    peak_color
                );

    void
    set_color( const char * group_id
             , PyObject   * values
             );

    // void
    // set_radius( const char * group_id
    //           , PyObject   * values
    //           );

    // void
    // set_height( const char * group_id
    //           , PyObject   * values
    //           );

    // void
    // set_size( const char * group_id
    //         , PyObject   * values
    //         )

    ~Morphology();

private:

    Morphology(const Morphology &);

    void
    _detach_neuron_geometry(ref_ptr<LOD>& neuron_node);

    void
    _attach_neuron_geometry(ref_ptr<LOD>& neuron_node);

    void
    _detach_compartment_geometry(Compartment * compartment);

    void
    _attach_compartment_geometry(Compartment * compartment);

    bool
    _add_neuron(const string &  neuron_id);

    void
    _initialize();

    Material *
    _create_material( const Vec4& diffuse   = MATERIAL_DIFFUSE
                    , const Vec4& specular  = MATERIAL_SPECULAR
                    , const Vec4& emission  = MATERIAL_EMISSION
                    , float shininess       = MATERIAL_SHININESS
                    , float alpha           = MATERIAL_ALPHA
                    );

    void
    _initialize_lights();

    LightSource *
    _create_light_source();

    Light *
    _create_light( const Vec4& position
                 , const Vec3& direction
                 , const Vec4& ambient          = LIGHT_AMBIENT
                 , const Vec4& diffuse          = LIGHT_DIFFUSE
                 , const Vec4& specular         = LIGHT_SPECULAR
                 , float constant_attenuation   = LIGHT_CONSTANT_ATTENUATION
                 , float linear_attenuation     = LIGHT_LINEAR_ATTENUATION
                 , float quadratic_attenuation  = LIGHT_QUADRATIC_ATTENUATION
                 , float spot_exponent          = LIGHT_SPOT_EXPONENT
                 , float spot_cutoff            = LIGHT_SPOT_CUTOFF
                 );

    ShadeModel *
    _create_shade_model();

    void
    _initialize_state_set();

};


#endif  /* __MORPHOLOGY_HPP__ */
