#ifndef __COMPARTMENT_HPP__
#define __COMPARTMENT_HPP__

#include "includes.hpp"
#include "globals.hpp"

using namespace osg;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
/// @class      Compartment
/// @brief      Creates a new compartment.
/// @details    Each compartment object stores all information about itself such as its geometry, colors etc. Its methods allow getting/setting geometry information, triggering events and changing colors.
/// @author     Aviral Goel <aviralg@ncbs.res.in>
////////////////////////////////////////////////////////////////////////////////

class Compartment
{
public:

////////////////////////////////////////////////////////////////////////////////
/// Compartment identifier. This should be unique for each compartment.
////////////////////////////////////////////////////////////////////////////////
    const string & id;

////////////////////////////////////////////////////////////////////////////////
/// Identifier for the neuron containing the compartment. This should be unique for each neuron.
////////////////////////////////////////////////////////////////////////////////
    const string & neuron_id;

////////////////////////////////////////////////////////////////////////////////
/// Vector for storing the geometries of the compartment at different resolutions. This is needed for the level of detail implementation. Each compartment needs to store a reference to its geometries as it has to change the geometry color when an event is triggered.
////////////////////////////////////////////////////////////////////////////////
    vector<ref_ptr<Geometry> > geometries;


    ontology_collection_t ontology_collection;

////////////////////////////////////////////////////////////////////////////////
/// @brief      Creates a compartment with the given id and neuron id.
/// @param[in]  id Compartment identifier.
/// @param[in]  neuron Identifier of the neuron containing the compartment
/// @author     Aviral Goel <aviralg@ncbs.res.in>
////////////////////////////////////////////////////////////////////////////////
    Compartment( const string & id
               , const string & neuron_id
               );

////////////////////////////////////////////////////////////////////////////////
/// @brief      Sets the proximal parameters of the compartment.
/// @param[in]  center Proximal x, y, z.
/// @param[in]  d Proximal diameter.
/// @author     Aviral Goel <aviralg@ncbs.res.in>
////////////////////////////////////////////////////////////////////////////////
    void
    set_proximal_parameters( const Vec3d & center
                           , double d
                           );

////////////////////////////////////////////////////////////////////////////////
/// @brief      Sets the proximal parameters of the compartment.
/// @param[in]  x Proximal x.
/// @param[in]  y Proximal y.
/// @param[in]  z Proximal z.
/// @param[in]  d Proximal diameter.
/// @author     Aviral Goel <aviralg@ncbs.res.in>
////////////////////////////////////////////////////////////////////////////////
    void
    set_proximal_parameters( double x
                           , double y
                           , double z
                           , double d
                           );

////////////////////////////////////////////////////////////////////////////////
/// @brief      Returns the proximal parameters of the compartment.
/// @return     Pair of proximal coordinates and proximal diameter.
/// @author     Aviral Goel <aviralg@ncbs.res.in>
////////////////////////////////////////////////////////////////////////////////
    pair<const Vec3d &, double>
    get_proximal_parameters();

////////////////////////////////////////////////////////////////////////////////
/// @brief      Sets the distal parameters of the compartment.
/// @param[in]  center Distal x, y, z.
/// @param[in]  d Distal diameter.
/// @author     Aviral Goel <aviralg@ncbs.res.in>
////////////////////////////////////////////////////////////////////////////////
    void
    set_distal_parameters( const Vec3d & center
                         , double d
                         );

////////////////////////////////////////////////////////////////////////////////
/// @brief      Sets the distal parameters of the compartment.
/// @param[in]  x Distal x.
/// @param[in]  y Distal y.
/// @param[in]  z Distal z.
/// @param[in]  d Distal diameter.
/// @author     Aviral Goel <aviralg@ncbs.res.in>
////////////////////////////////////////////////////////////////////////////////
    void
    set_distal_parameters( double x
                         , double y
                         , double z
                         , double d
                         );

////////////////////////////////////////////////////////////////////////////////
/// @brief      Returns the distal parameters of the compartment.
/// @return     Pair of distal coordinates and distal diameter.
/// @author     Aviral Goel <aviralg@ncbs.res.in>
////////////////////////////////////////////////////////////////////////////////
    pair<const Vec3d &, double>
    get_distal_parameters();


////////////////////////////////////////////////////////////////////////////////
/// @brief      Creates graphical representation of the model.
/// @details    This function fills the geometries vector with geometries of increasing quality. Each geometry in the \ref geometries vector has better quality than the previous one.
/// @note       This function should be called only once.
/// @param[in]  lod_resolution Number of geometries to be created.
/// @param[in]  min_points Number of points for lowest quality geometry.
/// @author     Aviral Goel <aviralg@ncbs.res.in>
////////////////////////////////////////////////////////////////////////////////
    void
    create_geometry( unsigned int lod_resolution
                   , float lod_distance_delta
                   , unsigned int min_points
                   , unsigned int points_delta
                   , StateSet * state_set
                   );
/*
    trigger_digital_event();

    trigger_analog_event(double analog_value);

    set_digital_event_parameters(double parameter_increment);

    set_analog_event_parameters(double parameter_normalization_factor);
*/

    void
    set_color( double value
             , double base_value
             , double peak_value
             , Vec4f& base_color
             , Vec4f& peak_color
             );


    void
    set_membrane_voltage( double vm
                        , double peak_vm
                        , double base_vm
                        , Vec4f& initial_color
                        , Vec4f& final_color
                        );

private:

    void
    _set_chromostat( double vm
                   , double base_vm
                   , double peak_vm
                   );

////////////////////////////////////////////////////////////////////////////////
/// Controls the color of the compartment. All event triggering functions modify this variable and another function maps this to the compartment's color.
////////////////////////////////////////////////////////////////////////////////
    double _chromostat;

////////////////////////////////////////////////////////////////////////////////
/// Compartment's proximal vector.
////////////////////////////////////////////////////////////////////////////////
    Vec3f _proximal;

////////////////////////////////////////////////////////////////////////////////
/// Compartment's distal vector.
////////////////////////////////////////////////////////////////////////////////
    Vec3f _distal;

////////////////////////////////////////////////////////////////////////////////
/// Compartment's proximal diameter.
////////////////////////////////////////////////////////////////////////////////
    float _proximal_d;

////////////////////////////////////////////////////////////////////////////////
/// Compartment's distal diameter.
////////////////////////////////////////////////////////////////////////////////
    float _distal_d;

////////////////////////////////////////////////////////////////////////////////
/// Compartment's center vector.
/// \f[
///    \mathbf{\_center} = \frac{\mathbf{\_proximal} + \mathbf{\_distal}}{2}
/// \f]
////////////////////////////////////////////////////////////////////////////////
    Vec3f _center;

////////////////////////////////////////////////////////////////////////////////
/// Compartment's direction vector.
/// \f[
///    \mathbf{\_direction} = \mathbf{\_distal} - \mathbf{\_proximal}
/// \f]
////////////////////////////////////////////////////////////////////////////////
    Vec3f _direction;

////////////////////////////////////////////////////////////////////////////////
/// Compartment's color.
////////////////////////////////////////////////////////////////////////////////
    Vec4f _color;

////////////////////////////////////////////////////////////////////////////////
/// Compartment's radius.
/// \f[
///    \_radius = \frac{\_distal\_d + \_proximal\_d}{4}
/// \f]
////////////////////////////////////////////////////////////////////////////////
    double _radius;

////////////////////////////////////////////////////////////////////////////////
/// Compartment's height.
/// \f[
///    \_height = \| \mathbf{\_direction} \|
/// \f]
////////////////////////////////////////////////////////////////////////////////
    double _height;
};

#endif /* __COMPARTMENT_HPP__ */
