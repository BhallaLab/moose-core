#include "constants.hpp"

const Vec4 MATERIAL_EMISSION        = Vec4(1.f, 0.f, 0.f, 1.f );
const Vec4 MATERIAL_DIFFUSE         = Vec4( 1.0f, 1.0f, 1.0f, 1.0f );
const Vec4 MATERIAL_SPECULAR        = Vec4( 1.0f, 1.0f, 1.0f, 1.0f );
const float MATERIAL_SHININESS      = 1.0f; // 0.0 to 128.0
const float MATERIAL_ALPHA          = 1.0f;  // 0.0 to 1.0

const Vec4 LIGHT_AMBIENT            = Vec4( 1.0f, 1.0f, 1.0f, 1.0f );
const Vec4 LIGHT_DIFFUSE            = Vec4( 0.0f, 0.0f, 0.0f, 1.0f );
const Vec4 LIGHT_SPECULAR           = Vec4( 0.0f, 0.0f, 0.0f, 1.0f );

const float LIGHT_CONSTANT_ATTENUATION    = 0.5f;
const float LIGHT_LINEAR_ATTENUATION      = 1.0f;
const float LIGHT_QUADRATIC_ATTENUATION   = 2.0f;
const float LIGHT_SPOT_EXPONENT           = 100.0f;
const float LIGHT_SPOT_CUTOFF             = 25.0f;

const Vec4 COMPARTMENT_COLOR              = Vec4(1.0f, 0.0f, 0.0f, 1.0f);
