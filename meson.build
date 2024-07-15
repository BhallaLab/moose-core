# Project to build MOOSE's python module.
# Author: Subhasis Ray
# Date: Mon Jul  8 03:11:13 IST 2024

# To build pymoose with meson, run the following
# `meson setup --wipe _build`
# `cd _build`
# `meson compile`

# To build installable wheels
# python -m build

project('pymoose', 'c', 'cpp',
	default_options: ['default_library=static', 'c_std=c11', 'cpp_std=c++14'])

pybind_dep = dependency('pybind11')
gsl_dep = dependency('gsl', version: '>=1.16')

use_mpi = get_option('use_mpi')
if use_mpi
  mpi_dep = dependency('mpi', required: true)
  add_global_arguments('-DUSE_MPI', language: ['c', 'cpp'])
else
  mpi_dep = []
endif  

add_global_arguments('-DMOOSE_VERSION="4.0.1"', language: ['c', 'cpp'])
add_global_arguments('-DUSE_GSL', language: ['c', 'cpp'])

##############################################################
# Credit: Borrowed from SciPy
##############################################################
python = import('python')
py = python.find_installation()
python_dep = py.dependency()

min_numpy_version = '1.23.5'  # keep in sync with pyproject.toml

# Emit a warning for 32-bit Python installs on Windows; users are getting
# unexpected from-source builds there because we no longer provide wheels.
is_windows = host_machine.system() == 'windows'
if is_windows and py.has_variable('EXT_SUFFIX')
  ext_suffix = py.get_variable('EXT_SUFFIX')
  if ext_suffix.contains('win32')
    warning('You are building from source on a 32-bit Python install. pymoose does not provide 32-bit wheels; install 64-bit Python if you are having issues!')
  endif
endif

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
# cy = meson.get_compiler('cython')

# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=9.1')
    error('pymoose requires GCC >= 9.1')
  endif
elif cc.get_id() == 'clang' or cc.get_id() == 'clang-cl'
  if not cc.version().version_compare('>=12.0')
    error('pymoose requires clang >= 12.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('pymoose requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
  # Several sub libs link against the same other sublib and MSVC
  # linker barfs at multiple defs of the same symbol
  add_project_link_arguments('/FORCE:MULTIPLE', '/WHOLEARCHIVE', language: ['c', 'cpp'])  
endif


_global_c_args = cc.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
)
add_project_arguments(_global_c_args, language : 'c')


if host_machine.system() == 'darwin'
  if cc.has_link_argument('-Wl,-ld_classic')
    # New linker introduced in macOS 14 not working yet, see gh-19357 and gh-19387
    add_project_link_arguments('-Wl,-ld_classic', language : ['c', 'cpp'])
  endif
  if cc.has_link_argument('-Wl,-dead_strip')
    # Allow linker to strip unused symbols
    add_project_link_arguments('-Wl,-dead_strip', language : ['c', 'cpp'])
  endif
endif

# Link against conda env GSL
conda_res = run_command('python', '-c', 'import os; print(os.environ["CONDA_PREFIX"])', check: true)
if conda_res.returncode() == 0
  conda_lib_path = conda_res.stdout().strip() / 'Library/lib'
else
  message('Environment variable CONDA_PREFIX not found. Libraries must be available in PATH, LD_LIBRARY_PARH, etc.')
endif

if is_windows
  # Deal with M_PI & friends; add `use_math_defines` to c_args or cpp_args
  add_global_arguments('-D_USE_MATH_DEFINES', language: ['c', 'cpp'])
  # On Windows some of the object files are too big
  add_global_arguments('/bigobj', language: ['c', 'cpp'])
  if conda_res.returncode() == 0
    add_project_link_arguments('/LIBPATH:' + conda_lib_path, language: ['c', 'cpp'])
  endif
endif



compilers = {
  'C': cc,
  'CPP': cpp,
}

machines = {
  'HOST': host_machine,
  'BUILD': build_machine,
}

conf_data = configuration_data()

# Set compiler information
foreach name, compiler : compilers
  conf_data.set(name + '_COMP', compiler.get_id())
  conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
  conf_data.set(name + '_COMP_VERSION', compiler.version())
  conf_data.set(name + '_COMP_CMD_ARRAY', ', '.join(compiler.cmd_array()))
  conf_data.set(name + '_COMP_ARGS', ', '.join(
      get_option(name.to_lower() + '_args')
    )
  )
  conf_data.set(name + '_COMP_LINK_ARGS', ', '.join(
      get_option(name.to_lower() + '_link_args')
    )
  )
endforeach

# Machines CPU and system information
foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

add_global_arguments('-DCOMPILER_STRING="' +
                     compiler.get_id() + ' ' +
                     compiler.version() + '"',
                     language: ['c', 'cpp'])

conf_data.set('CROSS_COMPILED', meson.is_cross_build())

# Python information
conf_data.set('PYTHON_PATH', py.full_path())
conf_data.set('PYTHON_VERSION', py.language_version())


# End Credit: SciPy
libsoda = static_library('lsoda', 'external/libsoda/LSODA.cpp',
                          include_directories: 'external/libsoda',
                          install: false)
libsoda_dep = declare_dependency(include_directories: 'external/libsoda',
                                 link_with: libsoda)


subdir('basecode')
subdir('biophysics')
subdir('builtins')
subdir('device')
subdir('diffusion')
subdir('examples')
subdir('hsolve')
subdir('intfire')
subdir('kinetics')
subdir('ksolve')
subdir('mesh')
subdir('mpi')
subdir('msg')
subdir('pybind11')
subdir('randnum')
subdir('scheduling')
subdir('shell')
subdir('signeur')
subdir('synapse')
subdir('utility')


sublibs = [
    basecode_lib,
    biophysics_lib,
    builtins_lib,
    device_lib,
    diffusion_lib,
    examples_lib,
    hsolve_lib,
    intfire_lib,
    kinetics_lib,
    ksolve_lib,
    mesh_lib,
    mpi_lib,
    msg_lib,
    pybind11_lib,
    randnum_lib,
    scheduling_lib,
    shell_lib,
    signeur_lib,
    synapse_lib,
    utility_lib,
]

moose_dir = py.get_install_dir()

pymoose = py.extension_module('_moose', 'pybind11/pymoose.cpp',
                              link_with: sublibs,
                              dependencies: [gsl_dep, mpi_dep, pybind_dep, libsoda_dep],
                              include_directories: ['external/getopt'],
                              install: true, install_dir: moose_dir / 'moose')

if is_windows
  rename_cmd = 'move'
else
  rename_cmd = 'mv'
endif

# custom_target('mooselib',
#               depends: pymoose,
#               input: pymoose,
#               output: '_moose.so',
#               install_dir: moose_dir / 'moose',
#               command: [rename_cmd, pymoose.full_path(), '@OUTPUT@'],
#               build_by_default: true,
#               install: true
#              )

install_subdir(
  'python/moose',
  install_dir: moose_dir)

install_subdir(
  'python/rdesigneur',
  install_dir: moose_dir)

# run_target('rename_lib',
#            command: [rename_cmd, pymoose.full_path(), moose_dir / '_moose.pyd'],
#            depends: pymoose)

message('Finished installing moose in', pymoose.full_path())
