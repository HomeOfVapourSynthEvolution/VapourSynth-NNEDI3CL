project('NNEDI3CL', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++14'],
  meson_version : '>=0.48.0',
  version : '7'
)

add_project_arguments('-ffast-math', '-DNNEDI3_DATADIR="@0@"'.format(join_paths(get_option('prefix'), get_option('datadir'), meson.project_name())), language : 'cpp')

sources = [
  'NNEDI3CL/NNEDI3CL.cpp'
]

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)

opencl_dep = dependency('OpenCL', required : false)
if not opencl_dep.found()
  opencl_dep = meson.get_compiler('cpp').find_library('OpenCL')
endif

boost_dep = dependency('boost', modules : ['filesystem', 'system'])

deps = [vapoursynth_dep, opencl_dep, boost_dep]

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-mfpmath=sse', '-msse2', language : 'cpp')
endif

shared_module('nnedi3cl', sources,
  dependencies : deps,
  install : true,
  install_dir : join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth'),
  gnu_symbol_visibility : 'hidden'
)

install_data('NNEDI3CL/nnedi3_weights.bin')
