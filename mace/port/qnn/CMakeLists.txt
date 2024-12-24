add_library(port_qnn STATIC
  env.cc
)

target_link_libraries(port_qnn port_posix)