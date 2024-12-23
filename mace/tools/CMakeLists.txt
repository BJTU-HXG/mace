file(GLOB MACE_RUN_SRCS
  mace_run.cc
)

if (QNN)
    link_directories(${PROJECT_SOURCE_DIR}/third_party/nnlib/qnn/)
endif()
add_executable(mace_run ${MACE_RUN_SRCS})
target_link_libraries(mace_run
  mace_static
  model
  extra_link_libs_target
  gflags
)

if(NOT ANDROID AND NOT QNX)
  target_link_libraries(mace_run)
endif()

if(MACE_ENABLE_HEXAGON_DSP)
  if (QNN)
      target_link_libraries(mace_run hexagon_controller cdsprpc libstd)
  else()
    target_link_libraries(mace_run hexagon_controller)
  endif()
endif()

install(TARGETS mace_run RUNTIME DESTINATION bin)
