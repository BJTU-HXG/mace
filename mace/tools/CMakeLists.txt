file(GLOB MACE_RUN_SRCS
  mace_run.cc
)
add_executable(mace_run ${MACE_RUN_SRCS})
target_link_libraries(mace_run
  mace_static
  model
  extra_link_libs_target
  gflags
)

file(GLOB MACE_TEST_SRCS
  ais_inference_hexagon.cc
  gsd_run.cc
)
add_executable(mace_test ${MACE_TEST_SRCS})
target_link_libraries(mace_test
  mace_static
  model
  extra_link_libs_target
  gflags
)


if(NOT ANDROID AND NOT QNX)
  target_link_libraries(mace_run)
endif()

if(MACE_ENABLE_HEXAGON_DSP)
  target_link_libraries(mace_run hexagon_controller)
endif()

install(TARGETS mace_run RUNTIME DESTINATION bin)

if(NOT ANDROID AND NOT QNX)
  target_link_libraries(mace_test)
endif()

if(MACE_ENABLE_HEXAGON_DSP)
  target_link_libraries(mace_test hexagon_controller)
endif()

install(TARGETS mace_test RUNTIME DESTINATION bin)
