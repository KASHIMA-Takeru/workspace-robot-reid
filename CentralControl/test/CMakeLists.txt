add_test(NAME ${PROJECT_NAME}_test COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Test.py -o "manager.components.precreate:${PROJECT_NAME}" -o "manager.components.preconnect:${PROJECT_NAME}0.image_data?port=${PROJECT_NAME}Test0.image_data,${PROJECT_NAME}0.depth_data?port=${PROJECT_NAME}Test0.depth_data,${PROJECT_NAME}0.motion_instruction?port=${PROJECT_NAME}Test0.motion_instruction" -o "manager.components.preactivation:${PROJECT_NAME}0,${PROJECT_NAME}Test0" WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
string(REPLACE ";" "\\;" PYTHONPATH "$ENV{PYTHONPATH}")
if(UNIX)
  set_tests_properties(${PROJECT_NAME}_test PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}:${PYTHONPATH}")
else()
  set_tests_properties(${PROJECT_NAME}_test PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}\;${PYTHONPATH}")
endif()
