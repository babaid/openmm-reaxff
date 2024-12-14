#---------------------------------------------------
# OpenMM CPU PME Plugin
#
# Creates OpenMMReaxFF plugin library.
#
# Windows:
#   OpenMMReaxFF.dll
#   OpenMMReaxFF.lib
#   OpenMMReaxFF_static.lib
# Unix:
#   libOpenMMReaxFF.so
#   libOpenMMReaxFF_static.a
#----------------------------------------------------

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(OPENMM_SOURCE_SUBDIRS . openmmapi)

set(OPENMM_BUILD_REAXFF_MPI TRUE CACHE BOOL "Wether to build the ReaxFF plugin with OpenMP enabled")
if (OPENMM_BUILD_REAXFF_MPI)
    #for reaxff_puremd
    set(OPENMP ON)
    find_package(OpenMP REQUIRED)
else()
    set(OPENMP OFF)
endif()
# Collect up information about the version of the OpenMM library we're building
# and make it available to the code so it can be built into the binaries.

SET(OPENMMREAXFF_LIBRARY_NAME OpenMMReaxFF)
ADD_DEFINITIONS(-DOPENMM_REAXFF_LIBRARY_NAME=${OPENMMREAXFF_LIBRARY_NAME})
SET(SHARED_TARGET ${OPENMMREAXFF_LIBRARY_NAME})
SET(STATIC_TARGET ${OPENMMREAXFF_LIBRARY_NAME}_static)


# These are all the places to search for header files which are
# to be part of the API.
SET(API_INCLUDE_DIRS) # start empty
FOREACH(subdir ${OPENMM_SOURCE_SUBDIRS})
    # append
    SET(API_INCLUDE_DIRS ${API_INCLUDE_DIRS}
                         ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include
                         ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include/internal)
ENDFOREACH(subdir)

# Find the include files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)	# returns full pathnames
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
SET(PUREMD_INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/reaxff_puremd/PuReMD/sPuReMD/src/*.c)
SET(PUREMD_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/reaxff_puremd/PuReMD/sPuReMD/src/*.h)
FILE(GLOB puremd_incl ${PUREMD_INCLUDE_FILES})
FILE(GLOB puremd_src ${PUREMD_SOURCE_FILES})
SET(SOURCE_FILES ${SOURCE_FILES} ${puremd_src})
SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${puremd_src})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/reaxff_puremd/PuReMD/sPuReMD/src)

FOREACH(subdir ${OPENMM_SOURCE_SUBDIRS})
    FILE(GLOB_RECURSE src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.c)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)
IF(NOT MSVC)
    IF(X86)
        SET_SOURCE_FILES_PROPERTIES(${SOURCE_FILES} PROPERTIES COMPILE_FLAGS "-msse4.1")
    ELSE()
        SET_SOURCE_FILES_PROPERTIES(${SOURCE_FILES} PROPERTIES COMPILE_FLAGS "")
    ENDIF()
ENDIF()

ADD_SUBDIRECTORY("${CMAKE_CURRENT_SOURCE_DIR}/reaxff_puremd")

# Build the shared plugin library.
IF (OPENMM_BUILD_SHARED_LIB)
    ADD_LIBRARY(${SHARED_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
    if (OPENMM_BUILD_REAXFF_MPI)
        TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME} ${PTHREADS_LIB} libspuremd OpenMP::OpenMP_CXX)
    else()
        TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME} ${PTHREADS_LIB} libspuremd)
    endif()
    SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES  LINKER_LANGUAGE CXX LINK_FLAGS "${EXTRA_LINK_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_REAXFF_BUILDING_SHARED_LIBRARY")
    install(TARGETS ${SHARED_TARGET} LIBRARY DESTINATION lib/plugins)
    install(TARGETS ${SHARED_TARGET} LIBRARY DESTINATION lib)
ENDIF (OPENMM_BUILD_SHARED_LIB)

# Build the static plugin library.
IF(OPENMM_BUILD_STATIC_LIB)
    ADD_LIBRARY(${STATIC_TARGET} STATIC ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
    if (OPENMM_BUILD_REAXFF_MPI)
        TARGET_LINK_LIBRARIES(${STATIC_TARGET} ${OPENMM_LIBRARY_NAME}_static ${PTHREADS_LIB} libspuremd OpenMP::OpenMP_CXX)
    else()
        TARGET_LINK_LIBRARIES(${STATIC_TARGET} ${OPENMM_LIBRARY_NAME}_static ${PTHREADS_LIB} libspuremd)
    endif()

    SET_TARGET_PROPERTIES(${STATIC_TARGET} PROPERTIES  LINKER_LANGUAGE CXX LINK_FLAGS "${EXTRA_LINK_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_REAXFF_BUILDING_STATIC_LIBRARY")
    install(TARGETS ${STATIC_TARGET} LIBRARY DESTINATION lib/plugins)
    install(TARGETS ${STATIC_TARGET} LIBRARY DESTINATION lib)
ENDIF(OPENMM_BUILD_STATIC_LIB)

FILE(GLOB CORE_HEADERS     include/*.h          */include/*.h)
FILE(GLOB TOP_HEADERS      include/openmm/*.h          */include/openmm/*.h)
FILE(GLOB INTERNAL_HEADERS include/openmm/internal/*.h */include/openmm/internal/*.h )
INSTALL_FILES(/include                 FILES ${CORE_HEADERS})
INSTALL_FILES(/include/openmm          FILES ${TOP_HEADERS})
INSTALL_FILES(/include/openmm/internal FILES ${INTERNAL_HEADERS})

#TESTING

IF (EXECUTABLE_OUTPUT_PATH)
  SET (TEST_PATH ${EXECUTABLE_OUTPUT_PATH})
ELSE (EXECUTABLE_OUTPUT_PATH)
  SET (TEST_PATH .)
ENDIF (EXECUTABLE_OUTPUT_PATH)

IF(BUILD_TESTING)
    SUBDIRS(tests)
ENDIF(BUILD_TESTING)