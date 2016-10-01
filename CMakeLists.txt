CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )  # set version
PROJECT( slam ) # set project name

SET(CMAKE_CXX_COMPILER "g++")
set(CMAKE_BUILD_TYPE Debug)
set (CMAKE_CXX_STANDARD 11)


# set the output bin directory
SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# set lib directory
SET( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# set the directory as link
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/bin)

# set head files
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include)

# add child directory
ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src)
