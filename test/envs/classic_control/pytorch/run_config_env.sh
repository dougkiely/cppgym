#!/bin/bash

LIBTORCH_LIB_DIR=/home/boolean/code/pytorch/libtorch/lib
OPENGL_LIB_DIR=/usr/lib/x86_64-linux-gnu
LIB_TMPFS_DIR=/home/boolean/code/libtmpfs
LIB_DIR=$LIB_TMPFS_DIR/lib

echo "Creating $LIB_TMPFS_DIR directory ..."
mkdir -p $LIB_TMPFS_DIR
echo "Creating $LIB_TMPFS_DIR directory ... complete"

echo "Mounting $LIB_TMPFS_DIR tmpfs ..."
sudo mount -t tmpfs -o size=350M libtmpfs $LIB_TMPFS_DIR
echo "Mounting $LIB_TMPFS_DIR tmpfs ... complete"

sleep 1

echo "Creating $LIB_DIR directory ..."
mkdir -p $LIB_DIR
echo "Creating $LIB_DIR directory ... complete"

echo "Copying files to $LIB_DIR tmpfs directory ..."
cp -p $LIBTORCH_LIB_DIR/{libtorch_cpu.so,libc10.so,libgomp-75eea7e8.so.1} $LIB_DIR
cp -p $OPENGL_LIB_DIR/{libGL.so,libglut.so} $LIB_DIR
echo "Copying files to $LIB_DIR tmpfs directory ... complete"

echo "Exporting LD_LIBRARY_PATH=$LIB_DIR ..."
export LD_LIBRARY_PATH=$LIB_DIR
echo "Exporting LD_LIBRARY_PATH=$LIB_DIR ... complete"
