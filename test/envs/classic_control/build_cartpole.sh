#!/bin/sh

g++-10 -g -std=c++20 -I../../../include/ -Wall -Wextra -Wshadow -Wpedantic -Wconversion -L/usr/lib/x86_64-linux-gnu/ -o cartpole_test cartpole_test.cpp -lGL -lglut
