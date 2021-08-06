CXX=g++-10
CXXFLAGS=-g -std=c++20 \
         -Wall -Wextra -Wshadow -Wpedantic -Wconversion \
         -I../../../include/
LDFLAGS=-L/usr/lib/x86_64-linux-gnu/
LIBS=-lGL -lglut

%: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

default:
	@echo "Specify the program to build:"
	@echo ""
	@echo " make cartpole_test"
	@echo " make mountain_car_test"

#clean:
#	rm -f mountain_car_test
