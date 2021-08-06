LIBTORCH_MAIN_DIR=/home/boolean/code/pytorch/libtorch
LIB_DIR=/home/boolean/code/libtmpfs

CXX=g++-10
#CXXFLAGS_OFF=-Wextra -Wshadow -Wpedantic -Wconversion -Wnoexcept
CXXFLAGS=-std=c++20 -fcoroutines \
         -g \
         -Wall \
         -I../../../../include \
         -I${LIBTORCH_MAIN_DIR}/include \
         -I${LIBTORCH_MAIN_DIR}/include/torch/csrc/api/include \
         -D_GLIBCXX_USE_CXX11_ABI=1
#OBJS=test_tensor.o main.o
OBJS=cartpole_dqn.o
LDFLAGS=-L${LIB_DIR}/lib
LIBS=-ltorch_cpu -lc10 -lGL -lglut

APPNAME=cartpole_dqn

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(APPNAME): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	rm -f *.o $(APPNAME)
