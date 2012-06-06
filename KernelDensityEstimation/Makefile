OPTINCDIR=`pkg-config --cflags opencv`
OPTLIBDIR=`pkg-config --libs opencv`
CXXFLAGS= -D__linux__ -O3 -fopenmp

fastkde: main.o FastKDEBackground.o
	$(CXX) -o $@ $^ -Wall $(CXXFLAGS) $(OPTINCDIR) $(LIBDIR) $(LIBS) $(OPTLIBDIR)

.cpp.o: FastKDEBackground.h
	$(CXX) $(CXXFLAGS) -c $(INCDIR) $(OPTINCDIR) -o $@ $< -Wall -D__linux__

