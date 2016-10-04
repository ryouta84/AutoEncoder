OBJS	= main.o AutoEncoder.o
PROGRAM = main

all:			$(PROGRAM)

$(PROGRAM):		$(OBJS)
				g++  $(OBJS) -o $(PROGRAM)

main.o:			main.cpp
				g++ -std=c++14 -c main.cpp


AutoEncoder.o: 	AutoEncoder.cpp
				g++ -std=c++14 -c AutoEncoder.cpp
