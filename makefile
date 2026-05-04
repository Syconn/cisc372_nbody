# FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h compute.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -x cu -c $< -o $@

compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -x cu -c $< -o $@

clean:
	rm -f *.o nbody
