ray: ray.c
	gcc -g -Wall -std=gnu99 -o ray ray.c -lm

clean:
	rm -f ray
	rm -f *.png