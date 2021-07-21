NAME=MNIST

${NAME}: ${NAME}.cpp
	g++ -I../NN ${NAME}.cpp -o ${NAME} -mavx2 -O3

test: ${NAME}
	./${NAME}

.PHONY: clean
clean:
	rm -f ${NAME}