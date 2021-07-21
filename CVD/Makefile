NAME=CVD

${NAME}: ${NAME}.cpp
	g++ -I../NN ${NAME}.cpp -o ${NAME} -mavx2 -O3

preprocessing: preprocessing.py
	mkdir -p Train/Cat
	mkdir -p Train/Dog
	mkdir -p Test/Cat
	mkdir -p Test/Dog
	python preprocessing.py

test: ${NAME} preprocessing
	./${NAME}

.PHONY: clean
clean:
	rm -f ${NAME}