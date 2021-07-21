NAME=TANG

${NAME}: ${NAME}.cpp
	g++ -I../NN ${NAME}.cpp -o ${NAME} -mavx2 -O3

preprocessing: preprocessing.py
	python preprocessing.py

test: ${NAME} preprocessing
	./${NAME}

.PHONY: clean
clean:
	rm -f ${NAME}