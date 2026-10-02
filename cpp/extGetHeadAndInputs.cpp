// This C++ script takes an array of char that represents an fml in lisp-like notation and outputs its first and rest elements

#include "utils.h"

using namespace std;

extern "C" void free_mem(const char* head, const char** inputs, i64 size) {
	delete[] head;
	for (i64 i = 0; i < size; i++) delete[] inputs[i];
	delete[] inputs;
}

extern "C" void extGetHeadAndInputs(const char*& funcStrRaw, const char*& headOut, const char**& inputsOut, i64& sizeOut) {
	// implicit convert from char* to string
	string funcStr = funcStrRaw;

	// get the head and the inputs
	string head = getHead(funcStr);
	vector<string> inputs = getRest(funcStr);
	i64 size = inputs.size();

	// convert back to char* and char** and assign to references
	headOut = cpy_str(head);
	inputsOut = cpy_vec(inputs, size);
	sizeOut = size;
}

