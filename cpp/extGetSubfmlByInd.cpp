// This C++ script takes an fml and an index and return its subfml corresponding to the index

#include "utils.h"
#include <iostream>
#include <typeinfo>
#include <cassert>

using namespace std;

string getSubfmlByInd(string& fml, i64& index) {
	// Base Case
	if (index == 0) return fml;
	
	// Inductive Case
	index--;
	vector<string> rest = getRest(fml);
	string subfml;
	for (i64 i = 0; i < rest.size(); i++) {
		string input = rest[i];
		subfml = getSubfmlByInd(input, index);
		if (subfml.size() != 0) break;
		index -= fmlSize(subfml);
	}
	return subfml;
}

extern "C" const char* extGetSubfmlByInd(const char*& fmlRaw, i64& index) {
	string fml = fmlRaw;
	string subfml = getSubfmlByInd(fml, index);
	const char* res = cpy_str(subfml);
	return res;
}

