// The library of utility functions and type definitions

#include <string>
#include <vector>
#include <cmath>
#include <cstdint>
#include <codecvt>
#include <iostream>

// type define
typedef int64_t i64;

using namespace std;

// the function below converts string to char*
const char* cpy_str(const string& str) {
	// allocate memory, +1 for the null terminator
	char* cstr = new char[str.size() + 1];
	copy(str.begin(), str.end(), cstr);
	cstr[str.size()] = '\0';
	return cstr;
}

// the function below converts vector<string> to char**
const char** cpy_vec(vector<string>& vec, i64& size) {
	const char** out = new const char*[size];
	for (i64 i = 0; i < size; i++) out[i] = cpy_str(vec[i]);
	return out;
}

double elegentPair(double& x, double& y) {
	if (y > x) return pow(y, 2) + x;
	else return pow(x, 2) + x + y;
}

string getNextSubtree(string& subtreeStr) {
	// base case 1, if empty, return empty
	if (subtreeStr == "") return subtreeStr;

	// base case 2, if atom, return itself
	if (subtreeStr.find(" ") == string::npos) return subtreeStr;

	// assuming first element of subtreeStr is a list or an atom
	if (subtreeStr.substr(0, 1) == "(") {
		string nextSubtree = "(";
		i64 unMatched = 1;
		i64 i = 1;
		while (unMatched) {
			string nextToken = subtreeStr.substr(i, 1);
			if (nextToken == "(") unMatched++;
			else if (nextToken == ")") unMatched--;
			nextSubtree.append(nextToken);
			i++;
		}
		return nextSubtree;
	} else return subtreeStr.substr(0, subtreeStr.find(" ")); // first element not list
}

string getHead(string& funcStr) {
	if (funcStr.substr(0, 1) == "(") {
		if (funcStr.substr(1, 1) == "(") {
			string tmp = funcStr.substr(1);
			return getNextSubtree(tmp);
		} else return funcStr.substr(1, funcStr.find(" ") - 1);
	} else return funcStr;
}

vector<string> getRest(string& funcStr) {
	vector<string> rest = {};
	if (funcStr.substr(0, 1) == "(") {
		// get rid of the head
		string headStr = getHead(funcStr);
		string restStr = funcStr.substr(headStr.length() + 2); // 1 for "("; 1 for " "

		// get rid of the tailing ")"
		restStr.pop_back();

		// get the next subtree
		string nextSubtree = getNextSubtree(restStr);
		while (1) {
			// append the next subtree to the string vector
			rest.push_back(nextSubtree);

			// update the rest of the function string
			if (restStr.length() > nextSubtree.length()) restStr = restStr.substr(nextSubtree.length() + 1);
			else break;

			// get the next subtree
			nextSubtree = getNextSubtree(restStr);
		}
		return rest;
	} else return rest;
}

i64 fmlSize(string& fml) {
	// Base Cases
	if (fml.size() == 0) return 0;
	if (fml.substr(0, 1) != "(") return 1;

	// Inductive Case
	vector<string> rest = getRest(fml);

	i64 numNodes = 1;
	for (i64 inputI = 0; inputI < rest.size(); inputI++) numNodes += fmlSize(rest[inputI]);
	return numNodes;
}

