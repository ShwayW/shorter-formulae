// This is a C++ implementation of the standard polish notation calculator

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <cassert>

using namespace std;
using i64 = int64_t;

enum class Op : uint8_t {
    V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, // variables
    CONST, PI, // constants
    INC, DEC, NEG, SQR, SQRT, EXP, LN, SIN, COS, TAN, ABS, ASIN, ACOS, ATAN, INV, TANH, // unaries
    POW2, POW3,                                                                           // power unaries
    ADD, SUB, MUL, DIV, POW,                                                         // binaries
};

struct Instr {
    Op op;
    double imm; // only used when op==CONST
};

// ultra-light tokenizer (split by ASCII space)
static inline vector<string> split_spaces(const string& s) {
    vector<string> out;
    out.reserve(256);
    const char* p = s.c_str();
    const char* end = p + s.size();
    while (p < end) {
        while (p < end && *p == ' ') ++p;
        const char* q = p;
        while (q < end && *q != ' ') ++q;
        if (q > p) out.emplace_back(p, q - p);
        p = q;
    }
    return out;
}

static inline bool eq(const string& a, const char* b) {
    return a.size() == strlen(b) && memcmp(a.data(), b, a.size()) == 0;
}

static vector<Instr> parse_program(const string& formula) {
    auto toks = split_spaces(formula);
    vector<Instr> prog;
    prog.reserve(toks.size());
    for (auto it = toks.rbegin(); it != toks.rend(); ++it) {
        const string& op = *it;
        Instr ins{};
        if (eq(op,"v1")) ins.op = Op::V1;
        else if (eq(op,"v2")) ins.op = Op::V2;
        else if (eq(op,"v3")) ins.op = Op::V3;
        else if (eq(op,"v4")) ins.op = Op::V4;
        else if (eq(op,"v5")) ins.op = Op::V5;
        else if (eq(op,"v6")) ins.op = Op::V6;
        else if (eq(op,"v7")) ins.op = Op::V7;
        else if (eq(op,"v8")) ins.op = Op::V8;
        else if (eq(op,"v9")) ins.op = Op::V9;
        else if (eq(op,"v10")) ins.op = Op::V10;
        else if (eq(op,"++")) ins.op = Op::INC;
        else if (eq(op,"--")) ins.op = Op::DEC;
        else if (eq(op,"neg")) ins.op = Op::NEG;
        else if (eq(op,"sqr")) ins.op = Op::SQR;
        else if (eq(op,"sqrt")) ins.op = Op::SQRT;
        else if (eq(op,"exp")) ins.op = Op::EXP;
        else if (eq(op,"ln")) ins.op = Op::LN;
        else if (eq(op,"sin")) ins.op = Op::SIN;
        else if (eq(op,"cos")) ins.op = Op::COS;
        else if (eq(op,"tan")) ins.op = Op::TAN;
        else if (eq(op,"abs")) ins.op = Op::ABS;
        else if (eq(op,"arcsin")) ins.op = Op::ASIN;
        else if (eq(op,"arccos")) ins.op = Op::ACOS;
        else if (eq(op,"arctan")) ins.op = Op::ATAN;
        else if (eq(op,"invert")) ins.op = Op::INV;
        else if (eq(op,"tanh"))   ins.op = Op::TANH;
        else if (eq(op,"pow2"))   ins.op = Op::POW2;
        else if (eq(op,"pow3"))   ins.op = Op::POW3;
        else if (eq(op,"pi")) ins.op = Op::PI;
		else if (eq(op,"+")) ins.op = Op::ADD;
        else if (eq(op,"-")) ins.op = Op::SUB;
        else if (eq(op,"*")) ins.op = Op::MUL;
        else if (eq(op,"/"))   ins.op = Op::DIV;
        else if (eq(op,"pow")) ins.op = Op::POW;
        else {
            ins.op = Op::CONST;
            ins.imm = strtod(op.c_str(), nullptr); // faster than stod
        }
        prog.push_back(ins);
    }
    return prog;
}

static inline double max_n(const double* v, int n) {
    double m = v[0];
    for (int i = 1; i < n; ++i) if (v[i] > m) m = v[i];
    return m;
}
static inline double min_n(const double* v, int n) {
    double m = v[0];
    for (int i = 1; i < n; ++i) if (v[i] < m) m = v[i];
    return m;
}
static inline double argmax_n(const double* v, int n) {
    int idx = 0; double m = v[0];
    for (int i = 1; i < n; ++i) if (v[i] > m) { m = v[i]; idx = i; }
    return double(idx);
}
static inline double argmin_n(const double* v, int n) {
    int idx = 0; double m = v[0];
    for (int i = 1; i < n; ++i) if (v[i] < m) { m = v[i]; idx = i; }
    return double(idx);
}

static inline void eval_program(const vector<Instr>& prog, const double* in, double& out, double& stackleft) {
    constexpr double PINF = numeric_limits<double>::infinity();

    // big enough for typical formulas; fallback to dynamic if you ever need more.
    double s_fast[256];
    double* s = s_fast;
    int sp = 0;

    for (const Instr& ins : prog) {
        switch (ins.op) {
            case Op::V1: s[sp++] = in[0]; break;
            case Op::V2: s[sp++] = in[1]; break;
            case Op::V3: s[sp++] = in[2]; break;
            case Op::V4: s[sp++] = in[3]; break;
            case Op::V5: s[sp++] = in[4]; break;
            case Op::V6: s[sp++] = in[5]; break;
            case Op::V7: s[sp++] = in[6]; break;
            case Op::V8: s[sp++] = in[7]; break;
            case Op::V9: s[sp++] = in[8]; break;
            case Op::V10: s[sp++] = in[9]; break;

            case Op::INC: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = a+1.0; } break;
            case Op::DEC: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = a-1.0; } break;
            case Op::NEG: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = -a; } break;
            case Op::SQR: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = a*a; } break;
            case Op::SQRT: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = sqrt(a); } break;
            case Op::EXP: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = exp(a); } break;
            case Op::LN: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = (!isfinite(a) || a<=0.0) ? PINF : log(a); } break;
            case Op::SIN: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = sin(a); } break;
            case Op::COS: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = cos(a); } break;
            case Op::TAN: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = tan(a); } break;
            case Op::ABS: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = fabs(a); } break;
            case Op::ASIN: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = asin(a); } break;
            case Op::ACOS: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = acos(a); } break;
            case Op::ATAN: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = atan(a); } break;
            case Op::INV:  if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = 1.0/a; } break;
            case Op::TANH: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = tanh(a); } break;
            case Op::POW2: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = a*a; } break;
            case Op::POW3: if (sp < 1) { s[sp++] = PINF; break; } { double a=s[--sp]; s[sp++] = a*a*a; } break;

            case Op::ADD: if (sp < 2) { s[sp++] = PINF; break; } { double a=s[--sp], b=s[--sp]; s[sp++] = (isfinite(a)&&isfinite(b)) ? a+b : PINF; } break;
            case Op::SUB: if (sp < 2) { s[sp++] = PINF; break; } { double a=s[--sp], b=s[--sp]; s[sp++] = (isfinite(a)&&isfinite(b)) ? a-b : PINF; } break;
            case Op::MUL: if (sp < 2) { s[sp++] = PINF; break; } { double a=s[--sp], b=s[--sp]; s[sp++] = (isfinite(a)&&isfinite(b)) ? a*b : PINF; } break;
            case Op::DIV: if (sp < 2) { s[sp++] = PINF; break; }
			  { double a=s[--sp], b=s[--sp];
                                      if (!isfinite(a) || !isfinite(b) || b == 0.0) {
                                          s[sp++] = PINF;
				      } else {
                                          double v = a / b;
                                          s[sp++] = !isfinite(v) ? PINF : v;
				      }
				  } break;
            case Op::POW: if (sp < 2) { s[sp++] = PINF; break; }
                          { double a=s[--sp], b=s[--sp];
                            double v = pow(a, b);
                            s[sp++] = isfinite(v) ? v : PINF;
                          } break;

            case Op::PI: s[sp++] = 3.1415; break;
            case Op::CONST: s[sp++] = ins.imm; break;
        }
    }
    out = s[--sp];
    stackleft = double(sp);
}

// public ABIs
extern "C" void free_mem(const char* funcStr, const double** inputs, const i64 size, double* outputs, double* stacklefts) {
	//delete[] funcStr;
	//for (i64 i = 0; i < size; i++) delete[] inputs[i];
	//delete[] inputs;
	free(outputs);
	free(stacklefts);
}

extern "C" {
	void extEvalPN(const char*& funcStr, const double** inputs, const i64& size, double*& outputs, double*& stacklefts) {
		const string formula(funcStr);

		// parse ONCE for all rows
		vector<Instr> prog = parse_program(formula);

		outputs = (double*)malloc(size * sizeof(double));
		stacklefts = (double*)malloc(size * sizeof(double));

		#pragma omp parallel for
		for (i64 i = 0; i < size; ++i) {
			double out=0.0, left=0.0;
			eval_program(prog, inputs[i], out, left);
			outputs[i] = out;
			stacklefts[i] = left;
		}
	}
}

