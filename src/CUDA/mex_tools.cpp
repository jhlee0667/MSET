#include <iostream>
#include <map>
#include "mex.h"
#include "mset.cuh"

using namespace std;


std::string get_MxString(const mxArray* pm)
{
	if (!mxIsChar(pm)){ 
        mexErrMsgTxt("ERROR:: The format of running mode: string (X)");}
    
    char* str = mxArrayToString(pm);
    std::string output = std::string(str);
	return output;

	return "";
}


void MxStruct2vector(const mxArray* ma, std::map<std::string, const mxArray*> & output)
{
    output.clear();
    if (!mxIsStruct(ma)) {
        mexErrMsgTxt("ERROR:: The format of input data: structure (X)");
    }
    int num_members = mxGetNumberOfFields(ma);
    for (int i = 0; i < num_members; i++)
    {
        const char* str = mxGetFieldNameByNumber(ma, i);
        std::string name = std::string(str);
        const mxArray* member = mxGetFieldByNumber(ma, 0, i);
        if (member == NULL) {
            mexErrMsgTxt("ERROR:: The input data: empty");
        }
        output[name] = member;
    }
}



