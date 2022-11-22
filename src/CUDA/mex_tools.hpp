#ifndef MEX_TOOLS_HPP
#define MEX_TOOLS_HPP

#include <iostream>
#include <vector>
#include <string>
#include <map>

// This function (getting matlab data) is implemented based on 'https://github.com/BMOLKAIST/Inverse_solver/blob/main/Codes/CUDA_CODE/main.cpp'.

std::string get_MxString(const mxArray* pm);
void MxStruct2vector(const mxArray* ma, std::map<std::string, const mxArray*> & output);
//void get_Struct_arraysizeInfo(const std::map<std::string,const mxArray*>& input,const std::string index, std::array<size_t,3> &output);

template <typename T0, std::size_t N0> void get_Struct_arraysizeInfo(const std::map<std::string,const mxArray*>& input, const std::string key, std::array<T0,N0> &output){
    auto map_kv = input.find(key);

    size_t n_dims = mxGetNumberOfDimensions(map_kv->second);
    const unsigned long * tmp_dims = mxGetDimensions(map_kv->second);

    for (size_t i=0; i< (size_t) n_dims; ++i){
        output[i]= (T0) tmp_dims[n_dims-i-1];
    }
}

template <typename T> void get_Mxdata(const std::map<std::string,const mxArray*>& input, const std::string key, std::vector<T>& array){
    auto map_kv = input.find(key);
    mxClassID mx_classid = mxGetClassID(map_kv->second);

    size_t ellement_num = mxGetNumberOfElements(map_kv->second);
    
    array=std::vector<T>((size_t)ellement_num);

    mxClassID classid = mxGetClassID(map_kv->second);
    switch (classid)
    {
    case mxUINT8_CLASS:
    {
        auto ptr = ((unsigned char*)mxGetUint8s(map_kv->second));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i]=ptr[i];
        }
        break;
    }
    case mxDOUBLE_CLASS:
    {
        auto ptr = ((double*)mxGetDoubles(map_kv->second));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxSINGLE_CLASS:
    {
        auto ptr = ((float*)mxGetSingles(map_kv->second));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxINT32_CLASS:
    {
        auto ptr = ((int32_t*)mxGetInt32s(map_kv->second));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxINT16_CLASS:
    {
        auto ptr = ((int16_t*)mxGetData(map_kv->second));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxUINT32_CLASS:
    {
        auto ptr = ((uint32_t*)mxGetData(map_kv->second));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxUINT16_CLASS:
    {
        auto ptr = ((uint16_t*)mxGetData(map_kv->second));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    case mxLOGICAL_CLASS:
    {
        auto ptr = (mxGetLogicals(map_kv->second));
        for (size_t i = 0; i < ellement_num; i++)
        {
            array[i] = ptr[i];
        }
        break;
    }
    default:
        mexErrMsgTxt("Not a supported mxClassID");
    }
}


void get_Mxdata(const std::map<std::string,const mxArray*>& input, const std::string key, float* output, size_t size){
    auto map_kv = input.find(key);
    size_t n_array = mxGetNumberOfElements(map_kv->second);
    mxClassID mx_classid = mxGetClassID(map_kv->second);

    if (n_array != size) {
        mexErrMsgTxt("Size of input array does not equal the input size");
    }

    switch (mx_classid)
    {
    case mxSINGLE_CLASS:
    {   
            //float* tmp_array;
            auto tmp_array = (float*)  mxGetSingles(map_kv->second);
            for (size_t i = 0; i < size; ++i)
            {
                output[i]=  tmp_array[i];
            }
            break;
    }
    default:
        mexErrMsgTxt("ERROR::Check array type");
    }
}

void get_Mxdata(const std::map<std::string,const mxArray*>& input, const std::string key, float2* output, size_t size){
    auto map_kv = input.find(key);
    size_t n_array = mxGetNumberOfElements(map_kv->second);
    mxClassID mx_classid = mxGetClassID(map_kv->second);

    if (n_array != size) {
        mexErrMsgTxt("Size of input array does not equal the input size");
    }

    switch (mx_classid)
    {
    case mxSINGLE_CLASS:
    {   
            //float* tmp_array;
            auto tmp_array = (float2*)  mxGetComplexSingles(map_kv->second);
            for (size_t i = 0; i < size; ++i)
            {
                output[i]= tmp_array[i];
            }
            break;
    }
    default:
        mexErrMsgTxt("ERROR::Check array type");
    }
}


#endif