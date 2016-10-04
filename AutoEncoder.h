#pragma once

#include <memory>
#include <random>
#include <cmath>
#include <vector>
#include <iostream>

using namespace std;
/*

*/
class AutoEncoder {
public:
    AutoEncoder(size_t dSize, size_t hidSize);
    double  forward(vector<double> &inpu, size_t cellNo);
    void    learn(size_t cellNo);
    void    init();
    const unique_ptr<double[]> output;
private:
    size_t         mHiddenSize;  //中間層のセルの個数
    size_t         mDataSize;    //一つの訓練データの要素数
    vector<double> mInput;
    double f(double u);
    void   outputLayerLearn(size_t cellNo);
    //AutoEncoderで使われる全ての重み、閾値
    const unique_ptr<double[]> weight;
    const unique_ptr<double[]> threshold;

    double alpha = 10.0; //学習係数
};
