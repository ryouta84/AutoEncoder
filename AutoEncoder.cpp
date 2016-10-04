#include "AutoEncoder.h"

AutoEncoder::AutoEncoder(size_t dSize, size_t hidSize) :
//mDataSize == 入力数 == 出力数
 mDataSize(dSize), mHiddenSize(hidSize),
//中間層の重みの数＝入力数x中間層のセル数, 出力層の重み＝中間層の出力数x出力層のセル数
 weight( new double[dSize*hidSize + hidSize*dSize] ),
 threshold( new double[hidSize+dSize] ), output( new double[hidSize+dSize] )
{
    std::cout << "mDataSize=" << mDataSize << std::endl;
    std::cout << "mHiddenSize" << mHiddenSize << std::endl;
    init();
}
//入力を受け取り、outputに計算結果を格納
double AutoEncoder::forward(vector<double> &input, size_t cellNo)
{
    mInput = input;
try{
    //中間層の計算
    for(size_t i=0; i<mHiddenSize; ++i){
        double hidbuf=0;
        for(size_t j=0; j<mDataSize; ++j){
            hidbuf += mInput.at(j) * weight[i*mDataSize + j];
        }
        hidbuf -= threshold[i];
        output[i] = f(hidbuf);
    }
    //出力層の計算
    size_t oLayerWeightBegin = mDataSize * mHiddenSize + mHiddenSize*cellNo; //出力層のcellNoセルの重みへのインデックス
    double outbuf = 0.0;
    for(size_t i=0; i<mHiddenSize; ++i) {
        outbuf += output[i] * weight[oLayerWeightBegin + i];
    }
    outbuf -= threshold[mHiddenSize+cellNo];
    output[mHiddenSize + cellNo] = f(outbuf);

}catch(out_of_range &e){
    std::cerr << "out_of_range : AutoEncoder::forward()" << e.what() << std::endl;
}
    return output[mHiddenSize+cellNo];
}

//学習するセルの番号を受け取る
void AutoEncoder::learn(size_t cellNo)
{
    //出力層の学習
    outputLayerLearn(cellNo);

    //中間層の学習
    double di =0.0;
    double o = output[mHiddenSize+cellNo];
    double E = mInput.at(cellNo) - o;
    //std::cout << "--------------------weight of hidden layer--------------------" << std::endl;
    for(size_t i=0; i<mHiddenSize; ++i){ //i番目のセル
        di = output[i] * (1-output[i]) * weight[mDataSize*mHiddenSize + mHiddenSize*cellNo + i] * E * o * (1-o);
        for(size_t j=0; j<mDataSize; ++j){ //i番目のセルのj番目の入力
            weight[i*mHiddenSize + j] += alpha * mInput.at(j) * di;
            //cout << weight[i * mHiddenSize + j] << " ";
        }
        threshold[i] += alpha * (-1.0) * di;
    }
    //std::cout << std::endl << "--------------------------------------------------------------" << std::endl;
}

void AutoEncoder::outputLayerLearn(size_t cellNo)
{
    double o = output[mHiddenSize + cellNo];
    double E = mInput.at(cellNo) - o;
    double d = E * o * (1-o);
    //std::cout << "--------------------weight of ouput layer-------------------" << std::endl;
    for (size_t i = 0; i < mHiddenSize; i++) {
        weight[mDataSize*mHiddenSize + mHiddenSize*cellNo + i] += alpha * output[i] * d;
        //cout << "mDataSize*mHiddenSize + mHiddenSize*cellNo + i:" << weight[mDataSize*mHiddenSize + mHiddenSize*cellNo + i] << " ";
    }
    threshold[mHiddenSize + cellNo] += alpha * (-1.0) * d;
    //std::cout << std::endl << "------------------------------------------------------------" << std::endl;
}

double AutoEncoder::f(double u)
{
    //シグモイド関数
    return 1.0 / ( 1.0 + exp(-u) );
}


void AutoEncoder::init()
{
    std::random_device seedGen;
    std::mt19937 engine(seedGen());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    //重みの初期化
    for(size_t i=0; i<mDataSize*mHiddenSize *2; ++i){
        weight[i] = dist(engine);
    }
    for(size_t i=0; i<mHiddenSize+mDataSize; ++i){
        threshold[i] = dist(engine);
    }
}
