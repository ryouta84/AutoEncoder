#include "AutoEncoder.h"
#include <iostream>

using namespace std;

int main()
{
    AutoEncoder ae(9,3);

    vector< vector<double> > data;
    data.push_back(vector<double>{0,0,1,0,0,1,0,0,1} );
    data.push_back(vector<double>{0,1,0,0,1,0,0,1,0} );
    data.push_back(vector<double>{1,0,0,1,0,0,1,0,0} );
    data.push_back(vector<double>{0,0,0,0,0,0,0,0,0} );
    data.push_back(vector<double>{0,0,0,1,1,1,0,0,0} );
    data.push_back(vector<double>{1,1,1,0,0,0,0,0,0} );

    double error = 100;
    size_t count=0;
    cout << fixed;
    while(error > 0.0001 && count < 400000){
        for(size_t i = 0; i<9; ++i){//出力層のセルiに対して
            error = 0.0;
            double o=0;
            for(size_t j=0; j<data.size(); ++j){//学習データを適用する
                o = ae.forward(data.at(j),i);
                ae.learn(i);
                error +=(o - data.at(j).at(i)) * (o - data.at(j).at(i));
            }
            count++;
            std::cout << count << ":error == "<< error << std::endl;
        }
        std::cout << std::endl;
    }

    double buf[6][9];
    for(size_t i = 0; i<9; i++){
        for(size_t j=0; j<data.size(); ++j){
            double o = ae.forward(data.at(j),i);
            buf[j][i] = o;
        }
    }

    //見やすくするために整数に直して表示
    std::cout << "------------test------------" << std::endl;
    for(auto &i : buf){
        for(auto j : i){
            std::cout << (int)(j+0.2) << " ";
        }
        cout << endl;
    }
    cout << "-----------------------------------" << endl;


    return 0;
}
