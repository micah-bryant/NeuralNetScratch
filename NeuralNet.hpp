#pragma once

#include <eigen-3.4.0/Eigen/Eigen>
#include <iostream>
#include <vector>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork{
public:
    //initialize constructor
    NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

    //method for forward propagation of the data
    void propagateForward(RowVector& input); //address to a pointer containing a vector of the input data

    //method for backward propagation of the data
    void propagateBackward(RowVector& output); //address to a pointer containing a vector of the NN output

    //method to calculate errors made by neurons
    void calcErrors(RowVector& output); //address to a pointer containing a vector of the NN output

    //method to update the weights according to stochastic gradient descent
    void updateWeights();

    //method to train the Neural Network
    void train(std::vector<RowVector*> data); //vector of pointers referring to data

    //TODO
        //implement smart pointers for these functions instead of normal pointers?

    std::vector<RowVector*> neuronLayers;
    std::vector<RowVector*> cacheLayers;
    std::vector<RowVector*> deltas;
    std::vector<Matrix*> weights;
    std::vector<uint> topology;
    Scalar learningRate; 
};