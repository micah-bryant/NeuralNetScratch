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
    void propagateForward(const RowVector& input); //address to a pointer containing a vector of the input data

    //method for backward propagation of the data
    void propagateBackward(const RowVector& output); //address to a pointer containing a vector of the NN output

    //method to calculate errors made by neurons
    void calcErrors(const RowVector& output); //address to a pointer containing a vector of the NN output

    //method to update the weights according to stochastic gradient descent
    void updateWeights();

    //method to train the Neural Network
    void train(std::vector<RowVector*> data); //vector of pointers referring to data

    //initialize destructor
    //~NeuralNetwork();

    //TODO
        //implement smart pointers for these functions instead of normal pointers?
            //use pointers as push_back calls the destructor of class when it is used apparently?
        //change from stochastic GD to batch or mini-batch GD
        //clean up all the dynamically allocated values in destructor

    std::vector<RowVector*> neuronLayers; //stores all the layers of the neural network
    std::vector<RowVector*> cacheLayers; //stores the unactivated values of the layers
    std::vector<RowVector*> deltas; //stores error contribution of each node
    std::vector<Matrix*> weights; //stores the weights for the nodes

    //the following code should replace the code above just need to make sure the shared pointers are used correctly throughout
//    std::vector<std::shared_ptr<RowVector>> neuronLayers; //stores all the layers of the neural network
//    std::vector<std::shared_ptr<RowVector>> cacheLayers; //stores the unactivated values of the layers
//    std::vector<std::shared_ptr<RowVector>> deltas; //stores error contribution of each node
//    std::vector<Matrix*> weights; //stores the weights for the nodes

    std::vector<uint> topology; //stores the design of the neural network
    Scalar learningRate; //learning rate of the gradient descent
    //one worry is if the vector of pointers utilizes smart pointers or not in the RowVector case
};