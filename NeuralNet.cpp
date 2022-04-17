#include "NeuralNet.hpp"

//create class constructor
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
    this->topology = topology; //size of topology = #layers; value at each index of topology = #nodes
    this->learningRate = learningRate;
    for (uint i = 0; i < topology.size();i++) //loops over all values in topology
    {
        //initialize neuron layers
        if (i == topology.size() - 1)
            neuronLayers.push_back(new RowVector(topology[i])); //creates output layer
        else
            neuronLayers.push_back(new RowVector(topology[i] + 1)); //adds bias neuron to each layer except output layer
        
        //initialize cache and delta vectors
        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        //use vector.back to get the last value added to vector
        //coeffRef gives the reference of the value accessed at that value
        //use this because we are using pointers for these values
        if (i != topology.size()-1)
        {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        //initialize weights
        if (i > 0){ //checks that it is not the input layer
            if (i != topology.size()-1){ //checks that it is not the output layer
                weights.push_back(new Matrix(topology[i-1]+1, topology[i]+1)); //creates Matrix that has size input+bias, output+bias
                weights.back()->setRandom();//sets the weights to be randomly initalized
                weights.back()->col(topology[i]).setZero();//sets all values in last column to be zero
                weights.back()->coeffRef(topology[i-1],topology[i]) = 1.0;//ensures bias is still set to 1
            }
            else{
                weights.push_back(new Matrix(topology[i-1]+1,topology[i]));//if output layer do not add bias neuron
                weights.back()->setRandom();
            }
        }
    }
}

void NeuralNetwork::propagateForward(const RowVector& input)
{
    //sets input layer to be the input data
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size()-1) = input;

    //perform forward propagation and pass value to activation function
    for (uint i = 1; i < topology.size(); i++){
        (*neuronLayers[i]) = (*neuronLayers[i-1]) * (*weights[i-1]);
        neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));
    }
}

void NeuralNetwork::propagateBackward(const RowVector& output)
{
    calcErrors(output);
    updateWeights();
}

void NeuralNetwork::calcErrors(const RowVector& output)
{
    //does the calculation for the error of the output layer
    (*deltas.back()) = output - (*neuronLayers.back());

    //finds errors of hidden layers
    for (uint i = topology.size()-2; i > 0; i--){
        (*deltas[i]) = (*deltas[i+1]) * (weights[i]->transpose());
    }
}

void NeuralNetwork::updateWeights()
{
    
}

void NeuralNetwork::train(std::vector<RowVector*> data)
{

}

Scalar activationFunction(const Scalar& x)
{
    return tanhf(x);
}

Scalar activationFunctionDerivative(const Scalar& x)
{
    return 1 - (tanhf(x)*tanhf(x)); //derivative of tanh is 1-tanh^2
}