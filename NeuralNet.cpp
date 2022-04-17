#include "NeuralNet.hpp"

//create class constructor
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
    this->topology = topology;
    this->learningRate = learningRate;
    for (uint i = 0; i < topology.size();i++) //loops over all values in topology
    {
        //initialize neuron layers
        if (i == topology.size() - 1)
            neuronLayers.push_back(new RowVector(topology[i]));
        else
            neuronLayers.push_back(new RowVector(topology[i] + 1));
    }
}