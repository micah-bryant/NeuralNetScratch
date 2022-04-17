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
    }
}