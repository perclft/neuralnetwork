package neural

import (
	"math"
	"math/rand"
)

// Basic transfer function type
type transferFunc func(float64) float64

// Pattern struct represents one pattern with dimensions and desired value
type Pattern struct {
	Features             []float64
	SingleRawExpectation string
	SingleExpectation    float64
	MultipleExpectation  []float64
}

// NeuronUnit struct represents a simple NeuronUnit network with a slice of n weights.
type NeuronUnit struct {
	Weights []float64
	Bias    float64
	Lrate   float64
	Value   float64
	Delta   float64
}

type NeuralLayer struct {
	NeuronUnits []NeuronUnit
	Length      int
}

type MultiLayerNetwork struct {
	L_rate       float64
	NeuralLayers []NeuralLayer
	T_func       transferFunc
	T_func_d     transferFunc // derivative of transfer function
}

// Common transfer functions
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid
func SigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// RandomNeuronInit initializes a neuron with random weights
func RandomNeuronInit(n *NeuronUnit, prevLayerSize int) {
	n.Weights = make([]float64, prevLayerSize)
	for i := range n.Weights {
		n.Weights[i] = rand.Float64()*2 - 1 // random weights between -1 and 1
	}
	n.Bias = rand.Float64()*2 - 1
}

// FeedForward propagates the input through the network
func (mlp *MultiLayerNetwork) FeedForward(input []float64) []float64 {
	// Set input layer values
	for i := range mlp.NeuralLayers[0].NeuronUnits {
		if i < len(input) {
			mlp.NeuralLayers[0].NeuronUnits[i].Value = input[i]
		}
	}

	// Propagate through hidden and output layers
	for i := 1; i < len(mlp.NeuralLayers); i++ {
		layer := &mlp.NeuralLayers[i]
		prevLayer := &mlp.NeuralLayers[i-1]

		for j := range layer.NeuronUnits {
			neuron := &layer.NeuronUnits[j]
			sum := neuron.Bias

			for k := range prevLayer.NeuronUnits {
				sum += prevLayer.NeuronUnits[k].Value * neuron.Weights[k]
			}

			neuron.Value = mlp.T_func(sum)
		}
	}

	// Get output layer results
	outputLayer := mlp.NeuralLayers[len(mlp.NeuralLayers)-1]
	output := make([]float64, len(outputLayer.NeuronUnits))
	for i := range outputLayer.NeuronUnits {
		output[i] = outputLayer.NeuronUnits[i].Value
	}

	return output
}

// Train performs one training step using backpropagation
func (mlp *MultiLayerNetwork) Train(pattern Pattern) float64 {
	// Feed forward
	output := mlp.FeedForward(pattern.Features)

	// Calculate output layer error
	outputLayer := &mlp.NeuralLayers[len(mlp.NeuralLayers)-1]
	for i := range outputLayer.NeuronUnits {
		neuron := &outputLayer.NeuronUnits[i]
		error := pattern.MultipleExpectation[i] - neuron.Value
		neuron.Delta = error * mlp.T_func_d(neuron.Value)
	}

	// Calculate hidden layers error
	for i := len(mlp.NeuralLayers) - 2; i > 0; i-- {
		layer := &mlp.NeuralLayers[i]
		nextLayer := &mlp.NeuralLayers[i+1]

		for j := range layer.NeuronUnits {
			neuron := &layer.NeuronUnits[j]
			var sum float64

			for k := range nextLayer.NeuronUnits {
				sum += nextLayer.NeuronUnits[k].Delta * nextLayer.NeuronUnits[k].Weights[j]
			}

			neuron.Delta = sum * mlp.T_func_d(neuron.Value)
		}
	}

	// Update weights
	for i := 1; i < len(mlp.NeuralLayers); i++ {
		layer := &mlp.NeuralLayers[i]
		prevLayer := &mlp.NeuralLayers[i-1]

		for j := range layer.NeuronUnits {
			neuron := &layer.NeuronUnits[j]

			for k := range prevLayer.NeuronUnits {
				neuron.Weights[k] += mlp.L_rate * neuron.Delta * prevLayer.NeuronUnits[k].Value
			}
			neuron.Bias += mlp.L_rate * neuron.Delta
		}
	}

	// Calculate total error
	var totalError float64
	for i := range output {
		diff := pattern.MultipleExpectation[i] - output[i]
		totalError += diff * diff
	}
	return totalError / float64(len(output))
}
