// main.go
package main

import (
	"fmt"

	"neuralnetwork/neural"
)

func main() {
	// Create a network with layers [2, 3, 1] (2 input, 3 hidden, 1 output)
	layers := []int{2, 3, 1}
	learningRate := 0.1
	network := neural.PrepareMLPNet(layers, learningRate, neural.Sigmoid, neural.SigmoidDerivative)

	// XOR training data
	patterns := []neural.Pattern{
		{
			Features:            []float64{0, 0},
			MultipleExpectation: []float64{0},
		},
		{
			Features:            []float64{0, 1},
			MultipleExpectation: []float64{1},
		},
		{
			Features:            []float64{1, 0},
			MultipleExpectation: []float64{1},
		},
		{
			Features:            []float64{1, 1},
			MultipleExpectation: []float64{0},
		},
	}

	// Train the network
	epochs := 10000
	for i := 0; i < epochs; i++ {
		var totalError float64
		for _, pattern := range patterns {
			totalError += network.Train(pattern)
		}
		if i%1000 == 0 {
			fmt.Printf("Epoch %d, Error: %f\n", i, totalError/float64(len(patterns)))
		}
	}

	// Test the network
	fmt.Println("\nTesting network:")
	for _, pattern := range patterns {
		output := network.FeedForward(pattern.Features)
		fmt.Printf("Input: %v, Expected: %v, Got: %v\n",
			pattern.Features,
			pattern.MultipleExpectation,
			output)
	}
}
