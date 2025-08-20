package neural

import (
	log "github.com/sirupsen/logrus"
)

// PrepareMLPNet create a multi layer Perceptron neural network.
// [l:[]int] is an int array with layers neurons number [input, ..., output]
// [lr:int] is the learning rate of neural network
// [tr:transferFunc] is a transfer function
// [tr:transferFunc] the respective transfer function derivative
func PrepareMLPNet(l []int, lr float64, tf transferFunc, trd transferFunc) (mlp MultiLayerNetwork) {

	// setup learning rate and transfer function
	mlp.L_rate = lr
	mlp.T_func = tf
	mlp.T_func_d = trd

	// setup layers
	mlp.NeuralLayers = make([]NeuralLayer, len(l))

	// for each layers specified
	for il, ql := range l {

		// if it is not the first
		if il != 0 {

			// prepare the GENERIC layer with specific dimension and correct number of links for each NeuronUnits
			mlp.NeuralLayers[il] = PrepareLayer(ql, l[il-1])

		} else {

			// prepare the INPUT layer with specific dimension and No links to previous.
			mlp.NeuralLayers[il] = PrepareLayer(ql, 0)

		}

	}

	log.WithFields(log.Fields{
		"level":          "info",
		"msg":            "multilayer perceptron init completed",
		"layers":         len(mlp.NeuralLayers),
		"learningRate: ": mlp.L_rate,
	}).Info("Complete Multilayer Perceptron init.")

	return

}
