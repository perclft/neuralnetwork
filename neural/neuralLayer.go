package neural

import (
	log "github.com/sirupsen/logrus"
)

// PrepareLayer create a NeuralLayer with n NeuronUnits inside
// [n:int] is an int that specifies the number of neurons in the NeuralLayer
// [p:int] is an int that specifies the number of neurons in the previous NeuralLayer
// It returns a NeuralLayer object
func PrepareLayer(n int, p int) (l NeuralLayer) {

	l = NeuralLayer{NeuronUnits: make([]NeuronUnit, n), Length: n}

	for i := 0; i < n; i++ {
		RandomNeuronInit(&l.NeuronUnits[i], p)
	}

	log.WithFields(log.Fields{
		"level":               "info",
		"msg":                 "multilayer perceptron init completed",
		"neurons":             len(l.NeuronUnits),
		"lengthPreviousLayer": l.Length,
	}).Info("Complete NeuralLayer init.")

	return

}