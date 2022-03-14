using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Linear_Classifier
{
    public class Perceptron
    {
        private List<(List<float> inputs, float targetOutput)> _inputsToOutputs = new();
        private List<float> _weights;
        private bool _isHiddenLayer;
        public Perceptron(List<float> weights, bool isHiddenLayer, float outputOverride = float.NaN)
        {
            _weights = weights;
            _isHiddenLayer = isHiddenLayer;
            if (!float.IsNaN(outputOverride))
                OverrideOutput(outputOverride);
        }

        public void AddInput((List<float>, float) input)
          => _inputsToOutputs.Add(input);

        public void OverrideInput(List<float> inputs)
        {
            // For now we only support one piece of data just to test.
            float targetOutput = _inputsToOutputs.Count > 0 ? _inputsToOutputs[0].targetOutput : -1f;
            _inputsToOutputs = new();
            _inputsToOutputs.Add((inputs, targetOutput));
        }

        public void OverrideWeights(List<float> weights)
            => _weights = weights;

        public void OverrideOutput(float targetOutput)
        {
            // TODO: Made this better
            List<List<float>> inputs = new();
            foreach (var input in _inputsToOutputs)
                inputs.Add(input.inputs);

            _inputsToOutputs.Clear();
            foreach (var input in inputs)
                _inputsToOutputs.Add((input, targetOutput));
        }

        public float  TrainBackwardStep()
        {
            List<float> inputs = _inputsToOutputs[0].inputs;
            int inputCount = inputs.Count;
            int attempts = 0;

            // Calculate summation of all inputs multiplied by their weights
            float net = 0f;
            for (int i = 0; i < inputCount; i++)
                net += inputs[i] * _weights[i];

            float sigmaK = 0f;
            for (int i = 0; i < inputs.Count; i++)
            {
                float input = inputs[i];
                float weight = _weights[i];
                sigmaK += (input * weight);
            }

            //float output = useActivationFunc ? Sigmoid(net) : net;
            float errorRate = net * (1 - net) * sigmaK;
            return errorRate;
        }

        // Calculates and returns net of inputs * weights 
        public float TrainForwardStep(bool sigma)
        {
            List<float> inputs = _inputsToOutputs[0].inputs;
            // Store by value since we need to access this multiple times
            int inputCount = inputs.Count;
            float net = 0f;
            for (int i = 0; i < inputCount; i++)
                net += inputs[i] * _weights[i];

            // Calculate output and error rate
            float output = sigma ? Sigmoid(net) : net;
            return output;
        }

        public float Sigmoid(float net)
            => 1 / (1 + MathF.Exp(-net));
    }
}
