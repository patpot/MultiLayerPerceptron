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
        public Perceptron(List<float> weights)
        {
            _weights = weights;
        }

        public void AddInput((List<float>, float) input)
          => _inputsToOutputs.Add(input);

        public void OverrideInput(List<float> inputs)
        {
            // For now we only support one piece of data just to test.
            float targetOutput = _inputsToOutputs[0].targetOutput;
            _inputsToOutputs = new();
            _inputsToOutputs.Add((inputs, targetOutput));
            Console.WriteLine("breakpoitn");
        }

        public (float, float) Train(bool useActivationFunc)
        {
            foreach (var train in _inputsToOutputs)
            {
                // Store by value since we need to access this multiple times
                int inputCount = train.inputs.Count;
                int attempts = 0;

                while (attempts < PerceptronManager.ATTEMPT_COUNT)
                {
                    // Calculate summation of all inputs multiplied by their weights
                    float net = 0f;
                    for (int i = 0; i < inputCount; i++)
                        net += train.inputs[i] * _weights[i];

                    // Calculate output and error rate
                    float output = useActivationFunc ? Sigmoid(net) : net;
                    float errorRate = train.targetOutput - output;
                    Console.WriteLine($"Error Rate: {errorRate}. Output: {output}"); // Log our data so we can monitor it

                    // Recalculate our weight
                    for (int i = 0; i < inputCount; i++)
                        _weights[i] = _weights[i] + (PerceptronManager.LEARNING_RATE * (errorRate) * train.inputs[i]);

                    attempts++;
                    // Store our final error rate after classifying
                    if (attempts == PerceptronManager.ATTEMPT_COUNT)
                    {
                        Console.WriteLine("---- END OF TRAINING ---- \n\n\n");
                        return (errorRate, output);
                    }
                }
            }

            // This will never be returned but VS needs it here.
            return (-1f, -1f);
        }

        public float Sigmoid(float net)
            => 1 / (1 + MathF.Exp(-net));
    }
}
