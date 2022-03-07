using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Linear_Classifier
{
    public class PerceptronManager
    {
        public const float LEARNING_RATE = 0.1f;
        public const int BIAS = 1;
        public const int ATTEMPT_COUNT = 1;
        private List<PerceptronLayer> _perceptronsByLayer;
        public PerceptronManager()
        {
            _perceptronsByLayer = new List<PerceptronLayer>();

            PerceptronLayer layer1 = new PerceptronLayer(useActivationFunc: true);
            layer1.AddPerceptron(new Perceptron(new List<float> { 0.5f, -0.2f, 0.5f }));
            layer1.AddPerceptron(new Perceptron(new List<float> { 0.1f, 0.2f, 0.3f }));
            _perceptronsByLayer.Add(layer1);

            PerceptronLayer layer2 = new PerceptronLayer(useActivationFunc: false);
            layer2.AddPerceptron(new Perceptron(new List<float> { 0.7f, 0.6f, 0.2f }));
            layer2.AddPerceptron(new Perceptron(new List<float> { 0.9f, 0.8f, 0.4f }));
            _perceptronsByLayer.Add(layer2);
        }
        public void ConvertInputData(string path)
        {
            List<(List<float> inputs, float targetOutput)> inputsToOutputs = new();

            // Read in our text and split it into lines
            bool weirdFormatting = path.Contains("linearly");
            string text = File.ReadAllText(path);
            string[] splitText;
            if (weirdFormatting)
                splitText = text.Split("\r\n");
            else
                splitText = text.Split("\n");

            // Split our lines into their collections
            foreach (var line in splitText)
            {
                if (weirdFormatting)
                    line.Replace("\t", " ");

                string[] split = System.Text.RegularExpressions.Regex.Split(line, @"\s{1,}");
                // Split apart all the inputs from the ideal output value
                List<float> inputs = new();
                for (int i = 0; i < split.Length - 1; i++)
                    inputs.Add(float.Parse(split[i]));
                inputs.Add(BIAS);
                inputsToOutputs.Add((inputs, float.Parse(split[split.Length - 1])));
            }

            // Add all data to our individual perceptrons
            foreach (var inputs in inputsToOutputs)
                _perceptronsByLayer.ForEach(_ => _.AddInputs(inputs));
        }

        public List<(float, float)> TrainPerceptrons()
        {
            List<(float errorRate, float output)> finalErrorRates = new List<(float, float)>();
            for (int i = 0; i < _perceptronsByLayer.Count; i++)
            {
                PerceptronLayer layer = _perceptronsByLayer[i];
                List<(float errorRate, float output)> errorRates = layer.Train();
                // If we're going to iterate again, go "down" a layer and set their values
                if (i + 1 < _perceptronsByLayer.Count)
                {
                    List<float> inputs = new();
                    foreach (var errorRate in errorRates)
                        inputs.Add(errorRate.output);
                    inputs.Add(BIAS);
                    _perceptronsByLayer[i + 1].OverrideInputs(inputs);
                }
                
                foreach (var errorRate in errorRates)
                    finalErrorRates.Add((errorRate.errorRate, errorRate.output));
            }

            return finalErrorRates;
        }
    }
}
