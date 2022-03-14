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

        private PerceptronLayer _outputLayer;
        private PerceptronLayer _hiddenLayer;
        public PerceptronManager()
        {
            // Output layer
            PerceptronLayer outputLayer = new PerceptronLayer(useActivationFunc: false);
            outputLayer.AddPerceptron(new Perceptron(new List<float> { 0.7f, 0.6f, 0.2f }, false, 1));
            outputLayer.AddPerceptron(new Perceptron(new List<float> { 0.9f, 0.8f, 0.4f }, false, 0));
            _outputLayer = outputLayer;

            // Hidden Layer
            PerceptronLayer hiddenLayer = new PerceptronLayer(useActivationFunc: true);
            hiddenLayer.AddPerceptron(new Perceptron(new List<float> { 0.5f, -0.2f, 0.5f }, true));
            hiddenLayer.AddPerceptron(new Perceptron(new List<float> { 0.1f, 0.2f, 0.3f }, true));
            _hiddenLayer = hiddenLayer;

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

            // Add all data to our hidden
            foreach (var inputs in inputsToOutputs)
                _hiddenLayer.AddInputs(inputs);
        }

        public List<(float, float)> TrainPerceptrons()
        {
            // Run our forward step
            List<(float errorRate, float output)> finalErrorRates = new List<(float, float)>();
            //for (int i = _perceptronsByLayer.Count - 1; i > 0; i--)
            //{
            //    // Get our layer and train it
            //    PerceptronLayer layer = _perceptronsByLayer[i];
            //    List<(float errorRate, float output)> errorRates = layer.Train();

            //    // If we're going to iterate again, go "down" a layer and set their values
            //    if (i - 1 > 0)
            //    {
            //        List<float> inputs = new();
            //        foreach (var errorRate in errorRates)
            //            inputs.Add(errorRate.output);
            //        inputs.Add(BIAS);
            //        _perceptronsByLayer[i - 1].OverrideInputs(inputs);
            //    }

            //    foreach (var errorRate in errorRates)
            //        finalErrorRates.Add((errorRate.errorRate, errorRate.output));
            //}

            //List<(float errorRate, float output)> outputs = new();
            //List<float> errorRates = new();

            //outputs = _hiddenLayer.Train(true);
            //foreach (var output in outputs)
            //    errorRates.Add(output.errorRate);
            //_outputLayer.OverrideInputs(errorRates);

            //// Get our error rates from the output layer
            //outputs = _outputLayer.Train(true);
            //foreach (var output in outputs)
            //    errorRates.Add(output.errorRate);
            //_hiddenLayer.OverrideInputs(errorRates);

            //finalErrorRates = _hiddenLayer.Train(false);

            // This calculates o4 and o5 from Slide 30.
            List<float> nodeOutputs = _hiddenLayer.TrainForward(sigma:true);
            nodeOutputs.Add(BIAS); // Add our bias input in
            _outputLayer.OverrideInputs(nodeOutputs);
            // This calculates net6 and net7 from Slide 30
            List<float> nodeNets = _outputLayer.TrainForward(sigma:false);


            List<float> targetOutputs = new List<float> { 1, 0 };
            List<float> outputErrors = new List<float>();
            // If there is an error thrown here your target outputs don't match your node count
            // This calculates 𝛿(delta)6 and 𝛿(delta)7
            for (int i = 0; i < nodeNets.Count; i++)
                outputErrors.Add(targetOutputs[i] - nodeNets[i]);

            _hiddenLayer.OverrideWeights(new List<float> { 0.7f, 0.9f });
            _hiddenLayer.OverrideInputs(outputErrors);

            List<float> hiddenErrors = _hiddenLayer.TrainBackward();

            _outputLayer.OverrideInputs(outputErrors);
            //List<float> hiddenErrors = _outputLayer.TrainForward();

            //hiddenErrors.ForEach(_ => Console.WriteLine(_));


            // Now we've got our outputs
            return finalErrorRates;
        }
    }
}
