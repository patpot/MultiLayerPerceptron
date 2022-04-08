using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Linear_Classifier
{
    public class NodeManager
    {
        public List<Node> HiddenLayer = new();
        public List<Node> OutputLayer = new();

        public const int BIAS = 1;
        public const float LEARNING_RATE = 0.1f;

        public void Initialise()
        {
            List<float> inputs = ConvertInputData(Environment.CurrentDirectory + "\\data.txt");

            Node n4 = new Node(inputs: inputs, weights: new List<float>() { 0.5f, -0.2f, 0.5f });
            Node n5 = new Node(inputs: inputs, weights: new List<float>() { 0.1f,  0.2f, 0.3f });
            HiddenLayer.Add(n4); HiddenLayer.Add(n5);

            Node n6 = new Node(); n6.CopyWeights (new List<float>() { 0.7f, 0.6f, 0.2f }); n6.TargetOutput = 1f;
            Node n7 = new Node(); n7.CopyWeights (new List<float>() { 0.9f, 0.8f, 0.4f }); n7.TargetOutput = 0f;
            OutputLayer.Add(n6); OutputLayer.Add(n7);
        }

        public void OverrideHiddenLayerInputs(List<float> newInputs)
            => HiddenLayer.ForEach(node => node.CopyInputs(newInputs));
        public List<float> ConvertInputData(string path)
        {
            List<float> inputs = new();

            // Read in our text and split it into lines
            string text = File.ReadAllText(path);
            string[] split = System.Text.RegularExpressions.Regex.Split(text, @"\s{1,}");
            // Split apart all the inputs from the ideal output value
            for (int i = 0; i < split.Length; i++)
                inputs.Add(float.Parse(split[i]));

            return inputs;
        }

        public void Train(int epochCount)
        {
            List<List<float>> outputWeights = new();
            List<float> outputNodeErrors = new();
            List<float> hiddenNodeErrors = new();
            for (int i = 0; i < epochCount; i++)
            {
                List<float> outputWeight = new();
                // Add our weights to our output weights
                HiddenLayer.ForEach(node => node.Weights.ForEach(weight => outputWeight.Add(weight)));
                OutputLayer.ForEach(node => node.Weights.ForEach(weight => outputWeight.Add(weight)));
                outputWeights.Add(outputWeight);

                // Take our first forward step in the Hidden Layer
                List<float> hiddenNodeNets = new();
                foreach (var node in HiddenLayer)
                    hiddenNodeNets.Add(node.ForwardStep());

                outputNodeErrors.Clear();
                // Move our data gathered from the forward step into the Output Layer
                foreach (var node in OutputLayer)
                {
                    node.CopyInputs(hiddenNodeNets);
                    node.Inputs.Add(BIAS); // Add the bias from node 3
                    outputNodeErrors.Add(node.OutputBackwardStep());
                }

                hiddenNodeErrors.Clear();
                // Override our inputs and weights in our Hidden Layer, then perform our backward step
                for (int j = 0; j < HiddenLayer.Count; j++)
                {
                    List<float> weights = new();
                    foreach (var node in OutputLayer)
                        weights.Add(node.Weights[j]);
                    // Do a backwards step with our outputs and weights from the output layer
                    hiddenNodeErrors.Add(HiddenLayer[j].HiddenBackwardStep(weights, outputNodeErrors));
                }

                // Now that we're done calculating our error rates, update all our weights accordingly
                HiddenLayer.ForEach(node => node.UpdateWeights());
                OutputLayer.ForEach(node => node.UpdateWeights());
            }

            foreach (var output in outputNodeErrors)
                Console.WriteLine($"Error Rate:{output}");
        }

        public List<float> Test(List<float> inputs)
        {
            OverrideHiddenLayerInputs(inputs);

            // Take our first forward step in the Hidden Layer
            List<float> hiddenLayerNets = new();
            foreach (var node in HiddenLayer)
                hiddenLayerNets.Add(node.ForwardStep());

            List<float> ouputLayerNets = new();
            // Move our data gathered from the forward step into the Output Layer
            foreach (var node in OutputLayer)
            {
                node.CopyInputs(hiddenLayerNets);
                node.Inputs.Add(BIAS); // Add the bias from node 3
                ouputLayerNets.Add(node.Net);
            }

            return ouputLayerNets;
        }
    }
}
