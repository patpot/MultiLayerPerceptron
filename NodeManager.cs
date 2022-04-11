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

        public List<List<float>> Inputs = new();
        public List<(float, float)> TargetOutputs = new();

        public const int BIAS = 1;
        public const float LEARNING_RATE = 0.1f;

        public void Initialise()
        {
            LoadInputData(Environment.CurrentDirectory + "\\data.txt");

            // Create our nodes with weights declared in the brief. Add them to their assigned layers
            Node n4 = new Node(weights: new List<float>() { 0.9f,  0.74f, 0.8f, 0.35f });
            Node n5 = new Node(weights: new List<float>() { 0.45f, 0.13f, 0.4f, 0.97f });
            Node n6 = new Node(weights: new List<float>() { 0.36f,  0.68f, 0.10f, 0.96f });
            HiddenLayer.Add(n4); HiddenLayer.Add(n5); HiddenLayer.Add(n6);

            Node n7 = new Node(); n7.CopyWeights (new List<float>() { 0.98f, 0.35f, 0.50f, 0.90f });
            Node n8 = new Node(); n8.CopyWeights (new List<float>() { 0.92f, 0.8f, 0.13f, 0.8f });
            OutputLayer.Add(n7); OutputLayer.Add(n8);
        }

        public void OverrideHiddenLayerInputs(List<float> newInputs)
            => HiddenLayer.ForEach(node => node.CopyInputs(newInputs));
        public void LoadInputData(string path)
        {
            // Read in our text and split it into lines
            string text = File.ReadAllText(path);
            string[] split = System.Text.RegularExpressions.Regex.Split(text, @"\s{2,}");
            // Split apart all the inputs from the ideal output value
            for (int i = 0; i < split.Length; i++)
            {
                // split string[i] then parse into a list of floats
                string[] split2 = split[i].Split(' ');
                // This is specific to the format we were given and not very adaptable
                // Our inputs are the first 3 values in the file, our target outputs are the last 2
                List<float> inputs = new()
                {
                    float.Parse(split2[0]),
                    float.Parse(split2[1]),
                    float.Parse(split2[2]),
                };
                Inputs.Add(inputs);
                TargetOutputs.Add((float.Parse(split2[3]), float.Parse(split2[4])));
            }
        }

        public void Train(int epochCount)
        {
            // Setup data that exists outside of the epoch scope
            List<List<float>> outputWeights = new();
            List<float> outputNodeErrors = new();
            List<float> squaredErrors = new();
            for (int i = 0; i < epochCount; i++)
            {
                List<float> outputWeight = new();
                // Add our weights to our output weights
                HiddenLayer.ForEach(node => node.Weights.ForEach(weight => outputWeight.Add(weight)));
                OutputLayer.ForEach(node => node.Weights.ForEach(weight => outputWeight.Add(weight)));
                outputWeights.Add(outputWeight);
                Console.WriteLine($"Running epoch: {i}");
                for (int ii = 0; ii < Inputs.Count; ii++)
                {
                    // Assign the nodes the correct input data for this iteration
                    HiddenLayer.ForEach(node => node.CopyInputs(Inputs[ii]));

                    // Take our first forward step in the Hidden Layer
                    List<float> hiddenNodeNets = new();
                    foreach (var node in HiddenLayer)
                        hiddenNodeNets.Add(node.ForwardStep());

                    outputNodeErrors.Clear();
                    // Move our data gathered from the forward step into the Output Layer
                    foreach (var node in OutputLayer)
                    {
                        node.CopyInputs(hiddenNodeNets);
                        node.Inputs.Add(BIAS); // Add the bias from node x0
                        outputNodeErrors.Add(node.OutputBackwardStep());
                    }

                    float squaredErrorSummation = 0f;
                    // Total all the output node errors then square them
                    foreach (var outputNodeError in outputNodeErrors)
                        squaredErrorSummation += outputNodeError;
                    squaredErrorSummation *= squaredErrorSummation;
                    squaredErrorSummation *= 0.5f;
                    squaredErrors.Add(squaredErrorSummation);

                    List<float> hiddenNodeErrors = new();
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
            }

            foreach (var output in outputNodeErrors)
                Console.WriteLine($"Error Rate:{output}");

            foreach (var sqError in squaredErrors)
                Console.WriteLine($"Squared Error:{sqError}");

            Console.Write("| step");
            for (int i = 0; i <= 10; i++)
                Console.Write($"|  {i}  ");
            for (int i = 0; i < outputWeights[0].Count; i++)
            {
                Console.WriteLine($"\n");
                for (int ii = 0; ii < 10; ii++)
                    Console.Write($"|{outputWeights[ii][i].ToString("0.000")}");
            }
            Console.WriteLine($"\n");
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
