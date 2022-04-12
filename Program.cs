using System;
using System.Collections.Generic;
using System.IO;

namespace Linear_Classifier
{
    class Program
    {
        static void Main(string[] args)
        {
            NodeManager nodeMan = new NodeManager();
            nodeMan.Initialise();
            nodeMan.Train(100);

            List<float> testNets = nodeMan.Test(new List<float> { 0.30f, 0.70f, 0.90f});

            // Probability Distribution (node) = Softmax (net_i) = e^(net_i) / sum(e^(net_j))
            for (int i = 0; i < testNets.Count; i++)
            {
                float numerator = MathF.Exp(testNets[i]); // e^(net_i)
                float denominator = 0f;
                for (int j = 0; j < testNets.Count; j++) // sum(e^(net_i))
                    denominator += MathF.Exp(testNets[j]);
                
                float probabiltyDistribution = numerator / denominator; // e^(net_i) / sum(e^(net_i))
                Console.WriteLine($"Probabilty Distribution: {probabiltyDistribution}");
            }
        }
    }
}
