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
            List<float> outputLayerNets = nodeMan.Test(new List<float> { 0.30f, 0.70f, 0.90f });
            
            for (int i = 0; i < outputLayerNets.Count; i++)
            {
                float numerator = MathF.Exp(outputLayerNets[i]);
                float denominator = 0f;
                for (int j = 0; j < outputLayerNets.Count; j++)
                    denominator += MathF.Exp(outputLayerNets[j]);
                
                float probabiltyDistribution = numerator / denominator;
                Console.WriteLine($"Probabilty Distribution: {probabiltyDistribution}");
            }
        }
    }
}
