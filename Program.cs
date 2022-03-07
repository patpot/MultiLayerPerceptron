using System;
using System.Collections.Generic;
using System.IO;

namespace Linear_Classifier
{
    class Program
    {
        static void Main(string[] args)
        {
            PerceptronManager perceptronMan = new PerceptronManager();
            perceptronMan.ConvertInputData(Environment.CurrentDirectory + "\\data.txt");

            // Train our perceptrons in order to get our error rates
            List<(float, float)> finalErrorRates = new();
            finalErrorRates = perceptronMan.TrainPerceptrons();

            float averageErrorRate = 0f;
            foreach (var finalError in finalErrorRates)
            {
                Console.WriteLine($"Error Rate: {finalError.Item1}. Output: {finalError.Item2}");
                averageErrorRate += finalError.Item1;
            }
            averageErrorRate /= finalErrorRates.Count;
            Console.Write($"Average error rate: {averageErrorRate}");
        }
    }
}
