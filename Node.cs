using System;
using System.Collections.Generic;

namespace Linear_Classifier
{
    public class Node
    {
        public List<float> Inputs  = new();
        public List<float> Weights = new();
        public float Net
        {
            get
            {
                float net = 0f;
                for (int i = 0; i < Inputs.Count; i++)
                    net += (Inputs[i] * Weights[i]);
                return net;
            }
            private set { }
        }

        public float TargetOutput;
        public float PreviousNodeOutput;

        public Node() { }
        
        public Node(List<float> weights)
        {
            Inputs = new List<float>();
            Weights = new List<float>(weights);
        }

        public void CopyInputs(List<float> inputs)
            => Inputs = new List<float>(inputs);
        public void CopyWeights(List<float> weights)
            => Weights = new List<float>(weights);
        public float ForwardStep()
            => PreviousNodeOutput = Sigmoid(Net);
        public float OutputNodeErrorRate()
            => PreviousNodeOutput = TargetOutput - Net;

        public float HiddenNodeErrorRate(List<float> weights, List<float> outputs)
        {
            // Calculate hidden errors
            // 𝛿ℎ = 𝑜ℎ ∗ (1 − 𝑜ℎ) ∗ summation 𝑤𝑘ℎ𝛿𝑘
            // 𝑜ℎ = PreviousNodeOutput = Sigmoid(Net)
            // 𝑤𝑘ℎ = weights from this node to output layer node
            // 𝛿𝑘 = PreviousNodeOutput on the output layer node
            float offset = PreviousNodeOutput * (1 - PreviousNodeOutput);
            float net = 0f;
            for (int i = 0; i < outputs.Count; i++)
                net += (outputs[i] * weights[i]);
            return PreviousNodeOutput = offset * net;
        }

        public void UpdateWeights()
        {
            // Δin = 𝜂𝛿n𝑥i
            // 𝜂 = learning rate = 0.1
            // 𝛿n = error rate last step
            // 𝑥i = input
            for (int i = 0; i < Inputs.Count; i++)
                Weights[i] += (NodeManager.LEARNING_RATE * PreviousNodeOutput * Inputs[i]);
        }

        public static float Sigmoid(float net)
            => 1 / (1 + MathF.Exp(-net));
    }
}