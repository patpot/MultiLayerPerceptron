using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        public float OutputBackwardStep()
            => PreviousNodeOutput = TargetOutput - Net;

        public float HiddenBackwardStep(List<float> weights, List<float> outputs)
        {
            float offset = PreviousNodeOutput * (1 - PreviousNodeOutput);
            float net = 0f;
            for (int i = 0; i < outputs.Count; i++)
                net += (outputs[i] * weights[i]);
            return PreviousNodeOutput = offset * net;
        }

        public float GetLastErrorRate()
            => TargetOutput - PreviousNodeOutput;

        public void UpdateWeights()
        {
            for (int i = 0; i < Inputs.Count; i++)
                Weights[i] += (NodeManager.LEARNING_RATE * PreviousNodeOutput * Inputs[i]);
        }

        public static float Sigmoid(float net)
            => 1 / (1 + MathF.Exp(-net));
    }
}
