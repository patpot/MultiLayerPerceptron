using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Linear_Classifier
{
    public class PerceptronLayer
    {
        private List<Perceptron> _perceptrons = new List<Perceptron>();
        private bool _useActivationFunction;

        public PerceptronLayer(bool useActivationFunc = true)
            => _useActivationFunction = useActivationFunc;
        public void AddPerceptron(Perceptron perceptron)
            => _perceptrons.Add(perceptron);

        public void AddInputs((List<float>, float) inputs)
            => _perceptrons.ForEach(_ => _.AddInput(inputs));

        public void OverrideInputs(List<float> inputs)
            => _perceptrons.ForEach(_ => _.OverrideInput(inputs));

        public List<(float errorRate, float output)> Train()
        {
            List<(float errorRate, float output)> ret = new List<(float errorRate, float output)>();
            foreach (var perceptron in _perceptrons)
            {
                (float errorRate, float output) errorRate = perceptron.Train(_useActivationFunction);
                ret.Add(errorRate);
            }
            return ret;
        }
    }
}
