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

        public void OverrideWeights(List<float> weights)
            => _perceptrons.ForEach(_ => _.OverrideWeights(weights));

        //public List<(float errorRate, float output)> Train(bool activationFunction)
        //{
        //    List<(float errorRate, float output)> ret = new List<(float errorRate, float output)>();
        //    // Loop through all our nodes and train them
        //    foreach (var perceptron in _perceptrons)
        //    {
        //        (float errorRate, float output) errorRate = perceptron.Train(_useActivationFunction, activationFunction);
        //        ret.Add(errorRate);
        //    }
        //    return ret;
        //}

        public List<float> TrainForward(bool sigma)
        {
            List<float> ret = new();
            foreach (var perceptron in _perceptrons)
                ret.Add(perceptron.TrainForwardStep(sigma));
            return ret;
        }

        public List<float> TrainBackward()
        {
            List<float> ret = new();
            foreach (var perceptron in _perceptrons)
                ret.Add(perceptron.TrainBackwardStep());
            return ret;
        }
    }
}
