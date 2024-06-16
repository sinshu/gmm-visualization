using System;
using System.Collections.Generic;
using System.Linq;
using NumFlat;
using NumFlat.Clustering;
using NumFlat.Distributions;
using ScottPlot;

public static class Program
{
    private static Vec<double>[] colorValues =
    [
        [244,  67,  54], // R
        [ 33, 150, 243], // G
        [ 76, 175,  80], // B
    ];

    private static Vec<double>[] colors = colorValues.Select(vec => vec / 255).ToArray();

    public static void Main(string[] args)
    {
        var clusterCount = 3;
        var random = new Random(57);
        var data = CreateData(random);

        var whole = data.ToGaussian();
        var components = new GaussianMixtureModel.Component[clusterCount];
        for (var i = 0; i < clusterCount; i++)
        {
            var offset = VectorBuilder.FromFunc(2, j => GetRandom(random));
            components[i] = new(1.0 / clusterCount, new(whole.Mean + offset, whole.Covariance));
        }
        var gmm = new GaussianMixtureModel(components);

        for (var i = 0; i < 20; i++)
        {
            Console.WriteLine(i);
            Plot($"test{i}.svg", data, gmm);
            gmm = gmm.Update(data).Model;
        }
    }

    private static void Plot(string path, Vec<double>[] data, GaussianMixtureModel gmm)
    {
        using (var plot = new Plot())
        {
            plot.Axes.SetLimitsX(-8, 8);
            plot.Axes.SetLimitsY(-8, 8);

            foreach (var vec in data)
            {
                var color = colors.Mean(gmm.PredictProbability(vec)).Map(value => (float)value);
                var x = new double[] { vec[0] };
                var y = new double[] { vec[1] };
                var scatter = plot.Add.ScatterPoints(x, y, new Color(color[0], color[1], color[2]));
                scatter.MarkerSize = 15F;
            }

            foreach (var (component, color) in gmm.Components.Zip(colors))
            {
                var xCenter = component.Gaussian.Mean[0];
                var yCenter = component.Gaussian.Mean[1];
                var svd = component.Gaussian.Covariance.Svd();
                var radiusX = 2 * Math.Sqrt(svd.S[0]);
                var radiusY = 2 * Math.Sqrt(svd.S[1]);
                var rotation = Math.Atan2(-svd.U[0, 1], svd.U[0, 0]) / Math.PI * 180;
                var ellipse = plot.Add.Ellipse(xCenter, yCenter, radiusY, radiusY, (float)rotation);
                ellipse.LineColor = new Color((float)color[0], (float)color[1], (float)color[2]);
                ellipse.FillColor = new Color((float)color[0], (float)color[1], (float)color[2], 0.1F);
            }

            plot.SaveSvg(path, 1600, 1200);
        }
    }

    private static Vec<double>[] CreateData(Random random)
    {
        Vec<double>[] means =
        [
            [ 4.0,  4.3],
            [-4.5,  1.3],
            [ 0.5, -4.3],
        ];

        Mat<double>[] transformations =
        [
            [
                [ 1.5, -0.4],
                [-0.3,  1.2],
            ],
            [
                [ 0.9, -0.5],
                [ 0.0,  1.3],
            ],
            [
                [ 1.0,  0.4],
                [ 0.5,  1.5],
            ],
        ];

        var vectorCountPerCluster = 100;
        var vectors = new List<Vec<double>>();
        foreach (var (mean, transformation) in means.Zip(transformations))
        {
            for (var i = 0; i < vectorCountPerCluster; i++)
            {
                Vec<double> a = [GetRandom(random), GetRandom(random)];
                a = transformation * a + mean;
                vectors.Add(a);
            }
        }

        return vectors.ToArray();
    }

    private static double GetRandom(Random random)
    {
        var sum = 0.0;
        for (var i = 0; i < 12; i++)
        {
            sum += random.NextDouble();
        }
        return sum - 6;
    }
}
