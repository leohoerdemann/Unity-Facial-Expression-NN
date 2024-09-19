using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public static class ExpressionModelHandler
{
    private static IWorker worker;
    private static NNModel modelAsset;

    // Load the model (call this once at the beginning)
    public static void LoadModel(NNModel model)
    {
        modelAsset = model;
        var modelLoaded = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, modelLoaded);
    }

    // Process the camera feed and predict the emotion
    public static string PredictEmotion(Texture2D inputImage)
    {
        if (worker == null) return "No model loaded.";

        // Preprocess the input image
        Tensor inputTensor = new Tensor(inputImage, 3);
        worker.Execute(inputTensor);

        // Get the prediction
        Tensor outputTensor = worker.PeekOutput();
        string predictedEmotion = ProcessOutput(outputTensor);
        
        inputTensor.Dispose();
        outputTensor.Dispose();

        return predictedEmotion;
    }

    // Map the model output to a string emotion
    private static string ProcessOutput(Tensor output)
    {
        // Assuming the output is a classification, map it to an emotion
        int emotionIndex = output.ArgMax()[0];
        string[] emotions = { "Happy", "Sad", "Angry", "Surprised" }; // Modify based on your model's labels
        return emotions[emotionIndex];
    }
}
