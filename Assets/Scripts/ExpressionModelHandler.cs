using Unity.Barracuda;
using UnityEngine;

public static class ExpressionModelHandler
{
    private static IWorker worker;
    private static NNModel modelAsset;
    private static string outputName;

    /// <summary>
    /// Loads the emotion detection ONNX model.
    /// </summary>
    /// <param name="model">The ONNX model asset.</param>
    public static void LoadModel(NNModel model)
    {
        modelAsset = model;
        var modelLoaded = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, modelLoaded);

        if (modelLoaded.outputs.Count > 0)
        {
            outputName = modelLoaded.outputs[0];
            Debug.Log($"ExpressionModelHandler: Output name: {outputName}");
        }
        else
        {
            Debug.LogError("ExpressionModelHandler: No outputs found in the model.");
        }
    }

    /// <summary>
    /// Predicts emotion based on the detected face region.
    /// </summary>
    /// <param name="inputImage">The full camera frame.</param>
    /// <param name="faceRect">Bounding box of the detected face.</param>
    /// <returns>Predicted emotion as a string.</returns>
    public static string PredictEmotion(Texture2D inputImage, Rect faceRect)
    {
        if (worker == null)
        {
            Debug.LogError("ExpressionModelHandler: Model not loaded.");
            return "Model not loaded.";
        }

        // Crop the face from the image based on the detected face rectangle
        Texture2D faceTexture = CropTexture(inputImage, faceRect);
        faceTexture.Apply();

        // Resize the image to the required 48x48 size and convert to grayscale
        Texture2D resizedFaceTexture = ResizeTexture(faceTexture, 48, 48);
        Texture2D grayscaleFaceTexture = ConvertToGrayscale(resizedFaceTexture);

        // Convert to Tensor (assuming the model expects a single channel grayscale image)
        Tensor inputTensor = new Tensor(grayscaleFaceTexture, channels: 1); // 1 channel for grayscale

        // Run inference
        worker.Execute(inputTensor);

        // Get the output
        if (string.IsNullOrEmpty(outputName))
        {
            Debug.LogError("ExpressionModelHandler: Output name is not set.");
            inputTensor.Dispose();
            return "Output name not set.";
        }

        Tensor outputTensor = worker.PeekOutput(outputName);

        // Log tensor shape for debugging
        Debug.Log($"ExpressionModelHandler: Output shape: {outputTensor.shape}");

        // Process the output tensor to get the predicted emotion
        string predictedEmotion = ProcessOutput(outputTensor);

        // Dispose of tensors to prevent memory leaks
        inputTensor.Dispose();
        outputTensor.Dispose();

        return predictedEmotion;
    }

    /// <summary>
    /// Crops the texture to the specified rectangle.
    /// </summary>
    /// <param name="source">Original texture.</param>
    /// <param name="rect">Rectangle to crop.</param>
    /// <returns>Cropped texture.</returns>
    private static Texture2D CropTexture(Texture2D source, Rect rect)
    {
        int x = Mathf.Clamp((int)rect.x, 0, source.width);
        int y = Mathf.Clamp((int)rect.y, 0, source.height);
        int width = Mathf.Clamp((int)rect.width, 0, source.width - x);
        int height = Mathf.Clamp((int)rect.height, 0, source.height - y);

        Color[] pixels = source.GetPixels(x, y, width, height);
        Texture2D croppedTexture = new Texture2D(width, height);
        croppedTexture.SetPixels(pixels);
        croppedTexture.Apply();
        return croppedTexture;
    }

    /// <summary>
    /// Resizes a texture to the specified dimensions.
    /// </summary>
    /// <param name="source">Original texture.</param>
    /// <param name="newWidth">New width.</param>
    /// <param name="newHeight">New height.</param>
    /// <returns>Resized texture.</returns>
    private static Texture2D ResizeTexture(Texture2D source, int newWidth, int newHeight)
    {
        RenderTexture rt = RenderTexture.GetTemporary(newWidth, newHeight);
        Graphics.Blit(source, rt);
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D newTexture = new Texture2D(newWidth, newHeight, TextureFormat.RGB24, false);
        newTexture.ReadPixels(new Rect(0, 0, newWidth, newHeight), 0, 0);
        newTexture.Apply();

        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(rt);

        return newTexture;
    }

    /// <summary>
    /// Converts a texture to grayscale.
    /// </summary>
    /// <param name="source">Original texture.</param>
    /// <returns>Grayscale texture.</returns>
    private static Texture2D ConvertToGrayscale(Texture2D source)
    {
        Texture2D grayscale = new Texture2D(source.width, source.height);
        for (int y = 0; y < source.height; y++)
        {
            for (int x = 0; x < source.width; x++)
            {
                Color pixel = source.GetPixel(x, y);
                float gray = pixel.grayscale;
                grayscale.SetPixel(x, y, new Color(gray, gray, gray, pixel.a));
            }
        }
        grayscale.Apply();
        return grayscale;
    }

    /// <summary>
    /// Processes the model's output tensor to determine the predicted emotion.
    /// </summary>
    /// <param name="output">Output tensor from the model.</param>
    /// <returns>Predicted emotion as a string.</returns>
    private static string ProcessOutput(Tensor output)
    {
        // Assuming the output is a 1D tensor with probabilities for each emotion class
        int maxIndex = 0;
        float maxProb = output[0];
        for (int i = 1; i < output.length; i++)
        {
            if (output[i] > maxProb)
            {
                maxProb = output[i];
                maxIndex = i;
            }
        }

        string[] emotions = { "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear" }; // Modify based on your model's labels
        if (maxIndex < emotions.Length)
        {
            return emotions[maxIndex];
        }
        else
        {
            return "Unknown";
        }
    }
}
