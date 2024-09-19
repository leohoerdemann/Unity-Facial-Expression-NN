using Unity.Barracuda;
using UnityEngine;
using System.Collections.Generic;

public static class FaceDetectionModelHandler
{
    private static IWorker worker;
    private static NNModel modelAsset;
    private static string[] outputNames;

    /// <summary>
    /// Loads the face detection ONNX model.
    /// </summary>
    /// <param name="model">The ONNX model asset.</param>
    public static void LoadModel(NNModel model)
    {
        modelAsset = model;
        var modelLoaded = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, modelLoaded);

        outputNames = modelLoaded.outputs.ToArray();

        for (int i = 0; i < outputNames.Length; i++)
        {
            Debug.Log($"FaceDetectionModelHandler: Output {i}: {outputNames[i]}");
        }
    }

    /// <summary>
    /// Scales the bounding box to ensure it's square.
    /// </summary>
    /// <param name="box">Original bounding box.</param>
    /// <returns>Scaled bounding box.</returns>
    private static Rect ScaleBox(Rect box)
    {
        float width = box.width;
        float height = box.height;
        float maxSize = Mathf.Max(width, height);
        float dx = (maxSize - width) / 2f;
        float dy = (maxSize - height) / 2f;

        return new Rect(box.xMin - dx, box.yMin - dy, box.width + 2 * dx, box.height + 2 * dy);
    }

    /// <summary>
    /// Detects faces in the input image using the loaded ONNX model.
    /// </summary>
    /// <param name="inputImage">The input camera frame.</param>
    /// <param name="threshold">Confidence threshold for detections.</param>
    /// <returns>Array of detected face bounding boxes.</returns>
    public static Rect[] DetectFaces(Texture2D inputImage, float threshold = 0.7f)
    {
        if (worker == null)
        {
            Debug.LogError("FaceDetectionModelHandler: Model not loaded.");
            return new Rect[0];
        }

        // Preprocess the image: resize, mean subtraction, normalization
        Texture2D preprocessedImage = PreprocessImage(inputImage, 320, 240);

        // Convert the preprocessed image to a tensor
        Tensor inputTensor = new Tensor(preprocessedImage, channels: 3); // RGB channels

        // Run inference
        worker.Execute(inputTensor);

        // Retrieve confidences and bounding boxes from the model output
        if (outputNames.Length < 2)
        {
            Debug.LogError("FaceDetectionModelHandler: Model does not have enough outputs.");
            inputTensor.Dispose();
            return new Rect[0];
        }

        // Adjust these output names based on your model's actual output names
        string confidenceOutputName = outputNames[0]; // e.g., "scores"
        string boxesOutputName = outputNames[1];       // e.g., "boxes"

        Tensor confidences = worker.PeekOutput(confidenceOutputName);
        Tensor boxes = worker.PeekOutput(boxesOutputName);

        // Log tensor shapes for debugging
        Debug.Log($"FaceDetectionModelHandler: Confidences shape: {confidences.shape}");
        Debug.Log($"FaceDetectionModelHandler: Boxes shape: {boxes.shape}");

        // Postprocess output to extract valid bounding boxes
        Rect[] faceRects = Postprocess(confidences, boxes, inputImage.width, inputImage.height, threshold);

        // Dispose of tensors to prevent memory leaks
        inputTensor.Dispose();
        confidences.Dispose();
        boxes.Dispose();

        return faceRects;
    }

    /// <summary>
    /// Preprocesses the input image: resizing, mean subtraction, and normalization.
    /// </summary>
    /// <param name="image">Original input image.</param>
    /// <param name="targetWidth">Target width for resizing.</param>
    /// <param name="targetHeight">Target height for resizing.</param>
    /// <returns>Preprocessed image.</returns>
    private static Texture2D PreprocessImage(Texture2D image, int targetWidth, int targetHeight)
    {
        // Resize the image
        Texture2D resizedImage = ResizeTexture(image, targetWidth, targetHeight);

        // Subtract mean [127, 127, 127] and normalize by 128
        for (int y = 0; y < resizedImage.height; y++)
        {
            for (int x = 0; x < resizedImage.width; x++)
            {
                Color pixel = resizedImage.GetPixel(x, y);
                pixel.r = (pixel.r * 255 - 127) / 128f;
                pixel.g = (pixel.g * 255 - 127) / 128f;
                pixel.b = (pixel.b * 255 - 127) / 128f;
                resizedImage.SetPixel(x, y, pixel);
            }
        }
        resizedImage.Apply();
        return resizedImage;
    }

    /// <summary>
    /// Postprocesses the model's output to extract valid face bounding boxes.
    /// </summary>
    /// <param name="confidences">Confidence scores tensor.</param>
    /// <param name="boxes">Bounding boxes tensor.</param>
    /// <param name="imageWidth">Original image width.</param>
    /// <param name="imageHeight">Original image height.</param>
    /// <param name="threshold">Confidence threshold.</param>
    /// <returns>Array of valid face bounding boxes.</returns>
    private static Rect[] Postprocess(Tensor confidences, Tensor boxes, int imageWidth, int imageHeight, float threshold)
    {
        var faceRects = new List<Rect>();

        // Log tensor dimensions
        Debug.Log($"FaceDetectionModelHandler: Confidences Tensor Shape: {confidences.shape}");
        Debug.Log($"FaceDetectionModelHandler: Boxes Tensor Shape: {boxes.shape}");

        // Verify tensor shapes
        if (confidences.shape.batch != 1)
        {
            Debug.LogError("FaceDetectionModelHandler: Unexpected batch size in confidences.");
            return faceRects.ToArray();
        }

        if (boxes.shape.batch != 1 || boxes.shape.channels != 4)
        {
            Debug.LogError("FaceDetectionModelHandler: Unexpected shape in boxes.");
            return faceRects.ToArray();
        }

        int numBoxes = confidences.length;
        Debug.Log($"FaceDetectionModelHandler: Number of boxes: {numBoxes}");

        for (int i = 0; i < numBoxes; i++)
        {
            float confidence = confidences[i];
            if (confidence > threshold)
            {
                float xMin = boxes[i, 0] * imageWidth;
                float yMin = boxes[i, 1] * imageHeight;
                float xMax = boxes[i, 2] * imageWidth;
                float yMax = boxes[i, 3] * imageHeight;

                Rect faceRect = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
                faceRects.Add(ScaleBox(faceRect));

                // Log each detected face's details
                Debug.Log($"FaceDetectionModelHandler: Detected face {i}: {faceRect}");
            }
        }

        return faceRects.ToArray();
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
}
