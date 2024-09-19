using UnityEngine;
using TMPro;
using UnityEngine.UI;
using Unity.Barracuda;

public class UIManager : MonoBehaviour
{
    [Header("UI Elements")]
    public RawImage cameraDisplay;       // Display camera feed
    public TMP_Text emotionText;         // Display emotion text

    [Header("Models")]
    public NNModel faceDetectionModel;   // ONNX model for face detection
    public NNModel emotionModel;         // ONNX model for emotion detection

    private WebCamTexture webCamTexture;

    void Start()
    {
        // Load both face detection and emotion detection models
        FaceDetectionModelHandler.LoadModel(faceDetectionModel);
        ExpressionModelHandler.LoadModel(emotionModel);

        // Start the camera feed
        OnStartCamera();
    }

    /// <summary>
    /// Starts the webcam feed.
    /// </summary>
    public void OnStartCamera()
    {
        if (webCamTexture == null)
        {
            webCamTexture = new WebCamTexture();
            cameraDisplay.texture = webCamTexture;
            webCamTexture.Play();
        }
    }

    /// <summary>
    /// Stops the webcam feed.
    /// </summary>
    public void OnStopCamera()
    {
        if (webCamTexture != null && webCamTexture.isPlaying)
        {
            webCamTexture.Stop();
            cameraDisplay.texture = null;
        }
    }

    void Update()
    {
        if (webCamTexture != null && webCamTexture.isPlaying)
        {
            // Capture the current frame from the camera
            Texture2D frame = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGB24, false);
            frame.SetPixels(webCamTexture.GetPixels());
            frame.Apply();

            // Detect faces in the current frame
            Rect[] detectedFaces = FaceDetectionModelHandler.DetectFaces(frame, threshold: 0.7f);

            if (detectedFaces.Length > 0)
            {
                // Iterate through detected faces
                foreach (Rect faceRect in detectedFaces)
                {
                    // Draw rectangles around detected faces (for debugging purposes)
                    DrawRectangle(frame, faceRect, Color.red);

                    // Predict emotion for the detected face
                    string predictedEmotion = ExpressionModelHandler.PredictEmotion(frame, faceRect);
                    emotionText.text = predictedEmotion;
                }
            }
            else
            {
                emotionText.text = "No face detected";
            }

            // Display the updated camera feed
            cameraDisplay.texture = frame;
        }
    }

    /// <summary>
    /// Draws rectangles around detected faces on the texture.
    /// </summary>
    /// <param name="texture">The camera frame texture.</param>
    /// <param name="rect">Bounding box of the detected face.</param>
    /// <param name="color">Color of the rectangle.</param>
    private void DrawRectangle(Texture2D texture, Rect rect, Color color)
    {
        // Ensure rectangle is within texture bounds
        int startX = Mathf.Clamp((int)rect.x, 0, texture.width - 1);
        int startY = Mathf.Clamp((int)rect.y, 0, texture.height - 1);
        int endX = Mathf.Clamp((int)(rect.x + rect.width), 0, texture.width - 1);
        int endY = Mathf.Clamp((int)(rect.y + rect.height), 0, texture.height - 1);

        // Draw left and right edges
        for (int y = startY; y <= endY; y++)
        {
            if (startX >= 0 && startX < texture.width)
                texture.SetPixel(startX, y, color);                      // Left edge
            if (endX >= 0 && endX < texture.width)
                texture.SetPixel(endX, y, color);                        // Right edge
        }

        // Draw top and bottom edges
        for (int x = startX; x <= endX; x++)
        {
            if (startY >= 0 && startY < texture.height)
                texture.SetPixel(x, startY, color);                      // Top edge
            if (endY >= 0 && endY < texture.height)
                texture.SetPixel(x, endY, color);                        // Bottom edge
        }

        texture.Apply();
    }
}
