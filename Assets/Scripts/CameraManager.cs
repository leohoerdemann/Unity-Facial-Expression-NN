using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class CameraManager
{
    private static WebCamTexture _webCamTexture;

    // Starts the camera feed
    public static void StartCamera()
    {
        if (_webCamTexture == null)
        {
            _webCamTexture = new WebCamTexture();
        }
        _webCamTexture.Play();
    }

    // Stops the camera feed
    public static void StopCamera()
    {
        if (_webCamTexture != null && _webCamTexture.isPlaying)
        {
            _webCamTexture.Stop();
        }
    }

    // Returns the current camera texture
    public static Texture GetCameraTexture()
    {
        return _webCamTexture;
    }

    // Checks if the camera is currently active
    public static bool IsCameraRunning()
    {
        return _webCamTexture != null && _webCamTexture.isPlaying;
    }
}
