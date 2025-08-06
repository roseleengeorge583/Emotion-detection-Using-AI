import React, { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import { Camera, Upload, RefreshCw } from 'lucide-react';

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [emotions, setEmotions] = useState<any>(null);
  const detectionIntervalRef = useRef<number>();

  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.faceExpressionNet.loadFromUri('/models')
        ]);
        setIsModelLoaded(true);
      } catch (error) {
        console.error('Error loading models:', error);
      }
    };
    loadModels();

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
      stopWebcam();
    };
  }, []);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 480 },
          height: { ideal: 360 },
          facingMode: 'user'
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = resolve;
          }
        });
        
        await videoRef.current.play();
        setIsWebcamActive(true);
        startDetection();
      }
    } catch (error) {
      console.error('Error accessing webcam:', error);
      alert('Could not access webcam. Please ensure you have granted camera permissions.');
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsWebcamActive(false);
      
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
      
      if (canvasRef.current) {
        const context = canvasRef.current.getContext('2d');
        context?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      
      setEmotions(null);
    }
  };

  const startDetection = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const displaySize = { width: video.videoWidth || 480, height: video.videoHeight || 360 };
    
    canvas.width = displaySize.width;
    canvas.height = displaySize.height;
    faceapi.matchDimensions(canvas, displaySize);

    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }

    detectionIntervalRef.current = window.setInterval(async () => {
      if (!video || video.paused || video.ended || !canvas) return;

      try {
        const detections = await faceapi
          .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
          .withFaceExpressions();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        const context = canvas.getContext('2d');
        
        if (context) {
          context.clearRect(0, 0, canvas.width, canvas.height);
          
          if (resizedDetections.length > 0) {
            setEmotions(resizedDetections[0].expressions);
            faceapi.draw.drawDetections(canvas, resizedDetections);
            faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
          } else {
            setEmotions(null);
          }
        }
      } catch (error) {
        console.error('Error in detection:', error);
      }
    }, 100);
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
        setEmotions(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setAnalyzing(true);
    const img = await faceapi.fetchImage(selectedImage);
    
    try {
      const detections = await faceapi
        .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceExpressions();

      if (detections.length > 0) {
        setEmotions(detections[0].expressions);
      } else {
        setEmotions(null);
      }
    } catch (error) {
      console.error('Error analyzing image:', error);
    }
    
    setAnalyzing(false);
  };

  const renderEmotions = () => {
    if (!emotions) return null;

    return (
      <div className="mt-4 p-4 bg-white rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">Detected Emotions:</h3>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(emotions).map(([emotion, value]) => (
            <div key={emotion} className="flex justify-between">
              <span className="capitalize">{emotion}:</span>
              <span className="font-semibold">{(Number(value) * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">Emotion Detection</h1>
        
        <div className="grid md:grid-cols-2 gap-8">
          {/* Webcam Section */}
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Webcam Detection</h2>
            <div className="relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="rounded-lg bg-gray-200 w-full h-auto"
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full"
              />
            </div>
            <div className="mt-4">
              {!isWebcamActive ? (
                <button
                  onClick={startWebcam}
                  disabled={!isModelLoaded}
                  className="flex items-center justify-center w-full py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
                >
                  <Camera className="mr-2" size={20} />
                  Start Webcam
                </button>
              ) : (
                <button
                  onClick={stopWebcam}
                  className="flex items-center justify-center w-full py-2 px-4 bg-red-600 text-white rounded-lg hover:bg-red-700"
                >
                  Stop Webcam
                </button>
              )}
            </div>
          </div>

          {/* Image Upload Section */}
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Image Analysis</h2>
            <div className="mb-4">
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="mb-2" size={24} />
                  <p className="text-sm text-gray-600">Click to upload an image</p>
                </div>
                <input
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageUpload}
                />
              </label>
            </div>
            
            {selectedImage && (
              <>
                <div className="relative">
                  <img
                    src={selectedImage}
                    alt="Uploaded"
                    className="w-full rounded-lg"
                  />
                </div>
                <button
                  onClick={analyzeImage}
                  disabled={analyzing || !isModelLoaded}
                  className="flex items-center justify-center w-full mt-4 py-2 px-4 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
                >
                  {analyzing ? (
                    <RefreshCw className="animate-spin mr-2" size={20} />
                  ) : (
                    'Analyze Image'
                  )}
                </button>
              </>
            )}
          </div>
        </div>

        {/* Emotions Display */}
        {renderEmotions()}
      </div>
    </div>
  );
}

export default App;