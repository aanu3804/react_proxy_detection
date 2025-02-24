import React, { useEffect, useRef, useState, useCallback } from "react";
import * as faceapi from "face-api.js";

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);

  const referenceDescriptorsRef = useRef([]);
  const isMonitoringRef = useRef(false);

  const [isReferenceCaptured, setIsReferenceCaptured] = useState(false);
  const [proxyDetected, setProxyDetected] = useState(false);
  const [audioProxyDetected, setAudioProxyDetected] = useState(false);
  const [faceExpressions, setFaceExpressions] = useState({});
  const [statusMessage, setStatusMessage] = useState("Loading models...");
  const [isCameraOn, setIsCameraOn] = useState(false);

  const TOLERANCE = 0.5;
  const NOISE_THRESHOLD = 0.4; // Adjust based on environment
  const MULTIPLE_VOICES_THRESHOLD = 5; // Number of frequency peaks indicating multiple voices

  // ✅ Load Models
  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
          faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
          faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
          faceapi.nets.faceExpressionNet.loadFromUri("/models"),
        ]);
        setStatusMessage("✅ Models loaded. Click 'Start Camera'.");
      } catch (error) {
        console.error("Error loading models:", error);
        setStatusMessage("❗ Error loading models. Please reload the page.");
      }
    };
    loadModels();
  }, []);

  // 🎥 Start Camera & Microphone
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      if (videoRef.current) videoRef.current.srcObject = stream;

      // Initialize audio context
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const audioStream = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      audioStream.connect(analyserRef.current);

      setIsCameraOn(true);
      setStatusMessage("📸 Camera started. Capture reference photo.");
    } catch (error) {
      console.error("Error accessing devices:", error);
      setStatusMessage("❗ Please allow camera and microphone permissions.");
    }
  };

  // ⏹️ Stop Camera & Microphone
  const stopCamera = () => {
    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsCameraOn(false);
    setIsReferenceCaptured(false);
    referenceDescriptorsRef.current = [];
    setFaceExpressions({});
    setAudioProxyDetected(false);
    setStatusMessage("⏹️ Camera stopped.");
  };

  // 📸 Capture Reference Photo
  const captureReferencePhoto = async () => {
    if (!videoRef.current) return;

    const detections = await faceapi
      .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors()
      .withFaceExpressions();

    if (detections.length > 0) {
      referenceDescriptorsRef.current = detections.map((det) => det.descriptor);
      setIsReferenceCaptured(true);
      setStatusMessage("✅ Reference photo captured! Click 'Start Monitoring'.");
    } else {
      setStatusMessage("❗ No face detected! Please try again.");
    }
  };

  // ✨ Draw Face Boxes and Expressions
  const drawFaceTracking = useCallback((detections) => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((det) => {
      const { x, y, width, height } = det.detection.box;
      const color =
        referenceDescriptorsRef.current.some(
          (refDesc) => faceapi.euclideanDistance(det.descriptor, refDesc) < TOLERANCE
        )
          ? "green"
          : "red";

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      ctx.font = "16px Arial";
      ctx.fillStyle = color;
      ctx.fillText(getMostLikelyExpression(det.expressions), x, y - 5);
    });
  }, []);

  // 💡 Get Most Likely Expression
  const getMostLikelyExpression = (expressions) => {
    return Object.entries(expressions).reduce((acc, [expr, value]) =>
      value > acc.value ? { expression: expr, value } : acc
    , { expression: "", value: 0 }).expression;
  };

  // 🎙️ Audio Detection Loop
  const startAudioDetection = useCallback(() => {
    const detectAudio = () => {
      if (!isMonitoringRef.current || !analyserRef.current) return;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyserRef.current.getByteFrequencyData(dataArray);

      // Calculate volume (RMS - Root Mean Square)
      const volume = Math.sqrt(dataArray.reduce((sum, val) => sum + val ** 2, 0) / bufferLength) / 255;

      // Detect multiple voices based on frequency peaks
      const peakCount = dataArray.filter((val) => val > 150).length; // Adjust threshold for multiple voices

      const isNoiseDetected = volume > NOISE_THRESHOLD;
      const isMultipleVoices = peakCount > MULTIPLE_VOICES_THRESHOLD;

      setAudioProxyDetected(isNoiseDetected || isMultipleVoices);

      if (isNoiseDetected) {
        setStatusMessage("🚨 Proxy in Audio! Noise detected!");
      } else if (isMultipleVoices) {
        setStatusMessage("🚨 Multiple Voices Detected!");
      }

      if (isMonitoringRef.current) {
        requestAnimationFrame(detectAudio);
      }
    };

    detectAudio();
  }, []);

  // 🔄 Face Detection Loop
  const startFaceDetection = useCallback(async () => {
    const detectFaces = async () => {
      if (!videoRef.current || !isMonitoringRef.current) return;

      try {
        const detections = await faceapi
          .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptors()
          .withFaceExpressions();

        if (detections.length > 0) {
          drawFaceTracking(detections);

          let newExpressions = {};
          let isProxyFound = false;

          detections.forEach((det, index) => {
            newExpressions[`Face ${index + 1}`] = getMostLikelyExpression(det.expressions);

            if (isReferenceCaptured && referenceDescriptorsRef.current.length > 0) {
              const isKnownFace = referenceDescriptorsRef.current.some(
                (refDesc) => faceapi.euclideanDistance(det.descriptor, refDesc) < TOLERANCE
              );
              if (!isKnownFace) isProxyFound = true;
            }
          });

          setFaceExpressions(newExpressions);
          setProxyDetected(isProxyFound);

          if (isProxyFound) {
            setStatusMessage("🚨 Proxy detected! Unauthorized person detected!");
          } else if (!audioProxyDetected) {
            setStatusMessage("✅ Monitoring in progress...");
          }
        } else {
          clearCanvas();
          setFaceExpressions({});
        }
      } catch (error) {
        console.error("Error during face detection:", error);
      }

      if (isMonitoringRef.current) {
        requestAnimationFrame(detectFaces);
      }
    };

    detectFaces();
  }, [drawFaceTracking, isReferenceCaptured, audioProxyDetected]);

  // ▶️ Start Monitoring
  const startMonitoring = () => {
    if (!isReferenceCaptured) {
      setStatusMessage("❗ Please capture your reference photo first.");
      return;
    }
    isMonitoringRef.current = true;
    setProxyDetected(false);
    setAudioProxyDetected(false);
    setFaceExpressions({});
    setStatusMessage("🔎 Monitoring in progress...");
    startFaceDetection();
    startAudioDetection();
  };

  // ⏹️ Stop Monitoring
  const stopMonitoring = () => {
    isMonitoringRef.current = false;
    clearCanvas();
    setFaceExpressions({});
    setProxyDetected(false);
    setAudioProxyDetected(false);
    setStatusMessage("⏹️ Monitoring stopped.");
  };

  // 🧹 Clear Canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>Advanced Proxy Detection System</h1>
      <div style={{ position: "relative", display: "inline-block" }}>
        <video ref={videoRef} autoPlay playsInline width="640" height="480"></video>
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0 }} />
      </div>

      <div style={{ marginTop: "10px" }}>
        {!isCameraOn ? (
          <button onClick={startCamera} style={buttonStyle}>
            📸 Start Camera
          </button>
        ) : (
          <>
            {!isReferenceCaptured && (
              <button onClick={captureReferencePhoto} style={buttonStyle}>
                📷 Capture Reference Photo
              </button>
            )}
            <button onClick={startMonitoring} style={buttonStyle}>
              ▶️ Start Monitoring
            </button>
            <button onClick={stopMonitoring} style={buttonStyle}>
              ⏹️ Stop Monitoring
            </button>
            <button onClick={stopCamera} style={buttonStyle}>
              🚫 Stop Camera
            </button>
          </>
        )}
      </div>

      <div>
        <h3>Detected Facial Expressions:</h3>
        {Object.entries(faceExpressions).map(([face, expr], index) => (
          <p key={index}>
            {face}: <strong>{expr}</strong>
          </p>
        ))}
      </div>

      {proxyDetected && (
        <div style={{ color: "red", fontWeight: "bold", marginTop: "10px" }}>
          🚨 Proxy Detected! Unauthorized person detected!
        </div>
      )}

      {audioProxyDetected && (
        <div style={{ color: "red", fontWeight: "bold", marginTop: "10px" }}>
          🚨 Proxy in Audio! Noise or Multiple Voices Detected!
        </div>
      )}

      <div style={{ marginTop: "20px", fontWeight: "bold", color: proxyDetected || audioProxyDetected ? "red" : "green" }}>
        {statusMessage}
      </div>
    </div>
  );
};

// 🌟 Button Styles
const buttonStyle = {
  margin: "5px",
  padding: "10px 15px",
  border: "none",
  borderRadius: "8px",
  backgroundColor: "#4CAF50",
  color: "#fff",
  fontSize: "16px",
  cursor: "pointer",
  transition: "background-color 0.3s",
};
export default App;
