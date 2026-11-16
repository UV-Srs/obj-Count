import React, { useState, useRef, useEffect } from 'react';
import {
  Upload,
  Download,
  FileSpreadsheet,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Layers,
  ImageIcon,
  Video,
  Zap,
  Cpu,
  CheckCircle2,
  RefreshCw,
} from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

export default function App() {
  const [activeMode, setActiveMode] = useState('image'); // 'image' | 'video'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Model Engine State
  const [activeModel, setActiveModel] = useState('yolov8m');
  const [modelSwitching, setModelSwitching] = useState(false);
  const [modelToast, setModelToast] = useState('');

  // Image Mode State
  const [originalImageSrc, setOriginalImageSrc] = useState(null);
  const [activeImageFile, setActiveImageFile] = useState(null);
  const [imageResult, setImageResult] = useState(null);

  // Video Mode State
  const [originalVideoSrc, setOriginalVideoSrc] = useState(null);
  const [activeVideoFile, setActiveVideoFile] = useState(null);
  const [videoResult, setVideoResult] = useState(null);

  const [showTable, setShowTable] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const imageInputRef = useRef(null);
  const videoInputRef = useRef(null);

  // Fetch initial active model from backend
  useEffect(() => {
    const fetchModelStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/model/info`);
        if (res.ok) {
          const data = await res.json();
          if (data.active_profile) {
            setActiveModel(data.active_profile);
          }
        }
      } catch (err) {
        console.warn('Could not query model status:', err);
      }
    };
    fetchModelStatus();
  }, []);

  // Handle switching between YOLOv8m (1024px) and YOLOv8n (640px)
  const handleModelSwitch = async (targetProfile) => {
    if (targetProfile === activeModel || modelSwitching) return;
    setModelSwitching(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('profile', targetProfile);

      const res = await fetch(`${API_BASE}/api/model/switch`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Switch failed with status ${res.status}`);
      }

      const data = await res.json();
      setActiveModel(data.active_profile);

      const modelDisplayName =
        data.active_profile === 'yolov8m'
          ? 'YOLOv8 Medium (1024px High-Res Engine)'
          : 'YOLOv8 Nano (640px Lightweight Engine)';

      setModelToast(`Switched active engine to: ${modelDisplayName}`);
      setTimeout(() => setModelToast(''), 4500);
      // If an image or video is already loaded, immediately re-run detection so user sees the new model's output!
      if (activeMode === 'video' && activeVideoFile) {
        processVideoFile(activeVideoFile);
      } else if (activeMode === 'image' && activeImageFile) {
        processImageFile(activeImageFile);
      }
    } catch (err) {
      setError(`Failed to switch model: ${err.message}`);
    } finally {
      setModelSwitching(false);
    }
  };

  // Open native OS file manager based on current mode
  const handleOpenFileDialog = () => {
    if (activeMode === 'video') {
      if (videoInputRef.current) {
        videoInputRef.current.value = '';
        videoInputRef.current.click();
      }
    } else {
      if (imageInputRef.current) {
        imageInputRef.current.value = '';
        imageInputRef.current.click();
      }
    }
  };

  const MAX_IMAGE_SIZE_MB = 25;
  const MAX_VIDEO_SIZE_MB = 100;

  // Process uploaded image file
  const processImageFile = async (file) => {
    if (!file) return;

    if (file.size > MAX_IMAGE_SIZE_MB * 1024 * 1024) {
      setError(`Image size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the maximum allowed limit of ${MAX_IMAGE_SIZE_MB} MB. Please upload an image under ${MAX_IMAGE_SIZE_MB} MB.`);
      return;
    }

    setLoading(true);
    setError(null);
    setActiveImageFile(file);

    const reader = new FileReader();
    reader.onload = (e) => setOriginalImageSrc(e.target.result);
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('preset', 'auto');

    try {
      const res = await fetch(`${API_BASE}/api/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      setImageResult(data);
    } catch (err) {
      setError(`Image detection failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Process uploaded video file
  const processVideoFile = async (file) => {
    if (!file) return;

    if (file.size > MAX_VIDEO_SIZE_MB * 1024 * 1024) {
      setError(`Video size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the maximum allowed limit of ${MAX_VIDEO_SIZE_MB} MB. Please upload a video under ${MAX_VIDEO_SIZE_MB} MB.`);
      return;
    }

    setLoading(true);
    setError(null);
    setActiveVideoFile(file);

    // Create local object URL for instant original video playback
    const localVideoUrl = URL.createObjectURL(file);
    setOriginalVideoSrc(localVideoUrl);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('preset', 'auto');

    try {
      const res = await fetch(`${API_BASE}/api/detect-video`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      setVideoResult(data);
    } catch (err) {
      setError(`Video detection failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Auto-detect file type and process
  const handleAnyFile = (file) => {
    if (!file) return;
    const isVideo =
      file.type.startsWith('video/') ||
      /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);

    if (isVideo) {
      setActiveMode('video');
      processVideoFile(file);
    } else {
      setActiveMode('image');
      processImageFile(file);
    }
  };

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleAnyFile(e.dataTransfer.files[0]);
    }
  };

  const handleImageFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processImageFile(e.target.files[0]);
    }
  };

  const handleVideoFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processVideoFile(e.target.files[0]);
    }
  };

  // Current active result & file
  const isVideo = activeMode === 'video';
  const currentResult = isVideo ? videoResult : imageResult;
  const currentFile = isVideo ? activeVideoFile : activeImageFile;

  // Download annotated image or video
  const handleDownload = () => {
    if (isVideo && videoResult?.video_url) {
      const link = document.createElement('a');
      link.href = `${API_BASE}${videoResult.video_url}`;
      link.download = `pipe_count_${videoResult.count}_video.mp4`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else if (!isVideo && imageResult?.annotated_image) {
      const link = document.createElement('a');
      link.href = imageResult.annotated_image;
      link.download = `pipe_count_${imageResult.count}_result.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // Export CSV
  const handleExportCSV = () => {
    if (!currentResult?.detections?.length) return;
    const headers = ['Pipe_ID', 'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height'];
    const rows = currentResult.detections.map((d) => [
      d.id,
      d.x1,
      d.y1,
      d.x2,
      d.y2,
      d.width,
      d.height,
    ]);
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.href = encodedUri;
    link.download = `pipe_detections_${currentResult.count}_${activeMode}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="app-layout">
      {/* Side Navigation Bar: Photo Count & Video Count */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon-box">
            <Layers size={18} />
          </div>
          <div className="brand-text">
            <span className="brand-name">PipeCount</span>
            <span className="brand-tag">Vision AI</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button
            type="button"
            className={`nav-item ${activeMode === 'image' ? 'active' : ''}`}
            onClick={() => setActiveMode('image')}
          >
            <ImageIcon size={18} />
            <span>Photo Count</span>
          </button>

          <button
            type="button"
            className={`nav-item ${activeMode === 'video' ? 'active' : ''}`}
            onClick={() => setActiveMode('video')}
          >
            <Video size={18} />
            <span>Video Count</span>
          </button>
        </nav>

        {/* Sidebar Engine Card */}
        <div className="sidebar-engine-box">
          <div className="engine-box-header">
            <span className="engine-pulse-dot" />
            <span className="engine-box-title">Engine Active</span>
          </div>
          <div className="engine-model-name">
            {activeModel === 'yolov8m' ? 'YOLOv8 Medium' : 'YOLOv8 Nano'}
          </div>
          <div className="engine-model-detail">
            <span>Input Size: </span>
            <strong>{activeModel === 'yolov8m' ? '1024 × 1024' : '640 × 640'}</strong>
          </div>
          <div className="engine-model-sub">
            {activeModel === 'yolov8m' ? 'High-Resolution Engine' : 'Lightweight Fast Baseline'}
          </div>
        </div>
      </aside>

      {/* Main Viewport */}
      <div className="main-viewport">
        <header className="topbar">
          <div>
            <h1 className="page-heading">
              {isVideo ? 'Video Counting Dashboard' : 'Photo Counting Dashboard'}
            </h1>
            <p className="page-subheading">
              {isVideo
                ? 'Upload a video to analyze and count pipes across video frames'
                : 'Upload a pipe bundle or box photo to calculate total count'}
            </p>
          </div>

          <div className="topbar-actions">
            {/* Interactive Model Switcher Toggle */}
            <div className="model-selector-wrapper">
              <span className="model-selector-caption">AI Engine:</span>
              <div className="model-selector-pills">
                <button
                  type="button"
                  className={`model-pill ${activeModel === 'yolov8m' ? 'active' : ''}`}
                  onClick={() => handleModelSwitch('yolov8m')}
                  disabled={modelSwitching}
                  title="Switch to YOLOv8 Medium (1024px)"
                >
                  <Zap size={14} className="pill-icon" />
                  <span className="pill-title">YOLOv8m</span>
                  <span className="pill-badge pill-badge-gold">1024px</span>
                </button>

                <button
                  type="button"
                  className={`model-pill ${activeModel === 'yolov8n' ? 'active' : ''}`}
                  onClick={() => handleModelSwitch('yolov8n')}
                  disabled={modelSwitching}
                  title="Switch to YOLOv8 Nano (640px)"
                >
                  <Cpu size={14} className="pill-icon" />
                  <span className="pill-title">YOLOv8n</span>
                  <span className="pill-badge pill-badge-gray">640px</span>
                </button>
              </div>
            </div>

            {currentResult && (
              <button
                type="button"
                className="btn btn-upload-new"
                onClick={handleOpenFileDialog}
              >
                <Upload size={15} />
                <span>{isVideo ? 'Upload New Video' : 'Upload New Image'}</span>
              </button>
            )}
          </div>
        </header>

        <div className="page-body">
          {/* Hidden File Inputs for native File Manager */}
          <input
            ref={imageInputRef}
            type="file"
            className="hidden-file-input"
            accept=".jpg,.jpeg,.jfif,.gfif,.png,.webp,.bmp,.tiff,.tif,.gif"
            onChange={handleImageFileInput}
          />
          <input
            ref={videoInputRef}
            type="file"
            className="hidden-file-input"
            accept=".mp4,.avi,.mov,.mkv,.webm,video/*"
            onChange={handleVideoFileInput}
          />

          {/* Model Switch Confirmation Notification */}
          {modelToast && (
            <div className="alert alert-success">
              <CheckCircle2 size={18} />
              <span>{modelToast}</span>
            </div>
          )}

          {/* Model Switching Feedback */}
          {modelSwitching && (
            <div className="alert alert-info">
              <RefreshCw size={18} className="spin-icon" />
              <span>Switching detection engine and updating results...</span>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="alert alert-error">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          {/* Upload Dropzone (First Time / Empty State) */}
          {!currentResult && !loading && (
            <div
              className={`dropzone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={handleOpenFileDialog}
            >
              <div className="dropzone-icon">
                {isVideo ? <Video size={32} /> : <Upload size={32} />}
              </div>
              <h2 className="dropzone-title">
                {isVideo
                  ? 'Click to upload or drag & drop video'
                  : 'Click to upload or drag & drop photo'}
              </h2>
              <p className="dropzone-text">
                {isVideo
                  ? 'Supports MP4, AVI, MOV, MKV, WEBM • Max size: 100 MB'
                  : 'Supports JPG, PNG, WEBP, BMP, TIFF • Max size: 25 MB'}
              </p>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="loading-card">
              <div className="spinner" />
              <p className="loading-title">
                {isVideo
                  ? 'Analyzing video frames and counting pipes...'
                  : 'Detecting and counting pipes...'}
              </p>
              <p className="loading-subtitle">Applying anti-double and distractor suppression</p>
            </div>
          )}

          {/* Results: Stacked Layout */}
          {currentResult && !loading && (
            <div className="results-wrapper">
              {/* Stats Bar */}
              <div className="stats-grid">
                <div className="stat-card primary-stat">
                  <span className="stat-label">Total Count</span>
                  <span className="stat-number">{currentResult.count}</span>
                </div>

                <div className="stat-card">
                  <span className="stat-label">Model Architecture</span>
                  <span className="stat-number stat-badge-text">
                    {currentResult.metrics?.model_profile === 'yolov8n'
                      ? 'YOLOv8n (640px)'
                      : 'YOLOv8m (1024px)'}
                  </span>
                </div>

                <div className="stat-card">
                  <span className="stat-label">Processed Time</span>
                  <span className="stat-number">
                    {currentResult.metrics.inference_time_ms} ms
                  </span>
                </div>

                {isVideo && currentResult.metrics?.total_frames && (
                  <div className="stat-card">
                    <span className="stat-label">Total Frames</span>
                    <span className="stat-number">
                      {currentResult.metrics.total_frames}
                    </span>
                  </div>
                )}

                <div className="stat-card actions-stat">
                  <span className="stat-label">Actions</span>
                  <div className="actions-buttons">
                    <button type="button" className="btn btn-primary" onClick={handleDownload}>
                      <Download size={14} />
                      <span>{isVideo ? 'Download Video' : 'Download Image'}</span>
                    </button>
                    <button type="button" className="btn btn-outline" onClick={handleExportCSV}>
                      <FileSpreadsheet size={14} />
                      <span>Export CSV</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* 1. Original Input (First, on top) */}
              <section className="image-section">
                <div className="section-header">
                  <h3 className="section-title">
                    {isVideo ? '1. Original Video' : '1. Original Image'}
                  </h3>
                  <span className="section-badge">{currentFile?.name || 'Input Source'}</span>
                </div>
                <div className="image-frame">
                  {isVideo ? (
                    <video
                      src={originalVideoSrc}
                      controls
                      playsInline
                      className="view-video"
                    />
                  ) : (
                    <img src={originalImageSrc} alt="Original input" className="view-image" />
                  )}
                </div>
              </section>

              {/* 2. Detection Result (Directly below) */}
              <section className="image-section detected-section">
                <div className="section-header">
                  <div className="header-flex">
                    <h3 className="section-title">
                      {isVideo ? '2. Detection Result (Video & Keyframe)' : '2. Detection Result'}
                    </h3>
                    <span className="count-pill">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '6px' }}>
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                        <polyline points="22 4 12 14.01 9 11.01" />
                      </svg>
                      {currentResult.count} Pipes Counted
                    </span>
                  </div>
                </div>

                <div className="image-frame">
                  {isVideo ? (
                    <div className="video-result-container">
                      <video
                        src={`${API_BASE}${currentResult.video_url}`}
                        controls
                        playsInline
                        className="view-video"
                      />
                      {currentResult.annotated_image && (
                        <div className="keyframe-card">
                          <div className="keyframe-banner">
                            <span>Detection Keyframe Snapshot</span>
                          </div>
                          <img
                            src={currentResult.annotated_image}
                            alt={`Detected ${currentResult.count} pipes`}
                            className="view-image"
                          />
                        </div>
                      )}
                    </div>
                  ) : (
                    <img
                      src={currentResult.annotated_image}
                      alt={`Detected ${currentResult.count} pipes`}
                      className="view-image"
                    />
                  )}
                </div>
              </section>

              {/* Coordinates Accordion without Confidence column */}
              <div className="table-accordion">
                <button
                  type="button"
                  className="accordion-toggle"
                  onClick={() => setShowTable(!showTable)}
                >
                  <span>Pipe Bounding Boxes ({currentResult.detections.length} items)</span>
                  {showTable ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>

                {showTable && (
                  <div className="table-container">
                    <table className="coords-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Bounding Box</th>
                          <th>Width</th>
                          <th>Height</th>
                        </tr>
                      </thead>
                      <tbody>
                        {currentResult.detections.map((d) => (
                          <tr key={d.id}>
                            <td>{d.id}</td>
                            <td>[{d.x1}, {d.y1}, {d.x2}, {d.y2}]</td>
                            <td>{d.width}px</td>
                            <td>{d.height}px</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
