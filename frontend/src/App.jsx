import React, { useState, useRef } from 'react';
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
} from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

export default function App() {
  const [activeMode, setActiveMode] = useState('image'); // 'image' | 'video'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

  // Process uploaded image file
  const processImageFile = async (file) => {
    if (!file) return;
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
                  ? 'Supports MP4, AVI, MOV, MKV, WEBM'
                  : 'Supports JPG, PNG, WEBP, JFIF, GFIF, BMP, TIFF, GIF'}
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
                    <span className="count-pill">{currentResult.count} Pipes</span>
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
