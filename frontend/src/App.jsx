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
  Zap,
  Cpu,
  CheckCircle2,
  RefreshCw,
  Sliders,
  Play,
  ShieldCheck,
} from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Model Engine State
  const [activeModel, setActiveModel] = useState('yolov8m');
  const [modelSwitching, setModelSwitching] = useState(false);
  const [modelToast, setModelToast] = useState('');

  // Image State
  const [originalImageSrc, setOriginalImageSrc] = useState(null);
  const [activeImageFile, setActiveImageFile] = useState(null);
  const [imageResult, setImageResult] = useState(null);

  // Confidence Threshold State (Universal static default: 0.10)
  const [confThreshold, setConfThreshold] = useState(0.10);
  const [appliedConfThreshold, setAppliedConfThreshold] = useState(0.10);
  const [showManualTuning, setShowManualTuning] = useState(false);

  const [showTable, setShowTable] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const imageInputRef = useRef(null);

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

      // If an image is already loaded, immediately re-run detection with current confidence!
      if (activeImageFile) {
        processImageFile(activeImageFile, confThreshold);
      }
    } catch (err) {
      setError(`Failed to switch model: ${err.message}`);
    } finally {
      setModelSwitching(false);
    }
  };

  // Open native OS file manager
  const handleOpenFileDialog = () => {
    if (imageInputRef.current) {
      imageInputRef.current.value = '';
      imageInputRef.current.click();
    }
  };

  const MAX_IMAGE_SIZE_MB = 25;

  // Process uploaded image file
  const processImageFile = async (file, customConf = null) => {
    if (!file) return;

    if (file.size > MAX_IMAGE_SIZE_MB * 1024 * 1024) {
      setError(`Image size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the maximum allowed limit of ${MAX_IMAGE_SIZE_MB} MB. Please upload an image under ${MAX_IMAGE_SIZE_MB} MB.`);
      return;
    }

    const confToUse = customConf !== null && customConf !== undefined ? customConf : confThreshold;

    setLoading(true);
    setError(null);
    setActiveImageFile(file);

    const reader = new FileReader();
    reader.onload = (e) => setOriginalImageSrc(e.target.result);
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('preset', 'auto');
    formData.append('conf', confToUse.toString());

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
      setAppliedConfThreshold(confToUse);
    } catch (err) {
      setError(`Image detection failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Recount current loaded file with desired confidence threshold
  const handleRecount = (overrideConf = null) => {
    const confToUse = overrideConf !== null && overrideConf !== undefined ? overrideConf : confThreshold;
    if (activeImageFile) {
      processImageFile(activeImageFile, confToUse);
    } else {
      handleOpenFileDialog();
    }
  };

  // Auto-detect file type and process
  const handleAnyFile = (file) => {
    if (!file) return;
    const isVideo =
      file.type.startsWith('video/') ||
      /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);

    if (isVideo) {
      setError('Video processing is disabled. Please upload a pipe photo/image.');
      return;
    }

    processImageFile(file);
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

  const hasPendingThresholdChange = Boolean(
    imageResult && activeImageFile && Math.abs(confThreshold - appliedConfThreshold) > 0.005
  );

  // Download annotated image
  const handleDownload = () => {
    if (imageResult?.annotated_image) {
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
    if (!imageResult?.detections?.length) return;
    const headers = ['Pipe_ID', 'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height'];
    const rows = imageResult.detections.map((d) => [
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
    link.download = `pipe_detections_${imageResult.count}_photo.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="app-layout">
      {/* Side Navigation Bar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon-box">
            <Layers size={18} />
          </div>
          <div className="brand-text">
            <span className="brand-name">Vision-intelligence</span>
            <span className="brand-tag">Vision AI</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button
            type="button"
            className="nav-item active"
          >
            <ImageIcon size={18} />
            <span>Dashboard</span>
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
            <h1 className="page-heading">Pipe Counting Dashboard</h1>
            <p className="page-subheading">
              Upload a pipe bundle or box photo to calculate total count
            </p>
          </div>

          <div className="topbar-actions">
            {/* Interactive Model Switcher Toggle */}
            <div className="model-selector-wrapper">
              <span className="model-selector-caption">Models:</span>
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

            {imageResult && (
              <button
                type="button"
                className="btn btn-upload-new"
                onClick={handleOpenFileDialog}
              >
                <Upload size={15} />
                <span>Upload New Image</span>
              </button>
            )}
          </div>
        </header>

        <div className="page-body">
          {/* Hidden File Input for native File Manager */}
          <input
            ref={imageInputRef}
            type="file"
            className="hidden-file-input"
            accept=".jpg,.jpeg,.jfif,.gfif,.png,.webp,.bmp,.tiff,.tif,.gif"
            onChange={handleImageFileInput}
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

          {/* Static Autonomous Production Calibration Banner */}
          <div className="confidence-control-card">
            <div className="confidence-control-main">
              <div className="confidence-label-group">
                <div className="confidence-title-row">
                  <ShieldCheck size={18} className="control-icon-static" />
                  <span className="confidence-title">Static Production Mode</span>
                  <span className="confidence-badge confidence-badge-static">
                    10% Calibrated (Locked)
                  </span>
                </div>
                <p className="confidence-hint">
                  Pre-calibrated universal static threshold. Automatically rejects carton vent holes, floor caster wheels, and worker heads across all camera angles with zero manual tuning.
                </p>
              </div>

              {/* Collapsible Manual Override Section */}
              {showManualTuning && (
                <div className="manual-tuning-box">
                  <div className="confidence-slider-group">
                    <div className="slider-header-inline">
                      <span className="slider-inline-caption">Manual Sensitivity: {Math.round(confThreshold * 100)}%</span>
                      {Math.abs(confThreshold - 0.10) > 0.005 && (
                        <button
                          type="button"
                          className="btn-reset-static"
                          onClick={() => {
                            setConfThreshold(0.10);
                            if (activeImageFile && !loading) {
                              handleRecount(0.10);
                            }
                          }}
                        >
                          Reset to 10% Static
                        </button>
                      )}
                    </div>
                    <input
                      type="range"
                      min="0.05"
                      max="0.80"
                      step="0.01"
                      value={confThreshold}
                      onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                      className="confidence-slider"
                      disabled={loading}
                    />
                    <div className="slider-range-labels">
                      <span>5% (Max Recall)</span>
                      <span>10% (Static Calibrated)</span>
                      <span>50%</span>
                      <span>80% (Strict)</span>
                    </div>
                  </div>

                  {/* Quick Presets */}
                  <div className="confidence-presets">
                    <span className="presets-caption">Presets:</span>
                    {[
                      { label: '10% Static (Universal)', val: 0.10, desc: 'Production recommended for all angles' },
                      { label: '12% Moderate', val: 0.12, desc: 'Slightly higher filter' },
                      { label: '15% Daytime', val: 0.15, desc: 'High daylight contrast' },
                      { label: '20% Strict', val: 0.20, desc: 'Ultra high precision' },
                    ].map((p) => (
                      <button
                        key={p.val}
                        type="button"
                        className={`preset-pill ${Math.abs(confThreshold - p.val) < 0.005 ? 'active' : ''}`}
                        onClick={() => {
                          setConfThreshold(p.val);
                          if (activeImageFile && !loading) {
                            handleRecount(p.val);
                          }
                        }}
                        title={p.desc}
                        disabled={loading}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="confidence-action-col">
              <button
                type="button"
                className={`btn btn-tuning-toggle ${showManualTuning ? 'active' : ''}`}
                onClick={() => setShowManualTuning(!showManualTuning)}
                title="Toggle manual slider override"
              >
                <Sliders size={14} />
                <span>{showManualTuning ? 'Hide Slider' : 'Manual Override'}</span>
              </button>

              {showManualTuning && (
                <button
                  type="button"
                  className={`btn btn-count-action ${hasPendingThresholdChange ? 'btn-count-pulse' : ''}`}
                  onClick={() => handleRecount()}
                  disabled={loading}
                >
                  <Play size={15} fill="currentColor" />
                  <span>Recount ({Math.round(confThreshold * 100)}%)</span>
                </button>
              )}
            </div>
          </div>

          {/* Upload Dropzone (First Time / Empty State) */}
          {!imageResult && !loading && (
            <div
              className={`dropzone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={handleOpenFileDialog}
            >
              <div className="dropzone-icon">
                <Upload size={32} />
              </div>
              <h2 className="dropzone-title">Click to upload or drag &amp; drop photo</h2>
              <p className="dropzone-text">
                Supports JPG, PNG, WEBP, BMP, TIFF • Max size: 25 MB
              </p>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="loading-card">
              <div className="spinner" />
              <p className="loading-title">Detecting and counting pipes...</p>
              <p className="loading-subtitle">Applying anti-double and distractor suppression</p>
            </div>
          )}

          {/* Results: Stacked Layout */}
          {imageResult && !loading && (
            <div className="results-wrapper">
              {/* Stats Bar */}
              <div className="stats-grid">
                <div className="stat-card primary-stat">
                  <span className="stat-label">Total Count</span>
                  <span className="stat-number">{imageResult.count}</span>
                </div>

                <div className="stat-card">
                  <span className="stat-label">Model Architecture</span>
                  <span className="stat-number stat-badge-text">
                    {imageResult.metrics?.model_profile === 'yolov8n'
                      ? 'YOLOv8n (640px)'
                      : 'YOLOv8m (1024px)'}
                  </span>
                </div>

                <div className="stat-card">
                  <span className="stat-label">Processed Time</span>
                  <span className="stat-number">
                    {imageResult.metrics?.inference_time_ms} ms
                  </span>
                </div>

                <div className="stat-card">
                  <span className="stat-label">Confidence</span>
                  <span className="stat-number">
                    {Math.round((imageResult.metrics?.conf_threshold || appliedConfThreshold) * 100)}%
                  </span>
                </div>

                <div className="stat-card actions-stat">
                  <span className="stat-label">Actions</span>
                  <div className="actions-buttons">
                    <button type="button" className="btn btn-primary" onClick={handleDownload}>
                      <Download size={14} />
                      <span>Download Image</span>
                    </button>
                    <button type="button" className="btn btn-outline" onClick={handleExportCSV}>
                      <FileSpreadsheet size={14} />
                      <span>Export CSV</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* 1. Original Input */}
              <section className="image-section">
                <div className="section-header">
                  <h3 className="section-title">1. Original Image</h3>
                  <span className="section-badge">{activeImageFile?.name || 'Input Source'}</span>
                </div>
                <div className="image-frame">
                  <img src={originalImageSrc} alt="Original input" className="view-image" />
                </div>
              </section>

              {/* 2. Detection Result */}
              <section className="image-section detected-section">
                <div className="section-header">
                  <div className="header-flex">
                    <h3 className="section-title">2. Detection Result</h3>
                    <span className="count-pill">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '6px' }}>
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                        <polyline points="22 4 12 14.01 9 11.01" />
                      </svg>
                      {imageResult.count} Pipes Counted
                    </span>
                  </div>
                </div>

                <div className="image-frame">
                  <img
                    src={imageResult.annotated_image}
                    alt={`Detected ${imageResult.count} pipes`}
                    className="view-image"
                  />
                </div>
              </section>

              {/* Coordinates Accordion with Confidence column */}
              <div className="table-accordion">
                <button
                  type="button"
                  className="accordion-toggle"
                  onClick={() => setShowTable(!showTable)}
                >
                  <span>Pipe Bounding Boxes ({imageResult.detections.length} items)</span>
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
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {imageResult.detections.map((d) => (
                          <tr key={d.id}>
                            <td>{d.id}</td>
                            <td>[{d.x1}, {d.y1}, {d.x2}, {d.y2}]</td>
                            <td>{d.width}px</td>
                            <td>{d.height}px</td>
                            <td>
                              <span className="conf-pill">
                                {d.conf !== undefined ? `${(d.conf * 100).toFixed(1)}%` : '—'}
                              </span>
                            </td>
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
