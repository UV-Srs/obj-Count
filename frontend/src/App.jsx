import React, { useState, useRef } from 'react';
import {
  LayoutDashboard,
  Upload,
  Download,
  FileSpreadsheet,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Layers,
} from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Active session data
  const [originalImageSrc, setOriginalImageSrc] = useState(null);
  const [activeFile, setActiveFile] = useState(null);
  const [result, setResult] = useState(null);
  const [showTable, setShowTable] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef(null);

  // Directly open native OS file manager
  const handleOpenFileDialog = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  // Process uploaded image file
  const processFile = async (file) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setActiveFile(file);

    // Read local preview for immediate display
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
      setResult(data);
    } catch (err) {
      setError(`Detection failed: ${err.message}`);
    } finally {
      setLoading(false);
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
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  // Download annotated result
  const handleDownloadImage = () => {
    if (!result?.annotated_image) return;
    const link = document.createElement('a');
    link.href = result.annotated_image;
    link.download = `pipe_count_${result.count}_result.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export CSV (Pipe_ID, X1, Y1, X2, Y2, Width, Height)
  const handleExportCSV = () => {
    if (!result?.detections?.length) return;
    const headers = ['Pipe_ID', 'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height'];
    const rows = result.detections.map((d) => [
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
    link.download = `pipe_detections_${result.count}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="app-layout">
      {/* Side Navigation Bar - Only 1 item: Dashboard */}
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
          <button type="button" className="nav-item active">
            <LayoutDashboard size={18} />
            <span>Dashboard</span>
          </button>
        </nav>
      </aside>

      {/* Main Viewport */}
      <div className="main-viewport">
        <header className="topbar">
          <div>
            <h1 className="page-heading">Dashboard</h1>
            <p className="page-subheading">Upload a pipe bundle or box photo to calculate total count</p>
          </div>

          {result && (
            <button
              type="button"
              className="btn btn-upload-new"
              onClick={handleOpenFileDialog}
            >
              <Upload size={15} />
              <span>Upload New Image</span>
            </button>
          )}
        </header>

        <div className="page-body">
          {/* Persistent Hidden File Input for instant OS file manager trigger */}
          <input
            ref={fileInputRef}
            type="file"
            className="hidden-file-input"
            accept=".jpg,.jpeg,.jfif,.gfif,.png,.webp,.bmp,.tiff,.tif,.gif"
            onChange={handleFileInput}
          />

          {/* Error Message */}
          {error && (
            <div className="alert alert-error">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          {/* Upload Dropzone (First Time / Empty State) */}
          {!result && !loading && (
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
              <h2 className="dropzone-title">Click to upload or drag & drop</h2>
              <p className="dropzone-text">Supports JPG, PNG, WEBP, JFIF, GFIF, BMP, TIFF, GIF</p>
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

          {/* Results: Original on Top, Detected Below */}
          {result && !loading && (
            <div className="results-wrapper">
              {/* Stats Bar */}
              <div className="stats-grid">
                <div className="stat-card primary-stat">
                  <span className="stat-label">Total Count</span>
                  <span className="stat-number">{result.count}</span>
                </div>

                <div className="stat-card">
                  <span className="stat-label">Processed Time</span>
                  <span className="stat-number">{result.metrics.inference_time_ms} ms</span>
                </div>

                <div className="stat-card actions-stat">
                  <span className="stat-label">Actions</span>
                  <div className="actions-buttons">
                    <button type="button" className="btn btn-primary" onClick={handleDownloadImage}>
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

              {/* 1. Original Image (First) */}
              <section className="image-section">
                <div className="section-header">
                  <h3 className="section-title">1. Original Image</h3>
                  <span className="section-badge">{activeFile?.name || 'Input Photo'}</span>
                </div>
                <div className="image-frame">
                  <img src={originalImageSrc} alt="Original input" className="view-image" />
                </div>
              </section>

              {/* 2. Detection Result (Directly below) */}
              <section className="image-section detected-section">
                <div className="section-header">
                  <div className="header-flex">
                    <h3 className="section-title">2. Detection Result</h3>
                    <span className="count-pill">{result.count} Pipes</span>
                  </div>
                </div>
                <div className="image-frame">
                  <img
                    src={result.annotated_image}
                    alt={`Detected ${result.count} pipes`}
                    className="view-image"
                  />
                </div>
              </section>

              {/* Coordinates Accordion without Confidence column */}
              <div className="table-accordion">
                <button
                  type="button"
                  className="accordion-toggle"
                  onClick={() => setShowTable(!showTable)}
                >
                  <span>Pipe Bounding Boxes ({result.detections.length} items)</span>
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
                        {result.detections.map((d) => (
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
