import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Download,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle,
  Eye,
  ImageIcon,
  RotateCcw,
  Sliders,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://127.0.0.1:8000';

const PRESETS = [
  { id: 'grid', label: 'Dense Grid (70 Pipes)', conf: 0.35 },
  { id: 'carton', label: 'Warehouse Carton', conf: 0.58 },
  { id: 'stack', label: 'Standard Stack', conf: 0.45 },
  { id: 'auto', label: 'Auto (Default)', conf: 0.35 },
];

export default function App() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [samples, setSamples] = useState([]);
  const [activePreset, setActivePreset] = useState('grid');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Current session data
  const [originalImageSrc, setOriginalImageSrc] = useState(null);
  const [activeFile, setActiveFile] = useState(null);
  const [activeSampleId, setActiveSampleId] = useState(null);
  const [result, setResult] = useState(null);
  const [viewMode, setViewMode] = useState('detection'); // 'detection' | 'original'
  const [showTable, setShowTable] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef(null);

  // Check backend health on load
  useEffect(() => {
    checkHealth();
    fetchSamples();
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
      if (res.ok) setBackendOnline(true);
      else setBackendOnline(false);
    } catch {
      setBackendOnline(false);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data);
      }
    } catch (e) {
      console.error('Could not fetch samples:', e);
    }
  };

  // Run detection on File
  const processFile = async (file, presetId = activePreset) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setActiveFile(file);
    setActiveSampleId(null);

    // Read original image preview
    const reader = new FileReader();
    reader.onload = (e) => setOriginalImageSrc(e.target.result);
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('preset', presetId);

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
      setViewMode('detection');
    } catch (err) {
      setError(`Detection failed: ${err.message}. Is backend running?`);
    } finally {
      setLoading(false);
    }
  };

  // Run detection on Sample
  const processSample = async (sampleId, presetId = null) => {
    setLoading(true);
    setError(null);
    setActiveFile(null);
    setActiveSampleId(sampleId);

    const targetPreset = presetId || activePreset;

    // Load original image preview
    setOriginalImageSrc(`${API_BASE}/api/sample-image/${sampleId}`);

    const formData = new FormData();
    formData.append('sample_id', sampleId);
    formData.append('preset', targetPreset);

    try {
      const res = await fetch(`${API_BASE}/api/detect-sample`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setViewMode('detection');
    } catch (err) {
      setError(`Sample detection failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle Preset Change
  const handlePresetChange = (newPreset) => {
    setActivePreset(newPreset);
    if (activeFile) {
      processFile(activeFile, newPreset);
    } else if (activeSampleId) {
      processSample(activeSampleId, newPreset);
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

  // Download annotated image
  const handleDownloadImage = () => {
    if (!result?.annotated_image) return;
    const link = document.createElement('a');
    link.href = result.annotated_image;
    link.download = `pipe_count_${result.count}_annotated.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export CSV
  const handleExportCSV = () => {
    if (!result?.detections?.length) return;
    const headers = ['Pipe_ID', 'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height', 'Confidence'];
    const rows = result.detections.map((d) => [
      d.id,
      d.x1,
      d.y1,
      d.x2,
      d.y2,
      d.width,
      d.height,
      (d.conf * 100).toFixed(1) + '%',
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

  // Reset
  const handleReset = () => {
    setResult(null);
    setOriginalImageSrc(null);
    setActiveFile(null);
    setActiveSampleId(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="header">
        <div className="header-brand">
          <span className="brand-title">PipeCount</span>
          <span className="brand-subtitle">Thermal Insulation Pipe Vision</span>
        </div>

        <div className="header-right">
          <div className="status-indicator">
            <span
              className={`status-dot ${backendOnline ? 'online' : 'offline'}`}
            ></span>
            <span className="status-text">
              {backendOnline ? 'Backend Online (Port 8000)' : 'Backend Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="main-content">
        {/* Controls Toolbar */}
        <section className="toolbar">
          <div className="toolbar-group">
            <label className="toolbar-label">Preset Mode:</label>
            <select
              value={activePreset}
              onChange={(e) => handlePresetChange(e.target.value)}
              className="preset-select"
              disabled={loading}
            >
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          <div className="toolbar-group samples-group">
            <span className="toolbar-label">Demo Samples:</span>
            <div className="sample-pills">
              {samples.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => processSample(s.id, s.preset)}
                  className={`sample-pill ${activeSampleId === s.id ? 'active' : ''}`}
                  disabled={loading}
                >
                  {s.name}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Error Alert */}
        {error && (
          <div className="alert alert-error">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* Content View: Dropzone or Result */}
        {!result && !loading ? (
          <section
            className={`dropzone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="file-input-hidden"
              accept=".jpg,.jpeg,.jfif,.gfif,.png,.webp,.bmp,.tiff,.tif,.gif"
              onChange={handleFileInput}
            />
            <div className="dropzone-icon">
              <Upload size={36} />
            </div>
            <h3 className="dropzone-title">Click to upload or drag & drop</h3>
            <p className="dropzone-desc">
              Supports all image formats: <strong>JPG, PNG, WEBP, JFIF, GFIF, BMP, TIFF, GIF</strong>
            </p>
          </section>
        ) : loading ? (
          <div className="loading-card">
            <div className="spinner"></div>
            <p className="loading-text">Running pipe detection neural network...</p>
          </div>
        ) : (
          /* Results View: Original on Top, Detected with Counting Below */
          <section className="result-view">
            {/* 1. Original Uploaded Image (First) */}
            <div className="image-card">
              <div className="card-header">
                <div className="card-title-group">
                  <h3 className="card-title">1. Original Image</h3>
                  <span className="card-meta">
                    {activeFile ? activeFile.name : 'Input Photo'}
                  </span>
                </div>

                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={handleReset}
                  title="Upload another image"
                >
                  <RotateCcw size={14} />
                  <span>Upload Different Image</span>
                </button>
              </div>

              <div className="image-frame">
                <img
                  src={originalImageSrc}
                  alt="Original Uploaded Pipe Photo"
                  className="stage-img"
                />
              </div>
            </div>

            {/* 2. Detected Image with Counting (Directly Below) */}
            <div className="image-card detection-card">
              <div className="card-header">
                <div className="count-highlight">
                  <span className="count-number">{result.count}</span>
                  <div className="count-text-group">
                    <span className="count-title">Pipes Counted</span>
                    <span className="count-meta">
                      Avg Conf: {(result.metrics.avg_conf * 100).toFixed(1)}% · Inference: {result.metrics.inference_time_ms} ms
                    </span>
                  </div>
                </div>

                <div className="card-actions">
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleDownloadImage}
                    title="Download image with pipe bounding boxes"
                  >
                    <Download size={15} />
                    <span>Download Image</span>
                  </button>

                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleExportCSV}
                    title="Export detection coordinates to CSV"
                  >
                    <FileSpreadsheet size={15} />
                    <span>Export CSV</span>
                  </button>
                </div>
              </div>

              <div className="image-frame">
                <img
                  src={result.annotated_image}
                  alt={`Detection Result: ${result.count} Pipes`}
                  className="stage-img"
                />
              </div>
            </div>

            {/* Coordinates Table Accordion */}
            <div className="table-accordion">
              <button
                type="button"
                className="table-accordion-toggle"
                onClick={() => setShowTable(!showTable)}
              >
                <span>Pipe Coordinates ({result.detections.length} items)</span>
                {showTable ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>

              {showTable && (
                <div className="table-wrapper">
                  <table className="coords-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>X1</th>
                        <th>Y1</th>
                        <th>X2</th>
                        <th>Y2</th>
                        <th>Width</th>
                        <th>Height</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.detections.map((d) => (
                        <tr key={d.id}>
                          <td>{d.id}</td>
                          <td>{d.x1}</td>
                          <td>{d.y1}</td>
                          <td>{d.x2}</td>
                          <td>{d.y2}</td>
                          <td>{d.width}px</td>
                          <td>{d.height}px</td>
                          <td>{(d.conf * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
