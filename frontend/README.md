# Vision-intelligence — React Frontend

The client-side web application for the Insulation Pipe Detection system, built with **React 19** and **Vite**. Features a clean industrial dashboard interface with drag-and-drop photo upload, instant pipe counting, dynamic AI engine switching, and CSV/image export.

---

## 🚀 Run Commands

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start Local Development Server
```bash
npm run dev
```
- **Local URL:** `http://localhost:5173`
- Automatically proxies all `/api/*` requests to the FastAPI backend running on `http://127.0.0.1:8000`.

### 3. Build for Production
```bash
npm run build
```
Generates optimized static assets in the `dist/` folder.

### 4. Preview Production Build
```bash
npm run preview
```
Spins up a local web server to preview the production build in `dist/`.

---

## 🌟 Dashboard Features

- **Photo Pipe Counting**: Focused exclusively on high-precision photo counting for packaging cartons, bundles, and carts.
- **AI Engine Switcher**: Interactive topbar pills to toggle between **YOLOv8m (1024px)** and **YOLOv8n (640px)** on the fly.
- **Static Production Calibration**: Default 10% locked threshold that automatically suppresses cardboard vent holes and floor wheels with zero manual slider tuning.
- **Stacked Viewport**: Displays original high-res photo on top and detection result with bounding boxes directly below.
- **Data Ledger**: Expandable table of pipe coordinates, dimensions, and confidence scores.
- **One-Click Export**: Instant CSV export for ERP systems (SAP, NetSuite) and full-resolution annotated image download.

---

## ⚙️ Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `""` (empty) | Base URL for the backend API. When blank, requests use relative paths (`/api/...`), routed via Vite's local dev proxy or Vercel's edge rewrites. |

---

## 📁 Key Files

- `src/App.jsx` — Main application component (Dashboard, drag-and-drop upload, model switcher, stacked image results).
- `src/App.css` — Responsive industrial dashboard stylesheet.
- `vite.config.js` — Vite configuration with `/api` reverse proxy to port 8000.
