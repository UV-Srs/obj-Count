# PipeCount - React Frontend

The client-side web application for the Insulation Pipe Detection system, built with React 19 and Vite. Features a minimalist dashboard interface with drag-and-drop image upload, instant pipe counting, and CSV/image export.

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
- Proxies all `/api/*` requests to the FastAPI backend running on `http://127.0.0.1:8000`.

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

## ⚙️ Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `""` (empty) | Base URL for the backend API. When blank, requests use relative paths (`/api/...`), which are routed via Vite's local proxy or Vercel's edge rewrites. |

---

## 📁 Key Files

- `src/App.jsx` - Main application component (Dashboard, drag-and-drop upload, stacked image results).
- `src/App.css` - Minimalist industrial dashboard styles.
- `vite.config.js` - Vite configuration with `/api` reverse proxy to port 8000.
