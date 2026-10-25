# 🛰️ SatQuery AI — Agentic Multimodal Satellite Intelligence

> **All-Weather Optical + SAR Fusion | Deterministic Rasterio Math | Autonomous TinyFish Web Grounding**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Interactive_Demo-Click_Here-brightgreen?style=for-the-badge)](https://your-live-demo-link.com)
[![Pitch Deck](https://img.shields.io/badge/📊_Pitch_Deck_PPT-View_Slides-blue?style=for-the-badge)](https://your-ppt-link.com)
[![Demo Video](https://img.shields.io/badge/🎥_Working_Demo_Video-Watch_Now-red?style=for-the-badge)](https://your-video-link.com)

---

## 📖 Overview
**SatQuery AI** is an agentic vision-language assistant for multimodal remote-sensing satellite image analysis. Instead of requiring heavy GIS software (like QGIS/ArcGIS) and hours of manual processing by geospatial scientists, SatQuery AI allows any decision-maker or rescue officer to select a region on an interactive map, ask questions in natural language (English/Hindi), and receive mathematically verified spatial measurements, segmentation masks, and live ground-truth intelligence in seconds.

---

## 🎯 The Problem We Solve
1. **Cloud & Weather Blindness:** Standard optical satellites (Sentinel-2) fail completely during floods, cyclones, night-time, or heavy cloud cover—precisely when disaster intelligence is needed most.
2. **GIS Complexity:** Raw `GeoTIFF` rasters are massive (500MB–2GB+) with embedded Coordinate Reference Systems (CRS) and multi-band arrays that non-experts cannot open or analyze.
3. **LLM Hallucination:** Standard Vision-Language Models (VLMs) guess spatial areas from simple PNG images and cannot calculate real-world square kilometers or exact Lat-Long coordinates.

---

## ⚡ Our Solution & Core USP
* **1. All-Weather Optical + SAR Fusion:** Co-registers **Sentinel-2 (Optical RGB/NIR)** with **Sentinel-1 (C-Band Synthetic Aperture Radar)** at the pixel level. Microwaves penetrate clouds, smoke, and darkness to detect floodwater (specular dark backscatter) and infrastructure (bright double-bounce reflections).
* **2. Zero-Hallucination Agentic Tools:** Our LLM orchestrator acts as a router—it delegates all counting and area math to deterministic Python tools (`Rasterio`, `GDAL`, `NumPy`), converting segmented pixel masks into 100% accurate square kilometers and geographic coordinates.
* **3. TinyFish Autonomous Web Grounding:** Satellite imagery shows *what* happened from above, while our integrated **TinyFish API Web Agent** autonomously scrapes live regional disaster bulletins, affected highways, hospitals, and local news for those exact coordinates.

---

## 🏗️ System Architecture & Workflow

```text
User Natural Language Query + Map Bounding Box (ROI)
                         │
                         ▼
       FastAPI Backend Orchestrator (ReAct Agent)
         ├───► [Tool 1: Optical + SAR Fusion Engine]
         │     (Co-registers Sentinel-1 VV/VH & Sentinel-2 Bands + Segmentation)
         │
         ├───► [Tool 2: Deterministic Rasterio / GDAL Math]
         │     (Extracts CRS Metadata, Pixel-to-Sq-Km Area & Center Lat/Long)
         │
         └───► [Tool 3: TinyFish Autonomous Web Agent]
               (Scrapes Live Disaster Alerts, Highways & Local Infrastructure)
                         │
                         ▼
   Interactive GeoJSON Map Overlay + Verified Actionable Report
```

---

## 🛠️ Tech Stack
* **Frontend:** React.js, Next.js, Tailwind CSS, Leaflet.js / Mapbox GL (Bounding Box ROI selector & Optical/SAR/Mask Layer Toggle)
* **Backend & Orchestration:** Python, FastAPI, Uvicorn, Function-Calling ReAct Agent Pipeline
* **Geospatial & Computer Vision:** `Rasterio`, `GDAL`, `GeoPandas`, `Shapely`, `NumPy`, `OpenCV`, SAR Backscatter Thresholding & Hybrid Vision Segmentation
* **Data & Live Web Grounding:** **TinyFish API** (Headless DOM Extraction & Ground-Truth Agent), Open-Source STAC Catalogs (Sentinel-1 & Sentinel-2)

---

## 🚀 Quick Start (Run on Your Device in 2 Minutes)

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/SatQuery-AI.git](https://github.com/YOUR_USERNAME/SatQuery-AI.git)
cd SatQuery-AI
```

### 2. Start the Backend (FastAPI + Geospatial Agent)
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
*(Note: Pre-cropped sample `.tif` files and resilient Geo-Caching fallbacks are included out-of-the-box so evaluators can test the pipeline immediately without configuring external API keys.)*

### 3. Start the Frontend (Interactive Map UI)
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## 📈 Scalability & Real-World Viability
* **Cloud-Optimized GeoTIFF (COG) Streaming:** Streams only the byte-range of the user's selected bounding box instead of downloading full 2GB rasters.
* **PostGIS Spatial Caching:** Caches segmented disaster masks so subsequent queries on the same flood/landslide zone return in `<0.2s` at zero GPU cost.
* **Target Sectors:** 
  * **B2G:** NDRF, State Disaster Management Authorities (SDMAs), Forest Departments, and Defense Border Surveillance.
  * **B2B SaaS (Pay-per-Sq-Km):** Crop Insurance (Fasal Bima) claim verification, Infrastructure monitoring, and Climate-Risk analytics.

---

## 👥 Team
Built with precision for real-time geospatial intelligence.
