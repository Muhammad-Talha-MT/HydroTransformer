# HydroTransformer

HydroTransformer is a spatio-temporal Transformer architecture for **daily streamflow prediction** that explicitly fuses **high-resolution distributed meteorological forcings** with **spatially distributed static basin attributes**.  
The model is designed to generalize robustly across both **time (future years)** and **space (ungauged basins)** without basin-specific calibration.

This repository contains the official implementation accompanying the paper:

> **From Aggregated to Distributed: A Transformer Framework for Generalizable Streamflow Modeling Across Space and Time**

---

## 🌍 Motivation

Estimating daily streamflow in ungauged basins remains a central challenge in hydrology due to strong spatial heterogeneity in meteorological forcing, basin structure, soils, and land cover.  
Most existing deep-learning approaches rely on **basin-averaged forcings** and **aggregated static attributes**, limiting their ability to capture within-basin variability and to generalize across watersheds.

HydroTransformer addresses these limitations by:
- Using **1 km × 1 km distributed meteorological inputs**
- Treating **static basin attributes as spatial rasters**
- Explicitly fusing static and dynamic information using **FiLM conditioning**
- Modeling temporal dependencies with a **Transformer-based temporal encoder**

---

## 🧠 Model Overview

HydroTransformer consists of three main components:

### 1. Spatial Encoding of Dynamic Forcings
Daily meteorological fields are patch-embedded and processed with a shared **Vision Transformer (ViT)** to capture spatial heterogeneity within each basin.

### 2. Static Attribute Encoding and FiLM Conditioning
Static basin attributes (e.g., elevation, soil texture, available water content) are encoded as spatial rasters.  
A **Feature-wise Linear Modulation (FiLM)** layer generates modulation parameters that condition the dynamic embeddings, allowing basin-specific hydrologic responses.

### 3. Temporal Transformer
FiLM-conditioned daily embeddings are passed through a **temporal Transformer encoder** to capture short-, medium-, and long-term hydrologic dependencies (5, 21, and 365-day windows).

The final temporal embedding is mapped to a daily streamflow estimate.

---

## 📊 Data Sources

### Dynamic Meteorological Inputs
- **Daymet V4** (1 km resolution, daily)
  - Precipitation
  - Minimum temperature
  - Maximum temperature
  - Shortwave radiation
  - Vapor pressure
  - Daylength
  - Snow water equivalent (SWE)

### Static Basin Attributes
Derived from **HydroATLAS / HydroSHEDS**, resampled to Daymet resolution:
- Digital Elevation Model (DEM)
- Soil texture (sand, silt, clay)
- Available water content (AWC)
- Field capacity (FC)

### Streamflow Observations
- **USGS daily mean discharge**
- Area-normalized and log-transformed

---

## 🧪 Experimental Design

HydroTransformer is evaluated under two complementary generalization settings:

### Temporal Generalization
- Train: 2000–2013
- Test: 2014–2020
- Same basins, future years

### Spatial Generalization (Ungauged Basins)
- Train: 47 watersheds
- Test: 12 unseen watersheds
- Same time period

### Metrics
- Nash–Sutcliffe Efficiency (NSE)
- Kling–Gupta Efficiency (KGE)

---

## 📈 Key Findings

- Consistently outperforms **LSTM**, **EA-LSTM**, and **CNN-LSTM** baselines
- Achieves higher median NSE/KGE in both temporal and spatial (ungauged) splits
- Requires **only six spatial static variables**, compared to 153 aggregated attributes used by LSTM baselines
- Demonstrates the **first daily cross-basin streamflow estimation** using fully distributed forcings with explicit static–dynamic fusion

---

## 📁 Repository Structure

```
HydroTransformer/
├── preprocessing/
├── src/
│   ├── data_loader_spatial.py
│   ├── data_loader_temporal.py
│   ├── training_spatial.py
│   ├── training_spatial_film.py
│   ├── training_temporal.py
│   ├── training_temporal_film.py
│   ├── training_ddp.py
│   ├── evaluation_film.py
│   ├── verify_data.py
│   └── utils/
│       └── h5_to_csv.py
├── config.yaml
├── environment.yml
└── README.md
```

---

## 🚀 Usage (High-Level)

### 1. Preprocess Data
```bash
python preprocessing/prepare_data.py
```

### 2. Train Model
```bash
python src/training_spatial_film.py --config config.yaml
```

### 3. Evaluate
```bash
python src/evaluation_film.py --checkpoint path/to/checkpoint.pt
```

---

## 📦 Requirements

See `requirements.txt` or `environment.yml` for dependencies.

---

## 📄 Citation

If you use this code, please cite:

```
Talha, M., Liu, X., Rasheed, R., & Nejadhashemi, A. P.
From Aggregated to Distributed: A Transformer Framework for Generalizable Streamflow Modeling Across Space and Time.
```

---

## ⚠️ Notes

- Large raster datasets (Daymet, HydroATLAS) are **not included**.
- Users must download and preprocess data independently.
- This repository is intended for **research and reproducibility**, not production deployment.
