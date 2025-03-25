# Lunar Site Selection & Open Science Framework

This repository hosts our **Lunar Site Selection MVP**—a data-driven algorithm designed to select optimal landing sites for planting the first “forest” on the Moon. Our approach serves as a stepping stone toward building a universal space knowledge system, embracing open science and social inclusion.

---

## Project Overview

### Vision & Mission

- **Mission**:  
  To provide a data-driven, socially inclusive approach to selecting optimal landing sites for planting the first “forest” on the Moon—paving the way for a universal space knowledge system.

- **Why It Matters**:  
  - **Ecological Complexity**: Ecosystems are inherently complex; planting on the Moon requires precise and careful site analysis.  
  - **Data Explosion**: NASA’s Planetary Data System (PDS) holds over **1850 TB** of data. Harnessing this data effectively demands advanced algorithms and open science collaboration.  
  - **Social Inclusion**: The future of space exploration must be equitable, diverse, and collaborative, ensuring that all voices contribute to the vision.

### High-Level Goals

- **MVP**:  
  Develop a Python-based site selection algorithm that leverages lunar temperature data (and other constraints like slope and roughness) to rank feasible planting locations.

- **Open Science Framework**:  
  Foster an inclusive community through hackathons, citizen science apps, and collaborative outreach to continuously improve and expand the MVP.

- **Long-Term Vision**:  
  Scale the approach to create a universal knowledge system that supports future planetary missions (e.g., Mars, interplanetary ecosystems) and Earth-based use cases (e.g., wind farm siting, real estate analysis).

---

## Methodology

### Data Aggregation

- **Source**:  
  Leverage NASA’s Planetary Data System (PDS), specifically using the Lunar Reconnaissance Orbiter’s bolometric brightness temperature data.
- **Processing**:  
  Data are sampled and compressed to efficiently handle the large volumes available.

### Analysis

- **Aggregation**:  
  Each coordinate is aggregated into minimum, maximum, and average temperature values.
- **Feasibility Score**:  
  A score is computed based on user-defined temperature constraints to rank each potential site.

### Visualization

- **Static Visuals**:  
  Generate 2D heatmaps and histograms using libraries like `matplotlib` and `seaborn`.
- **Interactive Dashboard**:  
  A Dash/Plotly web app provides real-time updates, allowing users to adjust parameters and view live map updates.

### Social & Strategic Framework

- **Open Science & Outreach**:  
  Our codebase is integrated into a larger vision that includes citizen science hackathons and inclusive outreach, ensuring broad participation in space exploration.
- **Future Integration**:  
  The framework is designed to evolve, supporting additional data layers and further applications both in space missions and Earth-based scenarios.

---

## Repository Structure

- **Lunar_Site_Selection_MVP.ipynb / .py**: Core analysis and visualization code.
- **application.py**: Interactive Dash web app.
- **deployment_map.py / deployment_hist.py**: Helper scripts for maps and histograms.
- **requirements.txt**: Python dependencies.
- **.gitignore**: Excludes virtual environments, caches, and temporary files.

---
