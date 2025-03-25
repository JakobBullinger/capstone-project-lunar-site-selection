Lunar Site Selection Algorithm & Product Strategy Development
“We are planting the seeds today for tomorrow’s forests on the Moon.”
This repository contains the code and documentation for a Lunar Site Selection Algorithm—an MVP that identifies potential landing sites for a future “Forest on the Moon” mission. The project also outlines a broader framework to evolve into a universal space knowledge system, incorporating open science and collective action principles.

Table of Contents
Project Overview

Key Features

Repository Structure

Installation & Setup

Usage

Methodology

Roadmap

Contributing

License

Acknowledgments

Project Overview
Vision
Mission: To provide a data-driven, socially inclusive approach to selecting optimal landing sites for planting the first “forest” on the Moon—serving as a stepping stone toward a universal space knowledge system.

Why It Matters:

Ecological Complexity: Ecosystems are complex; planting anything on the Moon requires careful site analysis.

Data Explosion: NASA’s Planetary Data System (PDS) has 1850+ TB of data; harnessing this effectively requires advanced algorithms and open science collaboration.

Social Inclusion: The future of space exploration must be equitable, diverse, and collaborative.

High-Level Goals
MVP: Develop a Python-based site selection algorithm that uses lunar temperature data and other constraints (e.g., slope, roughness) to rank feasible planting locations.

Open Science Framework: Foster an inclusive community (via hackathons, citizen science apps) to continually improve and expand the MVP.

Long-Term Vision: Scale this approach to a universal knowledge system, supporting future planetary missions (Mars, interplanetary ecosystems) and Earth-based use cases (e.g., wind farm siting, real estate location analysis).

Key Features
Lunar Temperature Analysis: Aggregates 600+ million measurements of bolometric brightness temperatures from NASA’s Lunar Reconnaissance Orbiter.

Feasibility Scoring: Users can define custom constraints (min, max, average temperatures) to compute a “feasibility score” for each coordinate.

Dynamic Dashboard: A Dash-based web application (application.py) allows real-time updates of feasibility maps.

Extensible Architecture: The codebase is structured to incorporate additional data layers (e.g., slope, crater distribution, solar exposure) for more robust site selection.

Repository Structure
bash
Copy
capstone-project-lunar-site-selection/
│
├── Lunar_Site_Selection_MVP.ipynb  # Jupyter Notebook for data preprocessing & static visualization
├── Lunar_Site_Selection_MVP.py     # Python script equivalent of the MVP notebook
├── main_notebook.ipynb             # Additional analysis or final presentation notebook
├── application.py                  # Dash web application for dynamic visualization
├── deployment_map.py               # Helper script for generating map-based visuals
├── deployment_hist.py              # Helper script for generating histograms or distribution plots
├── requirements.txt                # Dependencies for the analysis and/or web app
├── .gitignore                      # Ignores venv, __pycache__, etc.
└── README.md                       # This file
Lunar_Site_Selection_MVP.ipynb / .py
Core code for loading, preprocessing, and analyzing lunar data (temperature, feasibility scoring).

application.py
Dash/Plotly web app that loads the preprocessed data and provides interactive sliders to update feasibility constraints.

main_notebook.ipynb
An additional or “final presentation” notebook. May consolidate results or demonstrate the pipeline in a more narrative form.

Installation & Setup
Clone the Repo:

bash
Copy
git clone https://github.com/JakobBullinger/capstone-project-lunar-site-selection.git
cd capstone-project-lunar-site-selection
Create & Activate a Virtual Environment (recommended):

bash
Copy
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install Dependencies:

bash
Copy
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Usage
1. Running the Static Analysis & Visualization
Open Lunar_Site_Selection_MVP.ipynb in Jupyter (or Lunar_Site_Selection_MVP.py in your IDE):

bash
Copy
jupyter notebook Lunar_Site_Selection_MVP.ipynb
Step through each cell to:

Load and preprocess the lunar temperature dataset.

Generate static heatmaps and feasibility maps.

Save any outputs (e.g., .png images).

2. Running the Interactive Dashboard
From your terminal (with the virtual environment activated), run:

bash
Copy
python application.py
Open your browser to the URL provided (e.g., http://127.0.0.1:8050).

Use the sliders to adjust temperature constraints; watch the feasibility map update in real time.

Methodology
Data Aggregation

We leverage NASA’s Planetary Data System (PDS), specifically the Lunar Reconnaissance Orbiter’s bolometric brightness temperatures.

Data are sampled and compressed to handle large volumes efficiently.

Analysis

Each coordinate is aggregated into min/max/avg temperature values.

A feasibility score is computed based on user-defined temperature constraints.

Visualization

Static: 2D heatmaps, histograms (via matplotlib, seaborn).

Interactive: Dash/Plotly web app, allowing parameter tuning and live map updates.

Social & Strategic Framework

The code and approach are embedded in a larger vision for open science, citizen science hackathons, and inclusive outreach to ensure broad participation in space exploration.

Roadmap
Short-Term (H1/H2)

Refine MVP: Add additional lunar data layers (e.g., slope, crater distribution).

Hackathons: Collaborate with organizations like Zindi Africa to crowdsource improvements.

Open Science: Host the code and data on platforms like Radiant Earth or similar.

Mid-Term (H3)

Universal Site Selection: Adapt the algorithm for Earth-based use cases (renewable energy siting, real estate, etc.).

IoT Integration: Explore sensors for plant growth (e.g., AstroPlant) to feed real-time data into the model.

Long-Term (H4)

Universal Space Knowledge System: Expand to a fully integrated platform that can handle multi-planet data sets, AI-driven optimization, and advanced ecosystem modeling.

Inclusive Governance: Work with organizations (NASA, ESA, citizen science groups) to ensure that the project remains globally accessible and socially equitable.

Contributing
We welcome contributions from both technical and non-technical collaborators:

Technical: Submit pull requests for bug fixes, new data layers, improved algorithms, or performance enhancements.

Social/Outreach: Help us design hackathons, citizen science initiatives, or D&I frameworks.

Ideas/Requests: Open an issue describing your proposal, question, or feedback.

License
This project is licensed under the MIT License (or whichever license you choose).
Please see the LICENSE file for details.

Acknowledgments
Spring Institute for inspiring the Forest on the Moon Mission and focusing on ecological + social synergy.

NASA’s Planetary Data System for providing the lunar temperature data used in this MVP.

Expert Interviews from a wide range of fields (space science, diversity & inclusion, citizen science, etc.) for guiding the vision.

Hackathon Platforms (Zindi Africa, Space Apps Challenge, etc.) for offering opportunities to further develop this project collaboratively.

Final Note
“It is almost impossible to understand the complexity of ecological processes in a system precisely manipulated in order to exclude uncontrolled disturbances.”
(Granjou, 2016)

Our Lunar Site Selection MVP is a first step toward understanding—and shaping—those complexities in the harsh environment of the Moon. By uniting data science, social inclusion, and open collaboration, we aim to push the boundaries of both space exploration and earthly environmental solutions.
