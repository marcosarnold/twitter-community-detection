# Twitter Community Detection

This project explores how **community detection algorithms** uncover structure in real-world social networks.  It was developed as part of coursework and independent experimentation in Data Structures and Algorithms at Colorado College.

## Overview

The project analyzes **Twitter ego-network graphs** using both **from-scratch implementations** and **official library implementations** of community detection algorithms.

The goal is to understand:
- how different quality functions (modularity vs. map equation) behave
- how algorithm design choices affect detected communities
- how media-centered Twitter networks cluster

Algorithms are applied to **ego networks of major news accounts**, and results are visualized and compared.

## Features

- Community detection on real Twitter graph data
- Scratch implementations of:
  - **Infomap**
  - **Leiden Algorithm**
- Comparisons against official implementations:
  - `infomap`
  - `leidenalg` via `igraph`
- Ego-network extraction from large datasets
- Community visualization with labeled high-degree users
- Modularity and runtime analysis
- Optional Twitter ID → handle resolution

## Setup

1. Clone the repository:
```bash
git clone https://github.com/marcosarnold/twitter-community-detection.git
cd twitter-community-detection
```
  
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
  

## Running Experiments

All experiments should be run from the project root using Python’s module mode.

**Infomap Algorithm**:
```bash
python -m scratch_infomap.py
```

**Leiden Algorithm**:
```bash
python -m scratch_leiden.py
```

## Technologies Used

-  **Python 3.10+**
-  **gdown** – Downloads the SNAP Twitter dataset directly from Google Drive
-  **Networkx** – Constructs and manipulates large Twitter graphs and ego networks
-  **Matplotlib** – Visualizes graphs, communities, and annotated leaderboards
-  **Requests** – Fetches Twitter usernames from user IDs via external web requests
-  **python-igraph** – Provides efficient graph representations and layouts for large networks
-  **Infomap** – Flow-based community detection algorithm
-  **Leiden** – Modularity-based community detection with improved stability
