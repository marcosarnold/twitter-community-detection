# Twitter Community Detection

This project explores how **community detection algorithms** uncover structure in real-world social networks.  
It was developed as part of coursework and independent experimentation in **Data Structures and Algorithms** at Colorado College.

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
  - **Infomap (Map Equation)**
  - **Leiden Algorithm**
- Comparisons against official implementations:
  - `infomap`
  - `leidenalg` via `igraph`
- Ego-network extraction from large datasets
- Community visualization with labeled high-degree users
- Modularity and runtime analysis
- Optional Twitter ID → handle resolution
