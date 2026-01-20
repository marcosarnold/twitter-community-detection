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

## Setup

1. Clone the repository:
```bash
git clone https://github.com/marcosarnold/twitter-community-detection.git
cd twitter-community-detection
'''
  
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
  
3. Run the game:
You can start the game in different modes. For example:

**Human-controlled (no obstacles)**:
```bash
python snake_game.py human 0
```
  
**Greedy AI with 10 obstacles**:

```bash
python snake_game.py greedy 10
```
Replace `greedy` with `rand`, `dijkstra`, or `astar` to test different AI strategies, and replace the integer with the number of obstacles you want to include on the board.
  
4. Controls (for `human`):

Use arrow keys or WASD to control the snake's direction. The game does not allow wrap-around (if the snake hits the edge of the screen, the game will end). The game will also end if the snake runs into its own body or any obstacles placed on the board.

**Notes**:

The `CLOCK_SPEED` variable in `snake_game.py` controls how fast the game runs.  

Higher values make AI testing faster; lower values are better for human play.
  
## Technologies Used

-  **Python 3.10+**
-  **Pygame** – For game logic and visualization
-  **NumPy** – For grid and state manipulation
