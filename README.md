# Flappy Bird Clone

A simple clone of the classic Flappy Bird game built with Python and Pygame.

## 🎮 Features

- Classic Flappy Bird gameplay mechanics  
- Pixel art-style graphics  
- Restart functionality after game over  
- Smooth animations and physics using Pygame

## 🖼️ Assets

The game uses the following assets:

- `bg.png` – background image  
- `ground.png` – ground/base  
- `pipe.png` – pipe obstacles  
- `bird1.png`, `bird2.png`, `bird3.png` – bird animation frames  
- `restart.png` – restart button image

## 🧠 How It Works

- The bird continuously falls due to gravity.
- Press mouse button to make the bird flap upward.
- Avoid the pipes — colliding with them or the ground ends the game.
- Click the restart button to play again.

## ▶️ How to Run

1. **Install requirements** (you’ll need Python and Pygame):

    ```bash
    pip install pygame
    ```

2. **Run the game**:

    ```bash
    python main.py
    ```

Make sure all asset files (`.png`) are in the same directory as `main.py`.

## 🛠️ Requirements

- Python 3.x  
- Pygame (tested with version >=2.0)

## 📝 Notes

- `temp.txt` is used for temporary or debug information (if applicable).
- Tested on Linux and Windows.
