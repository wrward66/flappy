## Flappy Bird AI with Genetic Algorithm

## Project Overview
This project implements an AI-powered Flappy Bird game using a neural network and genetic algorithm to train birds to navigate through pipes. The system uses an evolutionary approach where birds learn to play the game through successive generations of selection, crossover, and mutation.

## Technical Architecture
## Neural Network Implementation
The AI uses a feedforward neural network with the following architecture:
Input layer: 7 nodes processing game state information
Hidden layer: 10 nodes with ReLU activation function
Output layer: 1 node with sigmoid activation determining flap decisions

## The network processes these inputs:
Normalized bird vertical position
Normalized next gap center position
Tanh-normalized vertical velocity
Normalized distance to next pipe
Tanh-normalized vertical distance from gap center
Normalized time since last flap
Tanh-normalized predicted future position relative to gap

## Genetic Algorithm Components
The evolutionary system includes:
Fitness-proportionate selection using roulette wheel method
Uniform crossover between parent neural networks
Controlled mutation with adjustable rate and strength parameters
Elite preservation mechanism to maintain successful traits
Adaptive mutation based on generation diversity

## Fitness Evaluation
Birds are evaluated using a multi-factor fitness function that considers:
Survival time (frames alive)
Pipes cleared (with exponential reward scaling)
Gap centering accuracy
Progress toward next pipe
Boundary avoidance penalties

## Game Configuration
The game has been calibrated for optimal AI learning with these parameters:
Pipe gap width: 160 pixels
Pipe spawn frequency: 1700 milliseconds
Scroll speed: 4.5 pixels per frame
Gravity: 0.5 pixels per frame squared
Flap strength: -9.8 pixels per frame

## Installation and Execution
Clone the repository to your local machine
Ensure Python 3.x is installed with Pygame and NumPy libraries
Run the main Python script
The program includes fallback graphics generation if image files are not available.

## Training Process
The AI training follows this iterative process:
Each generation consists of 150 birds with randomized neural networks
Birds are evaluated based on their in-game performance
The top 25 performers are preserved as elites for the next generation
Subsequent generations are created through selection, crossover, and mutation
Training continues indefinitely until manually terminated

## Real-time progress monitoring displays:
Current generation number
Best fitness score achieved
Highest pipes cleared count
Average performance metrics
Visual progress indicators

## Customization Options
Key parameters can be modified to adjust difficulty and learning characteristics:
pipe_gap: Adjust the width between top and bottom pipes
pipe_frequency: Modify time between pipe spawns
scroll_speed: Change horizontal movement speed
Gravity and flap strength values in the Bird class

## Performance Tracking
The system maintains comprehensive performance metrics:
Per-generation statistics
Active bird count during simulation
Current generation high score
All-time performance record
Learning progression trends across generations

## Dependencies
Python 3.x
Pygame library
NumPy library

## Technical Significance
This implementation demonstrates practical application of genetic algorithms for training neural networks to solve navigation challenges in game environments. The system shows clear improvement over successive generations through selective reproduction and mutation.

The project provides a foundation for further experimentation with different network architectures, selection strategies, and game parameters, showcasing machine learning concepts in an accessible game development context.


