# DeepSimplex: Reinforcement Learning for TSP

Welcome to DeepSimplex, where we leverage cutting-edge reinforcement learning techniques to tackle the Traveling Salesman Problem (TSP). Our mission is to provide efficient solutions for TSP instances, particularly those with more than 200 edges, where traditional Linear Programming approaches become prohibitively time-consuming.

## Key Features

- **Deep Learning Models**: DeepSimplex employs Deep Q-Networks (DQN), Actor-Critic, and Graph Neural Embeddings + Q-network algorithms to address the TSP problem.

- **Fast Convergence**: Our models demonstrate remarkable performance by finding solutions for 5-edge TSP problems after just 100 training steps, highlighting the efficiency of our approach.

- **Scalable Solutions**: DeepSimplex becomes indispensable for TSP instances with a high edge count, offering a practical alternative to the computationally intensive Linear Programming methods commonly used for such scenarios.

- **Research Foundation**: Our project draws inspiration from the research paper ["Deep Reinforcement Learning for Solving the Traveling Salesman Problem"](https://openreview.net/forum?id=SkgvvCVtDS), which serves as the foundation for our work.

## Getting Started

To dive into DeepSimplex and start solving TSP problems using reinforcement learning, follow these steps:

1. **Clone the Repository**:
   ```sh
   https://github.com/Mrov9072/DeepSimplex-Reinforcement-Learning-for-TSP.git
   cd DeepSimplex-Reinforcement-Learning-for-TSP
   pip install -r requirements.txt
### For running agent with actor-critic run
python train_a2c.py
### For running agent with DQN run
python train_dqn.py
### For running agent with Graph_Embedding_DQN run
python train_graph_dqn.py