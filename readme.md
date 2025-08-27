# Project: Ares Prime

## Grand Goal (The End State)
The ultimate objective is to develop a self-sustaining, multi-agent colony in the Ares Prime Simulation. The colony should demonstrate emergent behavior, such as role specialization and autonomous expansion (building new agents), without being explicitly programmed for these tasks.

---

## 🚀 Project Roadmap

### Phase 1: The Lone Surveyor (Foundation)
* **Objective**: Train a single agent to master survival, resource gathering, and basic construction.
* **Completed When**: A single agent can reliably build a basic power grid (solar panel + battery) to sustain itself.

### Phase 2: The First Outpost (Collaboration)
* **Objective**: Train multiple agents to coordinate and deconflict in a shared environment.
* **Completed When**: A small team of agents can cooperatively build a complex structure (e.g., a refinery) more efficiently than a single agent could alone.

### Phase 3: The Thriving Colony (Complexity)
* **Objective**: Achieve long-term planning and emergent, complex behaviors from the agent population.
* **Completed When**: The colony can successfully build a Fabricator and use it to construct a new agent, demonstrating autonomous expansion.

---

## 📊 Current Status
The project is currently in **Phase 1: The Lone Surveyor**.

**Milestones Achieved:**
* [x] **Basic Navigation**: Agent can solve a 10x10 grid.
* [x] **Complex Pathfinding**: Agent can navigate around obstacles and complete a multi-stage "collect-then-deliver" task.
* [x] **Survival Constraint**: Agent can successfully manage a finite energy resource to complete its mission within a time/step limit.

---

## 🛠️ Technologies Used
* **Language**: Python 3.x
* **Core Library**: [Gymnasium](https://gymnasium.farama.org/) (for reinforcement learning environments)
* **Computation**: [NumPy](https://numpy.org/)

---

## ⚙️ How to Run
1.  **Set up the environment:**
    ```bash
    pip install gymnasium numpy
    ```
2.  **Run the simulation:**
    Ensure both `grid_world_env.py` and `q_learning_agent.py` are in the same directory.
    ```bash
    python q_learning_agent.py
    ```

### File Descriptions
* `grid_world_env.py`: Contains the Gymnasium environment class. This file defines the "laws of physics" for our simulation, including the grid, agent actions, rewards, and objectives.
* `q_learning_agent.py`: Contains the training and execution logic for our Q-learning agent. This file defines the agent's "brain" and how it learns from its experiences.