# willy_kalisa_rl_summative

# LegalHelp RL Environment — Summative Assignment

This project simulates an intelligent legal assistant navigating a 5x5 grid-based environment inspired by Rwandan law. The agent must interact with legal components in sequence: 📂 Law Book → 📖 Inquiry → ⚖️ Lawyer → 👥 Client, while avoiding ❌ traps.

##  Project Structure

...


##  Environment Overview

- Grid: 5x5
- Agent starts at fixed position
- Must follow strict sequence:
  1. 📂 Law Book
  2. 📖 Client Inquiry
  3. ⚖️ Lawyer
  4. 👥 Client

- Penalties for:
  - Skipping steps
  - Visiting client early
  - Stepping into ❌ trap
  - Wandering or invalid interaction

## 🕹 Action Space
Discrete(5): `[UP, DOWN, LEFT, RIGHT, INTERACT]`

##  How to Use

### 1. Install dependencies
```bash
pip install -r requirements.txt
