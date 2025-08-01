# environment/rendering.py

import pygame
import os

# Constants
TILE_SIZE = 100
GRID_SIZE = 5
WINDOW_SIZE = TILE_SIZE * GRID_SIZE

# Load assets (make sure to put your images in a 'assets/' folder inside project root)
ICON_PATHS = {
    "agent": "assets/smiley.png",
    "client": "assets/people.png",
    "law_book": "assets/folder.png",
    "inquiry": "assets/book.png",
    "lawyer": "assets/lawyer.png",
    "penalty": "assets/redx.png"
}

class Renderer:
    def __init__(self, grid_size=GRID_SIZE):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Legal Help Env")
        self.clock = pygame.time.Clock()

        self.images = {}
        for name, path in ICON_PATHS.items():
            if os.path.exists(path):
                image = pygame.image.load(path)
                self.images[name] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
            else:
                print(f"⚠️ Missing image: {path}")

        self.grid_size = grid_size

    def draw_grid(self, agent_pos, object_positions):
        self.window.fill((255, 255, 255))  # White background

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                pygame.draw.rect(self.window, (0, 0, 0), (col*TILE_SIZE, row*TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

        for name, pos in object_positions.items():
            if name in self.images:
                self.window.blit(self.images[name], (pos[1] * TILE_SIZE, pos[0] * TILE_SIZE))

        # Agent is always last so it appears on top
        if "agent" in self.images:
            self.window.blit(self.images["agent"], (agent_pos[1] * TILE_SIZE, agent_pos[0] * TILE_SIZE))

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        pygame.quit()
