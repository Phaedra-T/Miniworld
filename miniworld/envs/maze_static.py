# miniworld/envs/maze_static.py
from gymnasium import spaces, utils
from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv
import numpy as np

class StaticMaze(MiniWorldEnv, utils.EzPickle):
    def __init__(self, max_episode_steps=300, **kwargs):
        self.num_rows = 3
        self.num_cols = 3
        self.room_size = 3
        self.gap_size = 0.25  # thin wall thickness

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(self, max_episode_steps=max_episode_steps, **kwargs)

        # Only allow movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # === Create a 2x2 grid of rooms ===
        grid = []
        for r in range(self.num_rows):
            row = []
            for c in range(self.num_cols):
                min_x = c * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size
                min_z = r * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    floor_tex="concrete_tiles"
                )
                row.append(room)
            grid.append(row)

        # === Deterministic connections (explicit) ===
        s = self.room_size

        # Connect top-left → top-right
        self.connect_rooms(
            grid[0][0],
            grid[1][0],
            min_x=grid[0][0].min_x,
            max_x=grid[0][0].max_x
        )

        # Connect top-right → bottom-right
        self.connect_rooms(
            grid[0][1],
            grid[1][1],
            min_x=grid[0][1].min_x,
            max_x=grid[0][1].max_x,
        )

        # Connect bottom-right → bottom-left (optional loop)

        self.connect_rooms(
            grid[2][0],
            grid[2][1],
            min_z=grid[2][0].min_z,
            max_z=grid[2][1].max_z,
        )

        self.connect_rooms(
            grid[2][1],
            grid[2][2],
            min_z=grid[2][1].min_z,
            max_z=grid[2][2].max_z,
        )

        self.connect_rooms(
            grid[1][1],
            grid[1][2],
            min_z=grid[1][1].min_z,
            max_z=grid[1][2].max_z
        )

        self.connect_rooms(
            grid[0][1],
            grid[0][2],
            min_z=grid[0][1].min_z,
            max_z=grid[0][2].min_z+ s / 2 + 0.75
        )


        self.connect_rooms(
            grid[2][2],
            grid[1][2],
            min_x=grid[0][2].min_x,
            max_x=grid[1][2].max_x,
        )


        self.connect_rooms(
            grid[2][0],
            grid[1][0],
            min_x=grid[1][0].min_x,
            max_x=grid[1][0].max_x,
        )


        # --- Agent spawn and goal ---
        self.place_agent(room=grid[0][0], dir=0)
        self.box = self.place_entity(Box(color="red"))
        self.goal = self.place_entity(self.box, room=grid[0][2])

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info
     

class StaticMazeSmall(MiniWorldEnv, utils.EzPickle):
    def __init__(self, max_episode_steps=300, **kwargs):
        self.num_rows = 2
        self.num_cols = 2
        self.room_size = 3
        self.gap_size = 0.25  

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(self, max_episode_steps=max_episode_steps, **kwargs)

        # Only allow movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # === Create a 2x2 grid of rooms ===
        grid = []
        for r in range(self.num_rows):
            row = []
            for c in range(self.num_cols):
                min_x = c * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size
                min_z = r * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    floor_tex="concrete_tiles"
                )
                row.append(room)
            grid.append(row)

        # === Deterministic connections (explicit) ===
        s = self.room_size

        # Connect top-left → top-right
        self.connect_rooms(
            grid[0][0],
            grid[0][1],
            min_z=grid[0][0].min_z + s / 2 - 0.75,
            max_z=grid[0][0].min_z + s / 2 + 0.75,
        )

        # Connect top-right → bottom-right
        self.connect_rooms(
            grid[0][1],
            grid[1][1],
            min_x=grid[0][1].min_x + s / 2 - 0.75,
            max_x=grid[0][1].min_x + s / 2 + 0.75,
        )

        # Connect bottom-right → bottom-left (optional loop)
        self.connect_rooms(
            grid[0][0],
            grid[1][0],
            min_x=grid[0][0].min_x + s / 2 - 0.75,
            max_x=grid[1][0].min_x + s / 2 + 0.75,
        )


        # --- Agent spawn and goal ---
        self.place_agent(room=grid[0][0], dir=0)
        self.box = self.place_entity(Box(color="red"))
        self.goal = self.place_entity(self.box)#, room=grid[1][1])

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        if self.near(self.goal):
            reward += self._reward()
            termination = True
        return obs, reward, termination, truncation, info
    


