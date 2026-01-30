# miniworld/envs/maze_static.py
from gymnasium import spaces, utils
from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv
import numpy as np

class StaticMaze(MiniWorldEnv, utils.EzPickle):
    def __init__(self, max_episode_steps=500, **kwargs): # Increased steps for larger room
        self.room_size = 4
        # Match MazeS2 spacing so gaps render as blue separators in top view
        self.gap_size = 0.25
        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        grid = []
        for r in range(3):
            row = []
            for c in range(3):
                room = self.add_rect_room(
                    min_x=c * (self.room_size + self.gap_size),
                    max_x=c * (self.room_size + self.gap_size) + self.room_size,
                    min_z=r * (self.room_size + self.gap_size),
                    max_z=r * (self.room_size + self.gap_size) + self.room_size,
                    wall_tex="brick_wall",
                )
                row.append(room)
            grid.append(row)

        # SINGLE ROUTE TO GOAL + DEAD ENDS (tree, no cycles)
        # Main route: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)
        self.connect_rooms(grid[0][0], grid[0][1], min_z=grid[0][0].min_z, max_z=grid[0][0].max_z)
        self.connect_rooms(grid[0][1], grid[0][2], min_z=grid[0][1].min_z, max_z=grid[0][1].max_z)
        self.connect_rooms(grid[0][2], grid[1][2], min_x=grid[0][2].min_x, max_x=grid[0][2].max_x)

        # Dead-end branches off the main route
        self.connect_rooms(grid[1][1], grid[1][2], min_z=grid[1][1].min_z, max_z=grid[1][1].max_z)
        door_w = self.room_size * 0.6
        # Make the two left-column cells one open room (full-width portal)
        self.connect_rooms(
            grid[1][0],
            grid[2][0],
            min_x=grid[1][0].min_x,
            max_x=grid[1][0].max_x,
        )
        # Entrance gap on the separating wall to the right (centered)
        door_min_z = grid[1][0].min_z + (self.room_size - door_w) * 0.5
        self.connect_rooms(
            grid[1][0],
            grid[1][1],
            min_z=door_min_z,
            max_z=door_min_z + door_w,
        )
        self.connect_rooms(grid[1][1], grid[2][1], min_x=grid[1][1].min_x, max_x=grid[1][1].max_x)
        goal_door_min_z = grid[2][2].min_z + (self.room_size - door_w) * 0.5
        self.connect_rooms(
            grid[2][2],
            grid[2][1],
            min_z=goal_door_min_z,
            max_z=goal_door_min_z + door_w,
        )

        # FIXED SPAWNS
        self.place_agent(room=grid[0][0], dir=0) # Start
        self.box = self.place_entity(Box(color="red"), room=grid[2][2]) # End

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        if self.near(self.box):
            reward += self._reward()
            termination = True
        return obs, reward, termination, truncation, info
     

class StaticMazeSmall(MiniWorldEnv, utils.EzPickle):
    def __init__(self, max_episode_steps=300, **kwargs):
        self.room_size = 3.0
        # Match MazeS2 spacing so gaps render as blue separators in top view
        self.gap_size = 0.25
        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        grid = []
        for r in range(2):
            row = []
            for c in range(2):
                room = self.add_rect_room(
                    min_x=c * (self.room_size + self.gap_size),
                    max_x=c * (self.room_size + self.gap_size) + self.room_size,
                    min_z=r * (self.room_size + self.gap_size),
                    max_z=r * (self.room_size + self.gap_size) + self.room_size,
                    wall_tex="brick_wall",
                )
                row.append(room)
            grid.append(row)

        # Treat bottom-left as the enclosed room (goal room)
        goal_room = grid[0][0]

        # Open area is the other three rooms; connect them with full portals
        # Bottom-right to Top-right (vertical)
        self.connect_rooms(
            grid[0][1], grid[1][1],
            min_x=grid[0][1].min_x, max_x=grid[0][1].max_x
        )
        # Top-right to Top-left (horizontal)
        self.connect_rooms(
            grid[1][1], grid[1][0],
            min_z=grid[1][1].min_z, max_z=grid[1][1].max_z
        )

        # Entrance between goal room and open area (east wall of goal room)
        door_w = self.room_size * 0.6
        door_min_z = goal_room.min_z + (self.room_size - door_w) * 0.5
        door_max_z = door_min_z + door_w
        self.connect_rooms(
            goal_room, grid[0][1],
            min_z=door_min_z, max_z=door_max_z
        )

        # Spawn in open area (e.g., top-right)
        self.place_agent(room=grid[1][1], dir=0)

        # Goal explicitly inside the goal room
        goal_margin = 0.4
        self.box = self.place_entity(
            Box(color="red"),
            min_x=goal_room.min_x + goal_margin,
            max_x=goal_room.max_x - goal_margin,
            min_z=goal_room.min_z + goal_margin,
            max_z=goal_room.max_z - goal_margin,
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        if self.near(self.box):
            reward += self._reward()
            termination = True
        return obs, reward, termination, truncation, info
