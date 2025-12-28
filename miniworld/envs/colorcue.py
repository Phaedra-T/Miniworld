import math
import numpy as np
from gymnasium import spaces, utils
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box


class ColorCueCorridor(MiniWorldEnv, utils.EzPickle):
    """
    Memory-forcing environment:
    - Agent sees a colored cue (red or blue) in the start room
    - Cue indicates whether the goal is in the left or right corridor
    - Corridors are visually identical
    - Cue is NOT visible at the decision point
    """

    def __init__(self, max_episode_steps=200, **kwargs):
        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(self, max_episode_steps, **kwargs)

        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # ---------- Start room ----------
        start_room = self.add_rect_room(
            min_x=-3, max_x=3,
            min_z=0, max_z=3
        )

        # ---------- Corridor hub ----------
        hub = self.add_rect_room(
            min_x=-2, max_x=2,
            min_z=-2, max_z=0
        )

        # ---------- Left corridor ----------
        left_room = self.add_rect_room(
            min_x=-4, max_x=-1,
            min_z=-5, max_z=-2
        )

        # ---------- Right corridor ----------
        right_room = self.add_rect_room(
            min_x=1, max_x=4,
            min_z=-5, max_z=-2
        )

        # ---------- Connect rooms ----------
        self.connect_rooms(start_room, hub, min_x=-1, max_x=1)
        self.connect_rooms(hub, left_room, min_x=-2, max_x=-1)
        self.connect_rooms(hub, right_room, min_x=1, max_x=2)


        # ---------- Randomize goal side ----------
        goal_on_left = self.np_random.integers(0, 2) == 0

        # ---------- Cue block (ONLY in start room) ----------
        cue_color = "green" if goal_on_left else "blue"
        self.place_entity(
            Box(color=cue_color),
            room=start_room,
            min_x=-0.5, max_x=0.5,
            min_z=1, max_z=1.5
        )

        # ---------- Goal ----------
        self.box = Box(color="red")

        if goal_on_left:
            self.place_entity(
                self.box,
                room=left_room,
                min_z=left_room.min_z + 0.5,
                max_z=left_room.min_z + 1.5
            )
        else:
            self.place_entity(
                self.box,
                room=right_room,
                min_z=right_room.min_z + 0.5,
                max_z=right_room.min_z + 1.5
            )

        # ---------- Agent ----------
        self.place_agent(
            room=start_room,
            dir=self.np_random.uniform(-0.1, 0.1),
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += 1.0
            termination = True

        return obs, reward, termination, truncation, info
