# """CHANGE IS PYGLET VIEW HERE""" #####################################################################################
PYGLET = False
########################################################################################################################

# """CHANGE PYGLET VIEW HERE""" ########################################################################################

if PYGLET:
    # """CHANGE CUSTOM ENV UTILS IMPORT HERE""" ########################################################################
    import time

    ####################################################################################################################
    from pyglet.gl import *

    from .custom_env import RES


class PygletView(pyglet.window.Window if PYGLET else object):
    """Abstract class for Pyglet-based visualization."""

    @staticmethod
    def points_to_pyglet_vertex(points, color):
        """Converts point coordinates to Pyglet vertex list."""
        return pyglet.graphics.vertex_list(
            len(points),
            (
                "v3f/stream",
                [
                    item
                    for sublist in ([p[0], p[1], 0] for p in points)
                    for item in sublist
                ],
            ),
            ("c3B", PygletView.color_polygon(len(points), color)),
        )

    @staticmethod
    def color_polygon(n, color):
        """Repeats color value for n vertices."""
        colors = []
        for _i in range(n):
            colors.extend(color)
        return colors

    @staticmethod
    def draw_polygons(polygons, color):
        """Draws filled polygons with given color."""
        [
            PygletView.points_to_pyglet_vertex(polygon, color).draw(gl.GL_TRIANGLE_FAN)
            for polygon in polygons
        ]

    @staticmethod
    def draw_vertices(vertices, color):
        """Draws lines connecting vertices."""
        [
            PygletView.points_to_pyglet_vertex(vertex, color).draw(gl.GL_LINES)
            for vertex in vertices
        ]

    @staticmethod
    def draw_label_top_left(
        text, x, y, y_offset=0, margin=50, font_size=40, color=(0, 0, 0, 255)
    ):
        """Renders text label at top left of window."""
        pyglet.text.Label(
            text,
            x=x + margin,
            y=y - y_offset * (font_size + margin) - margin,
            font_size=font_size,
            color=color,
        ).draw()

    @staticmethod
    def load_sprite(path, anchor_x=0.5, anchor_y=0.5):
        """Loads image sprite with specified anchor point."""
        img = pyglet.image.load(path)
        img.anchor_x = int(img.width * anchor_x)
        img.anchor_y = int(img.height * anchor_y)
        return pyglet.sprite.Sprite(img, 0, 0)

    def __init__(self, name, env):
        """Initializes Pyglet window and environment."""
        # """CHANGE VIEW INIT HERE""" ##################################################################################
        (width, height) = RES
        background_color = [255, 255, 255]
        ################################################################################################################

        super().__init__(width, height, name, resizable=True)
        glClearColor(
            background_color[0] / 255,
            background_color[1] / 255,
            background_color[2] / 255,
            1,
        )
        self.zoom = 1
        self.key = None

        self.env = env

        self.setup()

        # """CHANGE VIEW SETUP HERE""" #################################################################################
        ################################################################################################################

    def get_play_action(self):
        """Placeholder for human playback action."""
        play_action = 0

        # """CHANGE GET PLAY ACTION HERE""" ############################################################################
        ################################################################################################################

        return play_action

    def on_draw(self, dt=0.002):
        """Pyglet draw loop callback."""
        self.clear()

        self.loop()

        # """CHANGE VIEW LOOP HERE""" ##################################################################################
        ################################################################################################################

    def on_resize(self, width, height):
        """Handles window resize events."""
        glMatrixMode(gl.GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(-width, width, -height, height, -1, 1)
        glViewport(0, 0, width, height)
        glOrtho(-self.zoom, self.zoom, -self.zoom, self.zoom, -1, 1)

    def on_key_press(self, symbol, modifiers):
        """Handles key press events."""
        self.key = symbol

    def on_key_release(self, symbol, modifiers):
        """Handles key release events."""
        if self.key == symbol:
            self.key = None

    def setup(self):
        raise NotImplementedError

    def loop(self):
        raise NotImplementedError

    def run(self):
        """Starts Pyglet application loop."""
        pyglet.clock.schedule_interval(self.on_draw, 0.002)
        pyglet.app.run()


########################################################################################################################

# """CHANGE CUSTOM VIEW HERE""" ########################################################################################

if not PYGLET:
    import time

    # """CHANGE CLOCK SCHEDULE INTERVAL HERE""" ########################################################################
    DT = 0.0
    ####################################################################################################################


class CustomView:
    """Fallback view when Pyglet is disabled."""

    def __init__(self, name, env):
        """Initializes custom view."""
        self.name = name
        self.env = env

        self.setup()

        # """CHANGE VIEW SETUP HERE""" #################################################################################
        ################################################################################################################

    def get_play_action(self):
        """Placeholder for human playback action."""
        play_action = 0

        # """CHANGE GET PLAY ACTION HERE""" ############################################################################
        ################################################################################################################

        return play_action

    def on_draw(self):
        """Placeholder for draw loop."""
        # """CHANGE VIEW LOOP HERE""" ##################################################################################
        pass
        ################################################################################################################

    def clear(self):
        """Placeholder for screen clear."""
        # """CHANGE CLEAR VIEW HERE""" #################################################################################
        pass
        ################################################################################################################

    def setup(self):
        raise NotImplementedError

    def loop(self):
        raise NotImplementedError

    def run(self):
        """Starts primitive main loop."""
        while True:
            self.clear()
            self.loop()
            self.on_draw()
            time.sleep(DT)


########################################################################################################################
