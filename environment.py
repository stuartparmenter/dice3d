#!/usr/bin/env python3
"""
Environment Renderer Module
Creates retro 80s-style floor and walls for the dice simulation
"""

import numpy as np
import pybullet as p
import moderngl


class EnvironmentRenderer:
    """Renders retro-style floor and boundaries for dice simulation"""

    # Debug flag to disable wall rendering (physics boundaries remain)
    DEBUG_WALLS = False

    # Environment dimensions (matches camera projection bounds)
    BOUNDS = {
        'width': 5.0,      # Moderately larger play area to reduce early wall bounces
        'depth': 3.5,      # Moderately larger play area to reduce early wall bounces
        'wall_height': 6.0, # Much taller walls to contain energetic bounces
        'floor_size': 8.0,  # Visual floor extends beyond boundaries
    }

    # Wall physics properties
    WALL_PROPS = {
        'thickness': 0.1,   # Wall thickness
        'restitution': 0.6, # Higher wall bounce for more action
        'friction': 0.5,    # Lower friction for more bouncing
    }

    def __init__(self, ctx):
        """
        Initialize environment renderer

        Args:
            ctx: ModernGL context
        """
        self.ctx = ctx
        self.floor_prog = None
        self.floor_vao = None
        self.wall_prog = None
        self.wall_vao = None
        self.wall_ids = []  # PyBullet wall body IDs

    def create_physics_boundaries(self, physics_client_id):
        """
        Create physics walls and floor using PyBullet

        Args:
            physics_client_id: PyBullet physics client ID

        Returns:
            list: Wall body IDs for cleanup
        """
        bounds = self.BOUNDS
        props = self.WALL_PROPS

        # Create floor as a large box (instead of infinite plane)
        floor_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[bounds['width']*2, bounds['depth']*2, 0.2],  # Much larger floor
            physicsClientId=physics_client_id
        )
        floor_id = p.createMultiBody(
            baseMass=0,  # Static body
            baseCollisionShapeIndex=floor_collision,
            basePosition=[0, 0, -0.2],  # Floor surface at Z=0
            physicsClientId=physics_client_id
        )

        # Set floor properties
        p.changeDynamics(
            floor_id, -1,
            restitution=props['restitution'],
            lateralFriction=props['friction'],
            physicsClientId=physics_client_id
        )
        self.wall_ids.append(floor_id)

        # Create 4 walls around the perimeter
        wall_positions = [
            # [x, y, z], [half_x, half_y, half_z] for each wall
            ([bounds['width']/2, 0, bounds['wall_height']/2], [props['thickness'], bounds['depth']/2, bounds['wall_height']/2]),  # +X wall
            ([-bounds['width']/2, 0, bounds['wall_height']/2], [props['thickness'], bounds['depth']/2, bounds['wall_height']/2]), # -X wall
            ([0, bounds['depth']/2, bounds['wall_height']/2], [bounds['width']/2, props['thickness'], bounds['wall_height']/2]),   # +Y wall
            ([0, -bounds['depth']/2, bounds['wall_height']/2], [bounds['width']/2, props['thickness'], bounds['wall_height']/2])  # -Y wall
        ]

        for pos, extents in wall_positions:
            wall_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=extents,
                physicsClientId=physics_client_id
            )
            wall_id = p.createMultiBody(
                baseMass=0,  # Static body
                baseCollisionShapeIndex=wall_collision,
                basePosition=pos,
                physicsClientId=physics_client_id
            )

            # Set wall properties
            p.changeDynamics(
                wall_id, -1,
                restitution=props['restitution'],
                lateralFriction=props['friction'],
                physicsClientId=physics_client_id
            )
            self.wall_ids.append(wall_id)

        return self.wall_ids

    def setup_rendering(self):
        """Initialize shaders and geometry for visual environment"""

        # Create floor shader with retro grid pattern
        vertex_shader = '''
        #version 330
        in vec3 position;
        uniform mat4 mvp;
        out vec3 world_pos;
        out vec2 grid_uv;
        void main() {
            gl_Position = mvp * vec4(position, 1.0);
            world_pos = position;
            grid_uv = position.xy;
        }
        '''

        fragment_shader = '''
        #version 330
        in vec3 world_pos;
        in vec2 grid_uv;
        out vec4 fragColor;

        // Simple noise function for felt texture
        float noise(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
        }

        void main() {
            vec2 uv = grid_uv;

            // Darker green felt base color for HUB75 panels
            vec3 felt_color = vec3(0.0, 0.25, 0.1);

            // Add subtle felt texture using noise
            float texture_scale = 20.0;  // Controls texture density
            float noise1 = noise(uv * texture_scale);
            float noise2 = noise(uv * texture_scale * 2.0);

            // Combine noise for felt-like texture
            float felt_texture = mix(noise1, noise2, 0.5);

            // Apply subtle texture variation
            float texture_strength = 0.1;  // Keep it subtle
            vec3 textured_felt = felt_color * (1.0 + (felt_texture - 0.5) * texture_strength);

            fragColor = vec4(textured_felt, 1.0);
        }
        '''

        self.floor_prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        # Create large floor quad geometry
        floor_size = self.BOUNDS['floor_size']
        floor_vertices = np.array([
            # Large quad extending beyond camera view
            [-floor_size, -floor_size, 0.0],
            [ floor_size, -floor_size, 0.0],
            [ floor_size,  floor_size, 0.0],
            [-floor_size, -floor_size, 0.0],
            [ floor_size,  floor_size, 0.0],
            [-floor_size,  floor_size, 0.0],
        ], dtype=np.float32)

        # Create VBO and VAO
        floor_vbo = self.ctx.buffer(floor_vertices.tobytes())
        self.floor_vao = self.ctx.vertex_array(self.floor_prog, [
            (floor_vbo, '3f', 'position')
        ])

        # Create wall shader for translucent walls
        wall_vertex_shader = '''
        #version 330
        in vec3 position;
        in float wall_id;
        uniform mat4 mvp;
        out float wall_id_out;
        void main() {
            gl_Position = mvp * vec4(position, 1.0);
            wall_id_out = wall_id;
        }
        '''

        wall_fragment_shader = '''
        #version 330
        in float wall_id_out;
        out vec4 fragColor;
        void main() {
            // Bright wireframe colors for each wall
            vec4 colors[4] = vec4[4](
                vec4(1.0, 0.0, 0.0, 1.0),  // Bright Red - right wall
                vec4(0.0, 1.0, 0.0, 1.0),  // Bright Green - left wall
                vec4(0.0, 0.5, 1.0, 1.0),  // Bright Blue - back wall
                vec4(1.0, 0.8, 0.0, 1.0)   // Bright Orange - front wall
            );
            fragColor = colors[int(wall_id_out)];
        }
        '''

        self.wall_prog = self.ctx.program(
            vertex_shader=wall_vertex_shader,
            fragment_shader=wall_fragment_shader
        )

        # Create wall geometry as wireframe lines
        bounds = self.BOUNDS
        wall_height = bounds['wall_height']

        wall_data = []

        # +X wall (right) - ID 0 (Red) - 4 edges of rectangle
        x = bounds['width'] / 2
        wall_data.extend([
            # Bottom edge
            [x, -bounds['depth']/2, 0, 0.0], [x, bounds['depth']/2, 0, 0.0],
            # Top edge
            [x, -bounds['depth']/2, wall_height, 0.0], [x, bounds['depth']/2, wall_height, 0.0],
            # Left edge
            [x, -bounds['depth']/2, 0, 0.0], [x, -bounds['depth']/2, wall_height, 0.0],
            # Right edge
            [x, bounds['depth']/2, 0, 0.0], [x, bounds['depth']/2, wall_height, 0.0]
        ])

        # -X wall (left) - ID 1 (Green)
        x = -bounds['width'] / 2
        wall_data.extend([
            # Bottom edge
            [x, -bounds['depth']/2, 0, 1.0], [x, bounds['depth']/2, 0, 1.0],
            # Top edge
            [x, -bounds['depth']/2, wall_height, 1.0], [x, bounds['depth']/2, wall_height, 1.0],
            # Left edge
            [x, -bounds['depth']/2, 0, 1.0], [x, -bounds['depth']/2, wall_height, 1.0],
            # Right edge
            [x, bounds['depth']/2, 0, 1.0], [x, bounds['depth']/2, wall_height, 1.0]
        ])

        # +Y wall (back) - ID 2 (Blue)
        y = bounds['depth'] / 2
        wall_data.extend([
            # Bottom edge
            [-bounds['width']/2, y, 0, 2.0], [bounds['width']/2, y, 0, 2.0],
            # Top edge
            [-bounds['width']/2, y, wall_height, 2.0], [bounds['width']/2, y, wall_height, 2.0],
            # Left edge
            [-bounds['width']/2, y, 0, 2.0], [-bounds['width']/2, y, wall_height, 2.0],
            # Right edge
            [bounds['width']/2, y, 0, 2.0], [bounds['width']/2, y, wall_height, 2.0]
        ])

        # -Y wall (front) - ID 3 (Orange)
        y = -bounds['depth'] / 2
        wall_data.extend([
            # Bottom edge
            [-bounds['width']/2, y, 0, 3.0], [bounds['width']/2, y, 0, 3.0],
            # Top edge
            [-bounds['width']/2, y, wall_height, 3.0], [bounds['width']/2, y, wall_height, 3.0],
            # Left edge
            [-bounds['width']/2, y, 0, 3.0], [-bounds['width']/2, y, wall_height, 3.0],
            # Right edge
            [bounds['width']/2, y, 0, 3.0], [bounds['width']/2, y, wall_height, 3.0]
        ])

        wall_data = np.array(wall_data, dtype=np.float32)

        # Create wall VBO and VAO
        wall_vbo = self.ctx.buffer(wall_data.tobytes())
        self.wall_vao = self.ctx.vertex_array(self.wall_prog, [
            (wall_vbo, '3f 1f', 'position', 'wall_id')
        ])

    def render(self, mvp_matrix):
        """
        Render the environment (floor and walls)

        Args:
            mvp_matrix: Model-view-projection matrix from camera
        """
        if self.floor_prog and self.floor_vao:
            # Render floor
            self.floor_prog['mvp'].write(mvp_matrix.T.astype(np.float32).tobytes())
            self.floor_vao.render()

        if self.DEBUG_WALLS and self.wall_prog and self.wall_vao:
            # Completely disable depth for wireframe walls
            self.ctx.disable(moderngl.DEPTH_TEST)

            # Render walls as wireframe
            self.wall_prog['mvp'].write(mvp_matrix.T.astype(np.float32).tobytes())
            self.wall_vao.render(moderngl.LINES)

            # Re-enable depth testing
            self.ctx.enable(moderngl.DEPTH_TEST)

    def get_bounded_start_position(self):
        """
        Get a random starting position within the bounded area

        Returns:
            list: [x, y, z] position within bounds
        """
        bounds = self.BOUNDS
        margin = 0.3  # Keep dice away from walls initially

        return [
            np.random.uniform(-bounds['width']/2 + margin, bounds['width']/2 - margin),
            np.random.uniform(-bounds['depth']/2 + margin, bounds['depth']/2 - margin),
            np.random.uniform(2.5, 4.0)  # Higher starting height for more bouncing
        ]

    def cleanup_physics(self, physics_client_id):
        """
        Remove physics bodies when cleaning up

        Args:
            physics_client_id: PyBullet physics client ID
        """
        for wall_id in self.wall_ids:
            try:
                p.removeBody(wall_id, physicsClientId=physics_client_id)
            except:
                pass  # Body might already be removed
        self.wall_ids.clear()