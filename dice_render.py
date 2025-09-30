#!/usr/bin/env python3
"""
3D Dice Roll Animation Generator
Uses PyBullet for realistic physics and ModernGL for 3D rendering
Generates dice_animation.gif and prints final result to console
"""

import math
import time
import numpy as np
import pybullet as p
import moderngl
from PIL import Image
import random
import sys
import argparse
from contextlib import contextmanager
from dice_6sided import Dice6Sided
from environment import EnvironmentRenderer

# Visual constants matching your current style
COLORS = {
    'bg': (24, 24, 28),           # C_BG #18181C
    'floor_far': (28, 31, 37),    # C_FLOOR_FAR #1C1F25
    'floor_near': (38, 42, 49),   # C_FLOOR_NEAR #262A31
    'grid': (46, 51, 64),         # C_GRID #2E3340
    'ivory': (248, 246, 240),     # C_IVORY #F8F6F0
    'pip': (18, 18, 22),          # C_PIP #121216
    'edge': (36, 36, 42)          # C_EDGE #242A2A
}

# Animation settings
CANVAS_SIZE = 1024  # Larger for testing/development
FRAME_RATE = 60
SIMULATION_TIME = 10  # seconds - longer for testing
PHYSICS_FREQ = 120     # Hz for physics simulation

class DiceRenderer:
    def __init__(self):
        self.width = CANVAS_SIZE
        self.height = CANVAS_SIZE
        self.frames = []
        self.final_face = 1

        # Initialize dice and environment
        self.dice = Dice6Sided()
        self.environment = None

        # Initialize PyBullet
        self.physics_client = None
        self.dice_id = None

        # Initialize ModernGL
        self.ctx = None
        self.prog = None
        self.vao = None
        self.fbo = None

    def setup_physics(self):
        """Initialize PyBullet physics simulation"""
        # Create physics world
        self.physics_client = p.connect(p.DIRECT)  # No GUI
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1.0 / PHYSICS_FREQ, physicsClientId=self.physics_client)

        # Improve physics solver for better stability
        p.setPhysicsEngineParameter(
            numSolverIterations=10,  # More iterations for better accuracy
            numSubSteps=4,           # More substeps for stability
            physicsClientId=self.physics_client
        )

        # Create environment (floor and walls)
        self.environment = EnvironmentRenderer(None)  # Context will be set later
        self.environment.create_physics_boundaries(self.physics_client)

        # Create dice using dice class with bounded starting position
        start_pos = self.environment.get_bounded_start_position()
        start_orientation = p.getQuaternionFromEuler([
            random.uniform(0, 2*math.pi),
            random.uniform(0, 2*math.pi),
            random.uniform(0, 2*math.pi)
        ])

        # Create dice physics body
        self.dice_id = self.dice.create_physics_body(
            self.physics_client, start_pos, start_orientation
        )

        # Add energetic velocity for bouncing with guaranteed minimum energy
        # Ensure minimum horizontal velocity magnitude for dynamic motion
        linear_vel = [
            random.uniform(-4, 4),    # Stronger horizontal velocity for more bouncing
            random.uniform(-4, 4),    # Stronger horizontal velocity for more bouncing
            random.uniform(-2, 1)     # More varied vertical velocity
        ]

        # Ensure minimum total horizontal velocity for dynamic motion
        horizontal_speed = (linear_vel[0]**2 + linear_vel[1]**2)**0.5
        if horizontal_speed < 2.0:  # Minimum horizontal speed
            scale = 2.0 / horizontal_speed
            linear_vel[0] *= scale
            linear_vel[1] *= scale

        # Guarantee significant angular velocity for tumbling
        angular_vel = [
            random.uniform(-8, 8),    # Stronger spinning for more tumbling
            random.uniform(-8, 8),    # Stronger spinning for more tumbling
            random.uniform(-8, 8)     # Stronger spinning for more tumbling
        ]

        # Ensure minimum angular velocity magnitude
        angular_speed = (angular_vel[0]**2 + angular_vel[1]**2 + angular_vel[2]**2)**0.5
        if angular_speed < 4.0:  # Minimum angular speed
            scale = 4.0 / angular_speed
            angular_vel = [v * scale for v in angular_vel]

        p.resetBaseVelocity(
            self.dice_id,
            linearVelocity=linear_vel,
            angularVelocity=angular_vel,
            physicsClientId=self.physics_client
        )

    def setup_rendering(self):
        """Initialize ModernGL rendering context"""
        # Create an offscreen context
        self.ctx = moderngl.create_context(standalone=True)

        # Create framebuffer for offscreen rendering
        self.fbo = self.ctx.framebuffer(
            color_attachments=[
                self.ctx.texture((self.width, self.height), 4)
            ],
            depth_attachment=self.ctx.depth_texture((self.width, self.height))
        )

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)  # OK now that winding is consistent (CCW everywhere)

        # Get shaders from dice class
        vertex_shader = self.dice.get_vertex_shader()
        fragment_shader = self.dice.get_fragment_shader()

        try:
            self.prog = self.ctx.program(vertex_shader=vertex_shader,
                                       fragment_shader=fragment_shader)
        except Exception as e:
            print(f"Shader compilation error: {e}")
            print(f"Vertex shader:\n{vertex_shader}")
            print(f"Fragment shader:\n{fragment_shader}")
            raise

        # Initialize environment rendering
        if self.environment:
            self.environment.ctx = self.ctx
            self.environment.setup_rendering()

    def create_cube_geometry(self):
        """Create dice geometry using the dice class"""
        # Get geometry from dice class
        vertices = self.dice.get_geometry()

        # Create VBO and VAO
        vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (vbo, '3f 2f 1f', 'position', 'texcoord', 'face_id')
        ])

    def render_frame(self, dice_pos, dice_rot, camera_t=0.0):
        """Render a single frame with the dice at given position/rotation"""
        self.fbo.use()
        self.ctx.clear(color=(0.0, 0.0, 0.0, 1.0))  # Pure black background

        # Set up camera with swing animation
        view_matrix = self.create_view_matrix(camera_t)
        proj_matrix = self.create_projection_matrix()

        # Render environment (floor and walls) first
        if self.environment:
            # Environment uses identity model matrix (world coordinates)
            env_mvp = proj_matrix @ view_matrix
            self.environment.render(env_mvp)

        # Render dice on top
        model_matrix = self.create_model_matrix(dice_pos, dice_rot)
        dice_mvp = proj_matrix @ view_matrix @ model_matrix

        # Send column-major to GL (only MVP needed now)
        self.prog['mvp'].write(dice_mvp.T.astype(np.float32).tobytes())

        # Render the dice
        self.vao.render()

        # Read the framebuffer
        data = self.fbo.color_attachments[0].read()
        img = Image.frombytes('RGBA', (self.width, self.height), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL flips Y
        return img.convert('RGB')

    def create_view_matrix(self, t=0.0):
        """Create view matrix with camera swing from TRON to 45-degree view

        Args:
            t: Animation parameter 0.0=TRON view, 1.0=45-degree view
        """
        # TRON perspective (start) - pulled back for wider view with slight tilt
        tron_eye = np.array([0.0, -4.5, 1.8])       # Further back for wider starting view, slightly higher
        tron_target = np.array([0.0, 1.0, 0.0])

        # Angled view - stay close with tilt to dice top face
        angled_eye = np.array([1.0, -1.2, 3.5])     # Closer zoom, moderate height for tilt
        angled_target = np.array([0.0, 0.0, 0.5])   # Target slightly above floor to focus on dice

        # Apply easing to camera movement for smoother transition
        # Use smoothstep for more natural acceleration/deceleration
        eased_t = t * t * (3.0 - 2.0 * t)  # Smoothstep function

        # Interpolate between views using eased parameter
        eye = tron_eye + eased_t * (angled_eye - tron_eye)
        target = tron_target + eased_t * (angled_target - tron_target)
        up = np.array([0.0, 0.0, 1.0])

        return self.look_at(eye, target, up)

    def create_projection_matrix(self):
        """Create projection matrix"""
        # Perspective projection for realistic TRON look
        fov = 60.0  # Field of view in degrees
        aspect = self.width / self.height
        near, far = 0.1, 20.0

        return self.perspective(fov, aspect, near, far)

    def create_model_matrix(self, pos, rot):
        """Create model matrix from position and rotation quaternion"""
        # Convert quaternion to rotation matrix
        x, y, z, w = rot

        # Create 4x4 transformation matrix with rotation and translation
        model = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), pos[0]],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), pos[1]],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y), pos[2]],
            [0, 0, 0, 1]
        ])

        return model

    @staticmethod
    def look_at(eye, target, up):
        """Create look-at view matrix"""
        f = (target - eye)
        f = f / np.linalg.norm(f)

        s = np.cross(f, up)
        s = s / np.linalg.norm(s)

        u = np.cross(s, f)

        result = np.identity(4)
        result[0][0:3] = s
        result[1][0:3] = u
        result[2][0:3] = -f
        result[0][3] = -np.dot(s, eye)
        result[1][3] = -np.dot(u, eye)
        result[2][3] = np.dot(f, eye)

        return result

    @staticmethod
    def perspective(fov, aspect, near, far):
        """Create perspective projection matrix"""
        fov_rad = math.radians(fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        result = np.zeros((4, 4))
        result[0][0] = f / aspect
        result[1][1] = f
        result[2][2] = (far + near) / (near - far)
        result[2][3] = (2.0 * far * near) / (near - far)
        result[3][2] = -1.0
        return result

    @staticmethod
    def ortho(left, right, bottom, top, near, far):
        """Create orthographic projection matrix"""
        result = np.zeros((4, 4))
        result[0][0] = 2.0 / (right - left)
        result[1][1] = 2.0 / (top - bottom)
        result[2][2] = -2.0 / (far - near)
        result[3][3] = 1.0
        result[0][3] = -(right + left) / (right - left)
        result[1][3] = -(top + bottom) / (top - bottom)
        result[2][3] = -(far + near) / (far - near)
        return result

    def detect_final_face(self):
        """Determine which face is pointing up using dice class"""
        # Get dice orientation
        pos, orientation = p.getBasePositionAndOrientation(
            self.dice_id, physicsClientId=self.physics_client
        )

        # Use dice class method for face detection
        return self.dice.detect_top_face(orientation, self.physics_client)

    def is_dice_settled(self):
        """Check if dice has settled (low velocity and stable orientation)"""
        linear_vel, angular_vel = p.getBaseVelocity(
            self.dice_id, physicsClientId=self.physics_client
        )

        linear_speed = np.linalg.norm(linear_vel)
        angular_speed = np.linalg.norm(angular_vel)

        # Much stricter velocity criteria
        velocity_settled = linear_speed < 0.01 and angular_speed < 0.05

        # Also check if dice is lying flat (not on edge/corner)
        pos, orientation = p.getBasePositionAndOrientation(
            self.dice_id, physicsClientId=self.physics_client
        )

        # Convert quaternion to rotation matrix to check orientation
        rotation_matrix = p.getMatrixFromQuaternion(orientation)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

        # Check if any face is pointing nearly straight up (close to Z-axis)
        # For a cube, we need one of the face normals to be close to [0,0,1]
        face_normals = [
            [1, 0, 0], [-1, 0, 0],  # X faces
            [0, 1, 0], [0, -1, 0],  # Y faces
            [0, 0, 1], [0, 0, -1]   # Z faces
        ]

        max_z_component = 0
        for normal in face_normals:
            # Transform face normal by rotation matrix
            rotated_normal = rotation_matrix @ np.array(normal)
            # Check Z component (how much it points up)
            max_z_component = max(max_z_component, abs(rotated_normal[2]))

        # Dice is flat if one face is pointing mostly up (Z component > 0.9)
        orientation_settled = max_z_component > 0.9

        return velocity_settled and orientation_settled

    def simulate_and_render(self):
        """Main simulation and rendering loop"""
        print("Starting dice simulation...")

        total_frames = int(SIMULATION_TIME * FRAME_RATE)
        physics_steps_per_frame = PHYSICS_FREQ // FRAME_RATE

        settled_frames = 0

        for frame in range(total_frames):
            # Step physics simulation
            for _ in range(physics_steps_per_frame):
                p.stepSimulation(physicsClientId=self.physics_client)

            # Get dice state
            pos, orientation = p.getBasePositionAndOrientation(
                self.dice_id, physicsClientId=self.physics_client
            )

            # Check if settled
            if self.is_dice_settled():
                settled_frames += 1
                if settled_frames > FRAME_RATE * 2.0:  # Settled for 2.0 seconds to enjoy the tilted view
                    self.final_face = self.detect_final_face()
                    break
            else:
                settled_frames = 0

            # Calculate camera swing based on simulation progress
            # Start swinging camera when dice begins to settle
            simulation_progress = frame / total_frames
            if settled_frames > 0:
                # Start camera swing when dice begins settling, swing over 1.5 seconds
                camera_t = min(1.0, settled_frames / (FRAME_RATE * 1.5))  # Slower swing over 1.5 seconds
            else:
                # Use TRON view while dice is bouncing around
                camera_t = 0.0

            # Render frame with camera animation
            img = self.render_frame(pos, orientation, camera_t)
            self.frames.append(img)

            if frame % 10 == 0:
                print(f"Rendered frame {frame}/{total_frames}")

        # If not settled by end, detect face anyway
        if settled_frames <= FRAME_RATE * 0.5:
            self.final_face = self.detect_final_face()

        print(f"Simulation complete. Final face: {self.final_face}")

    def save_animation(self, filename="dice_animation.mp4"):
        """Save animation as video (MP4/H.264) or GIF"""
        if not self.frames:
            print("No frames to save!")
            return

        print(f"Saving {len(self.frames)} frames to {filename}...")

        file_ext = filename.lower().split('.')[-1]

        if file_ext == 'gif':
            # Save as GIF with centisecond rounding
            duration_exact = 1000 / FRAME_RATE
            frame_duration = round(duration_exact / 10) * 10
            actual_fps = 1000 / frame_duration
            print(f"Frame duration: {frame_duration}ms (target FPS: {FRAME_RATE}, actual: {actual_fps:.1f})")

            self.frames[0].save(
                filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=frame_duration,
                loop=0
            )

        elif file_ext in ['mp4', 'avi', 'mov']:
            # Save as video using imageio with H.264
            try:
                import imageio

                # Convert PIL Images to numpy arrays
                frames_array = []
                for frame in self.frames:
                    frames_array.append(np.array(frame))

                # Save as H.264 video with high quality settings
                print(f"Saving as H.264 video at {FRAME_RATE} FPS")
                imageio.mimsave(
                    filename,
                    frames_array,
                    fps=FRAME_RATE,
                    codec='libx264',
                    quality=10,  # Maximum quality (0-10 scale)
                    ffmpeg_params=[
                        '-crf', '15',        # Very low CRF for high quality (0-51, lower=better)
                        '-preset', 'slow'    # Slower encoding for better compression
                    ]
                )

            except ImportError:
                print("Error: imageio not installed. Install with: pip install imageio[ffmpeg]")
                return
            except Exception as e:
                print(f"Error saving video: {e}")
                return

        else:
            print(f"Unsupported format: {file_ext}")
            return

        print(f"Saved {filename}")

    def cleanup(self):
        """Clean up resources"""
        if self.environment and self.physics_client is not None:
            self.environment.cleanup_physics(self.physics_client)

        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)

        if self.ctx is not None:
            self.ctx.release()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate 3D dice roll animation')
    parser.add_argument('-o', '--output',
                        default='dice_animation.mp4',
                        help='Output file path (default: dice_animation.mp4). Format determined by extension (.mp4, .gif, .avi, .mov)')

    args = parser.parse_args()

    renderer = DiceRenderer()

    try:
        # Setup
        renderer.setup_physics()
        renderer.setup_rendering()
        renderer.create_cube_geometry()

        # Run simulation
        renderer.simulate_and_render()

        # Save results
        renderer.save_animation(args.output)

        # Print final result
        print(renderer.final_face)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        renderer.cleanup()

    return 0

if __name__ == "__main__":
    sys.exit(main())