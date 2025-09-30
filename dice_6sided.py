#!/usr/bin/env python3
"""
6-Sided Dice Module
Contains geometry, shaders, physics properties, and face detection for a standard 6-sided die
"""

import numpy as np
import pybullet as p


class Dice6Sided:
    """A standard 6-sided die with procedural pip rendering and physics properties"""

    # Face mapping: face number -> direction
    # Standard die: opposite faces sum to 7 (1-6, 2-5, 3-4)
    FACE_MAPPING = {
        1: (0, 0, 1),    # front (+Z)
        6: (0, 0, -1),   # back (-Z)
        4: (-1, 0, 0),   # left (-X)
        3: (1, 0, 0),    # right (+X)
        2: (0, -1, 0),   # bottom (-Y)
        5: (0, 1, 0),    # top (+Y)
    }

    # Physics properties
    PHYSICS_PROPS = {
        'size': 0.5,           # Half-extent of cube
        'mass': 1.0,           # Mass in kg
        'restitution': 0.7,    # Higher bounce for more action
        'lateral_friction': 0.6,  # Reduced friction for more bouncing
        'rolling_friction': 0.1,  # Less rolling resistance for better movement
        'linear_damping': 0.05,   # Less linear damping for more bouncing
        'angular_damping': 0.15,  # Less angular damping but still some control
        'contact_stiffness': 1e5, # Contact solver stiffness
        'contact_damping': 1e3,   # Contact solver damping
    }

    def __init__(self):
        """Initialize the 6-sided dice"""
        self.face_normals = self._get_face_normals()

    def _get_face_normals(self):
        """Get face normal vectors in local coordinates for face detection"""
        return [
            np.array([0, 0, 1]),   # Face 1 (front, +Z)
            np.array([0, -1, 0]),  # Face 2 (bottom, -Y)
            np.array([1, 0, 0]),   # Face 3 (right, +X)
            np.array([-1, 0, 0]),  # Face 4 (left, -X)
            np.array([0, 1, 0]),   # Face 5 (top, +Y)
            np.array([0, 0, -1])   # Face 6 (back, -Z)
        ]

    def get_geometry(self):
        """
        Generate 24-vertex cube geometry (4 vertices per face) with proper UVs and face IDs
        Returns: numpy array of vertex data (position, texcoord, face_id)
        """
        # Face definitions: position, normal, UV coordinates, face ID
        faces = [
            # +Z front (face 1)
            {
                "n": (0, 0, 1), "id": 1,
                "v": [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)],
                "uv": [(0, 0), (1, 0), (1, 1), (0, 1)],
            },
            # -Z back (face 6)
            {
                "n": (0, 0, -1), "id": 6,
                # CCW winding as seen from outside (looking toward -Z)
                "v": [(0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)],
                "uv": [(0, 0), (1, 0), (1, 1), (0, 1)],
            },
            # -X left (face 4)
            {
                "n": (-1, 0, 0), "id": 4,
                "v": [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)],
                "uv": [(0, 0), (1, 0), (1, 1), (0, 1)],
            },
            # +X right (face 3)
            {
                "n": (1, 0, 0), "id": 3,
                "v": [(0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)],
                "uv": [(0, 0), (1, 0), (1, 1), (0, 1)],
            },
            # -Y bottom (face 2)
            {
                "n": (0, -1, 0), "id": 2,
                "v": [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)],
                "uv": [(0, 0), (1, 0), (1, 1), (0, 1)],
            },
            # +Y top (face 5)
            {
                "n": (0, 1, 0), "id": 5,
                "v": [(-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)],
                "uv": [(0, 0), (1, 0), (1, 1), (0, 1)],
            },
        ]

        # Build vertex array: two triangles per face (0,1,2) and (0,2,3)
        verts = []
        for f in faces:
            fid = f["id"]
            (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = f["v"]
            (u0, v0), (u1, v1), (u2, v2), (u3, v3) = f["uv"]

            # Two triangles forming a quad
            quad = [
                (x0, y0, z0, u0, v0, fid),
                (x1, y1, z1, u1, v1, fid),
                (x2, y2, z2, u2, v2, fid),
                (x0, y0, z0, u0, v0, fid),
                (x2, y2, z2, u2, v2, fid),
                (x3, y3, z3, u3, v3, fid),
            ]
            for tup in quad:
                verts.extend(tup)

        return np.array(verts, dtype=np.float32)

    def get_vertex_shader(self):
        """Get the vertex shader source code"""
        return '''
        #version 330
        in vec3 position;
        in vec2 texcoord;
        in float face_id;
        uniform mat4 mvp;
        out vec2 v_uv;
        flat out int v_face;
        void main() {
            gl_Position = mvp * vec4(position, 1.0);
            v_uv = texcoord;
            v_face = int(face_id + 0.5);
        }
        '''

    def get_fragment_shader(self):
        """Get the fragment shader source code with procedural pip rendering"""
        return '''
        #version 330
        in vec2 v_uv;
        flat in int v_face;
        out vec4 fragColor;

        // Smooth circle SDF (anti-aliased)
        float dotMask(vec2 uv, vec2 center, float r, float aa) {
            float d = length(uv - center) - r;
            return 1.0 - smoothstep(-aa, aa, d);
        }

        // Rounded-rect edge factor (for bevel-ish darkening)
        // Returns ~1.0 at the outer rim, ~0.0 inside the face.
        float roundedEdge(vec2 uv, float corner_r, float width, float aa) {
            // Distance to the axis-aligned box with rounded corners
            // Map uv to centered coords
            vec2 c = uv - vec2(0.5);
            // Half-size of the face minus corner radius
            vec2 hs = vec2(0.5 - corner_r);
            // Signed distance to rounded box (Inigo Quilez style)
            vec2 q = abs(c) - hs;
            float outside = length(max(q, 0.0)) - corner_r;
            // Turn the *ring* near the edge into 1.0, interior 0.0
            // width controls how thick the dark rim appears.
            return 1.0 - smoothstep(width, width + aa, max(outside, 0.0));
        }

        // Classic die layout: opposite faces sum to 7
        // Face mapping: 1=+Z front, 6=-Z back, 4=-X, 3=+X, 2=-Y, 5=+Y
        void main() {
            vec2 uv = clamp(v_uv, 0.0, 1.0);

            // Pip centers (grid-ish layout)
            vec2 C  = vec2(0.50, 0.50);  // center
            vec2 TL = vec2(0.30, 0.70);  // top-left
            vec2 TR = vec2(0.70, 0.70);  // top-right
            vec2 BL = vec2(0.30, 0.30);  // bottom-left
            vec2 BR = vec2(0.70, 0.30);  // bottom-right
            vec2 ML = vec2(0.30, 0.50);  // middle-left
            vec2 MR = vec2(0.70, 0.50);  // middle-right

            float r = 0.085;      // pip radius
            float aa = 0.003;     // anti-alias width

            float pip = 0.0;
            if (v_face == 1) {                // 1 pip
                pip += dotMask(uv, C,  r, aa);
            } else if (v_face == 2) {         // 2 pips (diagonal)
                pip += dotMask(uv, TL, r, aa);
                pip += dotMask(uv, BR, r, aa);
            } else if (v_face == 3) {         // 3 pips (diagonal + center)
                pip += dotMask(uv, TL, r, aa);
                pip += dotMask(uv, C,  r, aa);
                pip += dotMask(uv, BR, r, aa);
            } else if (v_face == 4) {         // 4 pips (corners)
                pip += dotMask(uv, TL, r, aa);
                pip += dotMask(uv, TR, r, aa);
                pip += dotMask(uv, BL, r, aa);
                pip += dotMask(uv, BR, r, aa);
            } else if (v_face == 5) {         // 5 pips (corners + center)
                pip += dotMask(uv, TL, r, aa);
                pip += dotMask(uv, TR, r, aa);
                pip += dotMask(uv, BL, r, aa);
                pip += dotMask(uv, BR, r, aa);
                pip += dotMask(uv, C,  r, aa);
            } else if (v_face == 6) {         // 6 pips (two columns)
                pip += dotMask(uv, vec2(ML.x, 0.70), r, aa);
                pip += dotMask(uv, vec2(ML.x, 0.50), r, aa);
                pip += dotMask(uv, vec2(ML.x, 0.30), r, aa);
                pip += dotMask(uv, vec2(MR.x, 0.70), r, aa);
                pip += dotMask(uv, vec2(MR.x, 0.50), r, aa);
                pip += dotMask(uv, vec2(MR.x, 0.30), r, aa);
            }

            // Base colors
            vec3 faceColor = vec3(1.0);  // white die
            vec3 pipColor  = vec3(0.0);  // black pips
            float pipMix = clamp(pip, 0.0, 1.0);
            vec3 col = mix(faceColor, pipColor, pipMix);

            // ---- Simple "bevel" & form without lighting ----
            // Rounded edge darkening ring
            float corner_r = 0.08;  // corner roundness (0..0.2 looks good)
            float rim_w    = 0.018; // rim thickness
            float rim_aa   = 0.003; // rim AA width
            float rim = roundedEdge(uv, corner_r, rim_w, rim_aa);
            float rimDark = 0.12;   // how dark the rim gets (0.0.. ~0.3)
            col *= (1.0 - rimDark * rim);

            // Gentle face falloff (radial from center) for a touch of depth
            float d = length(uv - vec2(0.5));
            float falloff = smoothstep(0.4, 0.75, d); // 0 at center, ~1 toward corners
            col *= (1.0 - 0.06 * falloff);

            // Add thin edge outline for better definition against grid
            float edge_threshold = 0.02;  // How close to edge
            float edge_aa = 0.004;        // Anti-alias width

            // Calculate distance to nearest edge
            vec2 edge_dist = min(uv, 1.0 - uv);
            float min_edge_dist = min(edge_dist.x, edge_dist.y);

            // Create edge outline
            float edge_outline = 1.0 - smoothstep(0.0, edge_threshold, min_edge_dist);
            edge_outline = smoothstep(0.0, edge_aa, edge_outline);

            // Mix in subtle dark outline
            vec3 outline_color = vec3(0.1, 0.1, 0.15); // Dark outline color
            col = mix(col, outline_color, edge_outline * 0.4);

            fragColor = vec4(col, 1.0);
        }
        '''

    def get_physics_properties(self):
        """Get physics properties for PyBullet"""
        return self.PHYSICS_PROPS.copy()

    def detect_top_face(self, orientation, physics_client_id):
        """
        Determine which face is pointing upward based on dice orientation

        Args:
            orientation: Quaternion from PyBullet getBasePositionAndOrientation
            physics_client_id: PyBullet physics client ID

        Returns:
            int: Face number (1-6) that is pointing up
        """
        # Convert quaternion to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Transform face normals to world space and find which points most upward
        world_up = np.array([0, 0, 1])
        max_dot = -1
        top_face = 1

        for i, normal in enumerate(self.face_normals):
            world_normal = rot_matrix @ normal
            dot_product = np.dot(world_normal, world_up)
            if dot_product > max_dot:
                max_dot = dot_product
                top_face = i + 1

        return top_face

    def create_physics_body(self, physics_client_id, start_pos, start_orientation):
        """
        Create a PyBullet physics body for this dice

        Args:
            physics_client_id: PyBullet physics client ID
            start_pos: Starting position [x, y, z]
            start_orientation: Starting quaternion [x, y, z, w]

        Returns:
            int: PyBullet body ID
        """
        props = self.get_physics_properties()

        # Create collision shape
        dice_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[props['size'], props['size'], props['size']],
            physicsClientId=physics_client_id
        )

        # Create physics body
        dice_id = p.createMultiBody(
            baseMass=props['mass'],
            baseCollisionShapeIndex=dice_collision,
            basePosition=start_pos,
            baseOrientation=start_orientation,
            physicsClientId=physics_client_id
        )

        # Set physics properties
        p.changeDynamics(
            dice_id, -1,
            restitution=props['restitution'],
            lateralFriction=props['lateral_friction'],
            rollingFriction=props['rolling_friction'],
            linearDamping=props['linear_damping'],
            angularDamping=props['angular_damping'],
            contactStiffness=props['contact_stiffness'],
            contactDamping=props['contact_damping'],
            physicsClientId=physics_client_id
        )

        return dice_id