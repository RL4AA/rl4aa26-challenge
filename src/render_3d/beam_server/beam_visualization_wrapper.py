import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import trimesh
import websockets
from dotenv import load_dotenv
from gymnasium import Wrapper

# Calculate the path to the .env file, one levels up from the script's location
script_dir = Path(__file__).resolve().parent  # Directory of the current script
env_path = script_dir.parent / ".env"  # Two levels up to beam_3d_visualizer/.env

# Load the .env file
load_dotenv(dotenv_path=env_path)

# Set logging level based on environment
debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Setup logging with conditional log level
log_level = (
    logging.DEBUG if debug_mode else logging.WARNING
)  # Set to WARNING to suppress info/debug logs
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

logger.info(f"Loaded .env from {env_path}")
logger.info(f"NODE_ENV: {os.getenv('NODE_ENV')}")
logger.info(f"VITE_FRONTEND_PORT: {os.getenv('VITE_FRONTEND_PORT')}")

# Define constants at module level
DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 5173
DEFAULT_NUM_PARTICLES = 1000
BEAM_SOURCE_COMPONENT = "cla_s04_sim_aper_01"
SCREEN_NAME = "cla_s04_dia_scr_03"
DEFAULT_SCREEN_BINNING = 4
DEFAULT_RENDER_MODE = "human"
DEFAULT_WS_HOST = "0.0.0.0"
DEFAULT_WS_PORT = 8081
DEFAULT_CONNECTION_TIMEOUT = 1.0
DEFAULT_SPREAD_SCALE_FACTOR = 15
DEFAULT_MEAN_SCALE_FACTOR = 10


class BeamVisualizationWrapper(Wrapper):
    """
    A Gym wrapper that encapsulates the beam simulation logic and manages the
    initialization of the JavaScript web application for 3D visualization.
    """

    def __init__(
        self,
        env: gym.Env,
        http_host: str = DEFAULT_HTTP_HOST,
        http_port: int = DEFAULT_HTTP_PORT,
        is_export_enabled: bool = True,
        num_particles: int = DEFAULT_NUM_PARTICLES,
        render_mode: str = DEFAULT_RENDER_MODE,
        ws_host: str = DEFAULT_WS_HOST,
        ws_port: int = DEFAULT_WS_PORT,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
        spread_scale_factor: float = DEFAULT_SPREAD_SCALE_FACTOR,
        mean_scale_factor: float = DEFAULT_MEAN_SCALE_FACTOR,
    ):
        """
        Initialize the BeamVisualizationWrapper.

        Args:
            env (gym.Env): The underlying Gym environment (e.g., BeamControlEnv).
            http_host (str): Hostname for the JavaScript web application server.
            http_port (int): Port for the web application server.
            is_export_enabled (bool): Whether to enable 3D scene export.
            num_particles (int): Number of particles to simulate in the beam.
        """
        super().__init__(env)

        # Basic configuration
        self.base_path = Path(__file__).resolve().parent
        self.http_host = http_host
        self.http_port = http_port
        self.render_mode = render_mode
        self.num_particles = num_particles
        self.current_step = 0
        self.web_process = None
        self.web_thread = None
        self.data = OrderedDict()
        self.is_export_enabled = is_export_enabled
        self.screen_reading = None

        # Store host and port
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.connection_timeout = connection_timeout

        # Validate port number
        if not isinstance(self.ws_port, int) or not (1 <= self.ws_port <= 65535):
            logger.warning(
                f"Invalid port number {self.ws_port}. Defaulting to {self.ws_port}."
            )

        # WebSocket management attributes
        self.clients = set()
        self.connected = False
        self.server = None

        # Data to be transmitted to the JavaScript web application
        self.spread_scale_factor = spread_scale_factor
        self.mean_scale_factor = mean_scale_factor

        self.stop_simulation = False

        # Start the WebSocket server in a separate thread
        self._lock = asyncio.Lock()

        # Initialize state
        self.incoming_particle_beam = None
        self.last_action = np.zeros(5, dtype=np.float32)

        # Try to get the segment from the backend if available
        if hasattr(self.unwrapped, "backend") and hasattr(
            self.unwrapped.backend, "segment"
        ):
            self.segment = self.unwrapped.backend.segment
        else:
            self.segment = self.unwrapped.segment

        # Ensures the necessary npm dependencies are installed
        self._setup()

        # Start the JavaScript web application (dev or prod mode)
        self._start_web_application()

        # Set up screen configuration
        self._initialize_screen()

    def _initialize_screen(self) -> None:
        """Initialize the screen configuration for beam visualization."""
        # Define screen
        self.screen_name = SCREEN_NAME

        # Try to get the segment from the backend if available
        if hasattr(self.unwrapped, "backend") and hasattr(
            self.unwrapped.backend, "segment"
        ):
            self.screen = getattr(self.unwrapped.backend.segment, self.screen_name)
        else:
            self.screen = getattr(self.unwrapped.segment, self.screen_name)

        self.screen.binning = DEFAULT_SCREEN_BINNING
        self.screen.is_active = True

        # Obtain screen boundaries
        self.screen_boundary = self.get_screen_boundary()

        # Update visualization data with screen boundaries
        self.data.update(
            {
                "screen_boundary_x": float(self.screen_boundary[0]),
                "screen_boundary_y": float(self.screen_boundary[1]),
            }
        )

    @property
    def render_mode(self):
        """Get the render mode."""
        return self._render_mode

    @render_mode.setter
    def render_mode(self, value):
        """Set the render mode."""
        self._render_mode = value

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment, reset last_action, and run the simulation.

        Args:
            seed (Optional[int]): Seed for random number generation.
            options (Optional[Dict]): Additional reset options.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info.
        """
        # Reset action state
        self.last_action = np.zeros(5, dtype=np.float32)
        self.current_step = 0

        # Reset the underlying environment
        observation, info = self.env.reset(seed=seed, options=options)

        # Read screen image
        self.screen_reading = info["backend_info"]["screen_image"]

        # Run simulation
        self._simulate()

        return observation, info

    def _initialize_particle_beam(self) -> None:
        """
        Initialize the incoming particle beam for simulation.

        Raises:
            ValueError: If the incoming particle beam cannot be initialized.
        """
        # Try to get the beam from the backend if available
        if hasattr(self.unwrapped, "backend") and hasattr(
            self.unwrapped.backend, "incoming"
        ):
            self.incoming_particle_beam = (
                self.unwrapped.backend.incoming.as_particle_beam(
                    num_particles=self.num_particles
                )
            )
        # Otherwise get it from the incoming_beam attribute
        elif hasattr(self.unwrapped, "incoming_beam"):
            self.incoming_particle_beam = self.unwrapped.incoming_beam.as_particle_beam(
                num_particles=self.num_particles
            )
        else:
            raise ValueError(
                "Cannot initialize incoming particle beam. Neither backend.incoming "
                "nor incoming_beam attributes found in the unwrapped environment."
            )

        if self.incoming_particle_beam is None:
            raise ValueError(
                "Incoming particle beam is None. Check beam initialization."
            )

        # Log the initial beam state for debugging
        logger.info(
            f"Initialized incoming particle beam with {self.num_particles} particles."
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute a step in the environment and run the simulation.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: Observation, reward,
                terminated, truncated, and info.
        """
        # Update last_action with the action being applied
        self.last_action = action.copy()

        # Execute step in the underlying environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Read screen image
        self.screen_reading = info["backend_info"]["screen_image"]

        # Run simulation
        self._simulate()

        info.update({"stop_simulation": self.data["stop_simulation"]})

        return observation, reward, terminated, truncated, info

    async def render(self):
        """
        Render the environment by preparing simulation data and broadcasting it
        via WebSocket.
        This method does not rely on the underlying environment's render method, as all
        visualization logic is handled by this wrapper.

        Note: The simulation data is already updated in step() or reset(),
        so we don't need to call _simulate() again here.
        """
        if self.render_mode != "human":
            return  # Skip rendering if not in human mode

        logger.debug("Broadcasting data to WebSocket clients...")
        results = await self.broadcast(self.data)
        for result in results:
            if isinstance(result, Exception):
                logger.exception("broadcast task failed", exc_info=result)
        logger.debug("Data broadcast completed.")

        # Add delay after broadcasting to allow animation to complete
        # before sending new data
        await asyncio.sleep(1.25)

    def close(self):
        """
        Close the wrapper and terminate the web application process.
        """
        # Terminate the web application process if it exists
        if self.web_process:
            self.web_process.kill()
            self.web_process.wait(timeout=5)
            print("Terminated JavaScript web application process.")
        if self.server:
            self.server.close()  # This will stop the WebSocket server and trigger cleanup
            print("Closed WebSocket server.")

        # Close the underlying environment
        super().close()

    def get_screen_boundary(self) -> np.ndarray:
        """
        Computes the screen boundary based on resolution and pixel size.

        The boundary is calculated as half of the screen resolution multiplied
        by the pixel size, giving the physical dimensions of the screen
        in meters.

        Returns:
            np.ndarray: The screen boundary as a 2D numpy array [width, height]
            in meters.
        """
        return np.array(self.screen.resolution) / 2 * np.array(self.screen.pixel_size)

    def _setup(self):
        """
        Automates the setup process by running npm install to install dependencies.
        This should be run once to ensure the JavaScript dependencies are installed.
        """
        try:
            # Path to the node_modules directory
            node_modules_path = os.path.join(self.base_path.parent, "node_modules")

            # Check if package.json exists to confirm we are in the correct directory
            package_json_path = os.path.join(self.base_path.parent, "package.json")
            if not os.path.exists(package_json_path):
                raise FileNotFoundError(
                    f"{package_json_path} not found."
                    f" Make sure you are in the correct project directory."
                )

            # Check if node_modules exists and is not empty
            if os.path.exists(node_modules_path) and os.listdir(node_modules_path):
                logger.info("Dependencies are already installed. Skipping npm install.")
            else:
                logger.info("Running npm install...")
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=self.base_path.parent,  # Run in directory with package.json
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=(
                        True if sys.platform == "win32" else None
                    ),  # Only use shell=True on Windows
                )

                # Log the output for debugging purposes
                if result.returncode == 0:
                    logger.info("npm install completed successfully.")
                else:
                    logger.error(f"npm install failed with error: {result.stderr}")
                    raise RuntimeError(f"npm install failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Error during setup: {e}")
            raise

    def _start_web_application(self):
        """
        Start the JavaScript web application (Vite development server)
        in a background thread.
        """

        def run_web_server():
            try:
                # Determine the mode and load the appropriate .env file
                node_env = os.getenv("NODE_ENV", "production")
                if node_env == "production":
                    env_file = script_dir.parent / ".env.production"
                    # Load with override existing vars to ensure latest values
                    load_dotenv(
                        dotenv_path=env_file,
                        override=True,
                    )

                logger.debug(f"Running in mode: {node_env}")

                if node_env == "development":
                    # Development mode: Start Vite dev server
                    # Start Vite development server
                    cmd = [
                        "npx",
                        "vite",
                        "--host",
                        self.http_host,
                        "--port",
                        str(self.http_port),
                    ]
                    logger.debug(
                        f"Starting Vite dev server"
                        f" on http://{self.http_host}:{self.http_port}"
                    )
                else:
                    # Production mode: Start Express server (server.js)
                    dist_path = self.base_path.parent / "dist"
                    if not dist_path.exists():
                        raise FileNotFoundError(
                            f"Pre-built dist folder not found at {dist_path}"
                        )
                    cmd = ["node", "server.js"]
                    logger.debug(
                        f"Starting Express server (server.js)"
                        f" on http://{self.http_host}:{self.http_port}"
                    )

                self.web_process = subprocess.Popen(
                    cmd,
                    cwd=self.base_path.parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    # Pass environment variables (e.g., PORT from .env)
                    env=os.environ.copy(),
                    shell=(
                        True if sys.platform == "win32" else None
                    ),  # Only use shell=True on Windows
                )

                # Log output for debugging
                for line in self.web_process.stdout:
                    logger.debug(f"Vite stdout: {line.strip()}")
                for line in self.web_process.stderr:
                    logger.error(f"Vite stderr: {line.strip()}")

            except Exception as e:
                logger.error(f"Failed to start web application: {e}")
                # Consider raising the exception here for better error handling

        # Start the web server in a background thread
        self.web_thread = threading.Thread(target=run_web_server, daemon=True)
        self.web_thread.start()

        # Give the server a moment to start
        logger.debug(
            f"JavaScript web application setup initiated on "
            f"http://{self.http_host}:{self.http_port}"
        )

    def _simulate(self) -> None:
        """
        Calculate the positions of beam segments with dynamic angles.

        This method tracks the particle beam through each element in the segment,
        computing the positions of particles at each step. The beam travels along
        the x-axis, with position variations in the yz-plane. The simulation
        data is stored in self.data for later use in visualization.
        """
        # Reset segments data for this simulation step
        self.data["segments"] = []
        pos = 0
        # Reinitialize the incoming particle beam to ensure a fresh start
        # (i.e. initial state) at AREASOLA1
        self._initialize_particle_beam()

        # Track beam through each lattice element
        references = [self.incoming_particle_beam]
        for element in self.segment.elements:
            # Track beam through this element
            # Use the output beam of the previous segment as the input
            # for the next lattice section
            outgoing_beam = element.track(references[-1])
            references.append(outgoing_beam)

            # Extract particle positions
            x = -outgoing_beam.particles[:, 0]  # Column 0
            y = outgoing_beam.particles[:, 2]  # Column 2
            z = -outgoing_beam.particles[:, 4]  # Column 4

            # Note: In Cheetah, the coordinates of the particles are defined
            # by a 7-dimensional vector: x = (x, p_x, y, p_y, 𝜏, 1),
            # where 𝜏 = t - t_0 represents the time offset of a particle
            # relative to the reference particle.
            #
            # Since we use z to represent the `longitudinal position` of particles
            # in the beamline (instead of time offset), we flip the sign of 𝜏.
            #
            # This ensures that particles:
            # - `ahead` of the reference particle (bunch head) have `positive` z,
            # - `behind` the reference particle (bunch tail) have `negative` z.
            #
            # This sign convention aligns with spatial representations
            # of beam bunches, where a leading particle has a larger
            # longitudinal position z.
            #
            # Source:
            # https://cheetah-accelerator.readthedocs.io/en/latest/coordinate_system.html

            # Shift beam particles 3D position in reference to segment component
            positions = torch.stack([x, y, z + pos], dim=1)

            # Compute the mean position of the bunch
            mean_position = positions.mean(dim=0, keepdim=True)

            # Scale the spread (deviation from mean) using spread_scale_factor
            spread_scaled = (positions - mean_position) * self.spread_scale_factor

            # Scale the mean position using mean_scale_factor
            # Note: We only scale x and y components, leaving z unchanged
            mean_scaled = mean_position.clone()

            positions = spread_scaled + mean_scaled

            # Store segment data
            self.data["segments"].append(
                {
                    "segment_name": element.name,
                    "segment_type": element.__class__.__name__,
                    "particle_positions": positions.tolist(),
                    "element_position": pos,
                }
            )

            pos += element.length.item()

        self.current_step += 1

        # Update meta info to include particle reading from segments
        self.data.update(
            {
                "screen_reading": self.screen_reading.tolist(),
                "bunch_count": self.current_step,
                "stop_simulation": self.stop_simulation,
            }
        )

    def _amplify_displacement(
        self, x: torch.Tensor, amplification_factor: float = 50.0
    ) -> torch.Tensor:
        """
        Apply linear amplification to a displacement value to enhance small changes.

        Args:
            x (torch.Tensor): The input displacement value (in meters).
            amplification_factor (float): The factor by which to amplify
                small displacements (e.g., 10 or 100).

        Returns:
            torch.Tensor: The amplified displacement value.
        """
        return torch.sign(x) * amplification_factor * torch.abs(x)

    async def start_websocket(self):
        """Run the WebSocket server."""
        self.server = await websockets.serve(
            self._handle_client,
            host=self.ws_host,
            port=self.ws_port,
        )
        logger.debug(f"WebSocket server running on ws://{self.ws_host}:{self.ws_port}")
        await self.server.wait_closed()

    async def _handle_client(
        self, websocket: websockets.WebSocketServerProtocol, path: str = None
    ):
        """Handle incoming WebSocket connections and messages."""
        async with self._lock:
            logger.debug("acquired lock for client add")
            self.connected = True
            self.clients.add(websocket)
        logger.debug("WebSocket connection established.")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.debug(f"Received data: {data}")

                    if "controls" in data:
                        # Update the control parameters based on the WebSocket data
                        controls = data.get("controls", {})

                        self.spread_scale_factor = controls.get("scaleBeamSpread", 0.0)
                        self.mean_scale_factor = controls.get("scaleBeamPosition", 0.0)
                        self.stop_simulation = controls.get("stopSimulation", False)

                        hvcor_01_h = controls.get("HVCOR_01_H", 0.0)
                        hvcor_01_v = controls.get("HVCOR_01_V", 0.0)
                        quad_01 = controls.get("QUAD_01", 0.0)
                        quad_02 = controls.get("QUAD_02", 0.0)
                        quad_03 = controls.get("QUAD_03", 0.0)

                        # Store the control action as a numpy array
                        control_action = np.array(
                            [hvcor_01_h, hvcor_01_v, quad_01, quad_02, quad_03],
                            dtype=np.float32,
                        )

                        logger.debug(
                            "Received controls: HVCOR_01_H={}, HVCOR_01_V={}, "
                            "QUAD_01={}, QUAD_02={}, QUAD_03={}".format(
                                hvcor_01_h,
                                hvcor_01_v,
                                quad_01,
                                quad_02,
                                quad_03,
                            )
                        )
                        observation, reward, terminated, truncated, info = self.step(
                            control_action
                        )

                        # Render and broadcast data to clients
                        await self.render()

                        if info[
                            "stop_simulation"
                        ]:  # Stop the simulation if truncated is True
                            print("Truncated flag is True, stopping simulation...")
                            self.close()
                            break
                except json.JSONDecodeError:
                    logger.error("Error: Received invalid JSON data.")
        except asyncio.exceptions.CancelledError:
            logger.debug("WebSocket task was cancelled.")
            raise
        except websockets.ConnectionClosed:
            logger.debug("WebSocket connection closed by client.")
        finally:
            async with self._lock:
                logger.debug("acquired lock for client shutdown")
                self.clients.discard(websocket)
                if not self.clients:
                    self.connected = False
            logger.debug("Client cleanup completed.")

    async def broadcast(self, message: Dict):
        """Safely broadcast a message to all connected clients."""
        if message is None:
            logger.warning("No data to broadcast.")
            return

        tasks = []
        async with self._lock:
            logger.debug("acquired lock for broadcast")
            if not self.clients:
                logger.debug("No clients connected, skipping broadcast.")
                self.connected = False
                return
            tasks = [self.safe_send(client, message) for client in self.clients]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def safe_send(self, client, message):
        try:
            async with asyncio.timeout(2.5):
                await client.send(json.dumps(message))
        except asyncio.TimeoutError:
            logger.warning("WebSocket send timed out.")
        except websockets.ConnectionClosed:
            logger.debug("WebSocket connection closed during broadcast.")
        except asyncio.CancelledError:
            logger.debug("WebSocket task was cancelled.")
            raise
