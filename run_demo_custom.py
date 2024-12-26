import os
import logging
import numpy as np
import trimesh
import cv2
import open3d as o3d
import imageio.v2 as imageio
import argparse
from pathlib import Path

from estimater import *
from datareader import *
import argparse


class DataReader:
    """Custom data reader for your dataset structure."""

    def __init__(
        self, data_dir: str, mask_name: str = "source_mask.png", shorter_side: int = 480
    ):
        """
        Initialize the data reader.

        Args:
            data_dir: Path to the data directory
            mask_name: Name of the mask file (default: source_mask.png)
            shorter_side: Length of shorter side after resizing (default: 480)
        """
        self.data_dir = Path(data_dir)
        self.mask_name = mask_name
        self.shorter_side = shorter_side

        # Load camera intrinsics
        self.K = np.loadtxt(self.data_dir / "cam_K.txt")

        # Get sorted lists of RGB and depth files
        self.rgb_files = sorted((self.data_dir / "rgb").glob("*.png"))
        self.depth_files = sorted((self.data_dir / "depth").glob("*.png"))

        # Read first image to get original dimensions
        first_img = cv2.imread(str(self.rgb_files[0]))
        self.orig_h, self.orig_w = first_img.shape[:2]

        # Calculate resize scale
        self.scale = self.shorter_side / min(self.orig_h, self.orig_w)
        self.new_h = int(self.orig_h * self.scale)
        self.new_w = int(self.orig_w * self.scale)

        # Scale camera intrinsics
        self.K = self.K.copy()
        self.K[0:2] *= self.scale

        assert len(self.rgb_files) == len(
            self.depth_files
        ), "Mismatch in number of RGB and depth files"
        self.id_strs = [f.stem for f in self.rgb_files]

    def __len__(self):
        return len(self.rgb_files)

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio."""
        if img.shape[:2] != (self.new_h, self.new_w):
            img = cv2.resize(
                img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR
            )
        return img

    def get_color(self, idx: int) -> np.ndarray:
        """Read and resize RGB image."""
        img = cv2.imread(str(self.rgb_files[idx]))
        img = self._resize_image(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_depth(self, idx: int) -> np.ndarray:
        """Read, resize and convert depth image to meters."""
        depth = cv2.imread(str(self.depth_files[idx]), cv2.IMREAD_ANYDEPTH)
        depth = self._resize_image(depth)
        return depth.astype(float) / 1000.0  # convert to meters

    def get_mask(self) -> np.ndarray:
        """Read and resize object mask."""
        mask_path = self.data_dir / "mask" / self.mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = self._resize_image(mask)
        return mask.astype(bool)


class PoseEstimator:
    """Wrapper for Foundation Pose estimation."""

    def __init__(self, mesh_path: str, debug_dir: str = "debug", debug_level: int = 1):
        """
        Initialize the pose estimator.

        Args:
            mesh_path: Path to the object mesh file
            debug_dir: Directory for debug outputs
            debug_level: Level of debug visualization (0=none, 1=basic, 2=detailed)
        """
        self.debug_dir = Path(debug_dir)
        self.debug_level = debug_level

        # Create debug directories
        if debug_level > 0:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "track_vis").mkdir(exist_ok=True)
            (self.debug_dir / "ob_in_cam").mkdir(exist_ok=True)

        # Load and process mesh
        self.mesh = trimesh.load(mesh_path)
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(
            2, 3
        )

        # Initialize Foundation Pose components
        self.glctx = dr.RasterizeCudaContext()
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()

        # Create estimator
        self.estimator = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=str(self.debug_dir),
            debug=debug_level,
            glctx=self.glctx,
        )

        logging.info("Pose estimator initialized")

    def process_sequence(
        self, reader: DataReader, est_refine_iter: int = 5, track_refine_iter: int = 2
    ):
        """
        Process the full sequence of frames.

        Args:
            reader: DataReader instance
            est_refine_iter: Number of refinement iterations for initial pose estimation
            track_refine_iter: Number of refinement iterations for tracking
        """
        poses = {}

        for i in range(len(reader)):
            logging.info(f"Processing frame {i}")

            color = reader.get_color(i)
            depth = reader.get_depth(i)

            # Initial pose estimation for first frame
            if i == 0:
                mask = reader.get_mask()
                pose = self.estimator.register(
                    K=reader.K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=est_refine_iter,
                )

                # Debug output for first frame
                if self.debug_level >= 3:
                    self._save_debug_data(color, depth, pose, reader.K)

            # Tracking for subsequent frames
            else:
                pose = self.estimator.track_one(
                    rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter
                )

            # Save pose
            poses[reader.id_strs[i]] = pose
            np.savetxt(
                self.debug_dir / "ob_in_cam" / f"{reader.id_strs[i]}.txt",
                pose.reshape(4, 4),
            )

            # Visualization
            if self.debug_level >= 1:
                self._visualize_frame(color, pose, reader.K, reader.id_strs[i])

        return poses

    def _save_debug_data(self, color, depth, pose, K):
        """Save additional debug data for the first frame."""
        m = self.mesh.copy()
        m.apply_transform(pose)
        m.export(self.debug_dir / "model_tf.obj")

        xyz_map = depth2xyzmap(depth, K)
        valid = depth >= 0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(str(self.debug_dir / "scene_complete.ply"), pcd)

    def _visualize_frame(self, color, pose, K, frame_id):
        """Create and save visualization for a frame."""
        center_pose = pose @ np.linalg.inv(self.to_origin)

        # Draw bounding box
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=self.bbox)

        # Draw coordinate axes
        vis = draw_xyz_axis(
            color,
            ob_in_cam=center_pose,
            scale=0.1,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        # Show visualization
        cv2.imshow("Tracking", vis[..., ::-1])
        cv2.waitKey(1)

        # Save visualization
        if self.debug_level >= 2:
            imageio.imwrite(self.debug_dir / "track_vis" / f"{frame_id}.png", vis)


def main():
    parser = argparse.ArgumentParser(description="Foundation Pose Estimation")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--mesh_path", type=str, required=True, help="Path to object mesh file"
    )
    parser.add_argument(
        "--mask_name", type=str, default="source_mask.png", help="Name of the mask file"
    )
    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=5,
        help="Initial pose estimation refinement iterations",
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=2,
        help="Tracking refinement iterations",
    )
    parser.add_argument(
        "--debug", type=int, default=1, help="Debug level (0=none, 1=basic, 2=detailed)"
    )
    parser.add_argument(
        "--debug_dir", type=str, default="debug", help="Debug output directory"
    )

    args = parser.parse_args()

    # Set up logging and random seed
    set_logging_format()
    set_seed(0)

    # Initialize data reader and pose estimator
    reader = DataReader(args.data_dir, args.mask_name)
    estimator = PoseEstimator(args.mesh_path, args.debug_dir, args.debug)

    # Process sequence
    poses = estimator.process_sequence(
        reader,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
    )

    logging.info("Processing completed")


if __name__ == "__main__":
    main()


"""
python script.py --data_dir /home/haoyang/project/haoyang/pour_mani/vision/demo_data/0001 --mesh_path /home/haoyang/project/haoyang/pour_mani/vision/demo_data/0001/mesh/sparking_water_can.obj
"""
