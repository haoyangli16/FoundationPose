import open3d as o3d
import sys


def visualize_ply(file_path):
    # Load the PLY file
    try:
        point_cloud = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return False

    # Print basic information about the point cloud
    print(f"Points: {len(point_cloud.points)}")
    print(f"Has colors: {point_cloud.has_colors()}")
    print(f"Has normals: {point_cloud.has_normals()}")

    # Visualize the point cloud
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(point_cloud)

    # Set visualization options
    opt = viewer.get_render_option()
    # opt.background_color = [0.1, 0.1, 0.1]  # Dark gray background

    opt.point_size = 1.0

    # Update and run the visualizer
    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python visualize_ply.py <path_to_ply_file>")
    #     sys.exit(1)

    # ply_file = sys.argv[1]
    ply_file = "/home/haoyang/project/haoyang/FoundationPose/debug/scene_complete.ply"
    visualize_ply(ply_file)
