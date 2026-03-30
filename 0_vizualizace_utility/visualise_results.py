import pickle
import open3d as o3d
import numpy as np
import pandas as pd
import os
import PIL

min_score = 0.09

ids_csv = "./data/oi-eq-t1-v0/name_id_map.csv"
points_folder = "./data/oi-eq-t1-v0/points"
labels_folder = "./data/oi-eq-t1-v0/labels"
# result_file = "./tools/results/04-02-wtf-2/eval/predictions.pkl"
result_file = "./tools/sampleresults_test/powerpoint/predictions.pkl"

do_screenshots = False
screenshot_path = "./screens/septtest/"

limit_samples = None
skip_empty = False

def dummy_box(box):
    output = []
    for b in box:
        output.append(box_center_to_corner(b))
    return output

def get_boxes_from_pkl(path: str, index: int):
    with open(path, "rb") as fp:
        results = pickle.load(fp)[index]
    print(results['frame_id'], flush=True)
    output = []
    COUNTER = 0
    for row, score in zip(results["boxes_lidar"], results["score"]):
        if score < min_score:
            continue
        nums = get_box_nums(row)
        if len(nums) > 0:
            output.append(box_center_to_corner(nums))
        COUNTER += 1
    return output, results["frame_id"]

def get_gt_boxes(index: str):
    with open(os.path.join(labels_folder, f"{index}.txt"), "r") as f:
        labels = f.readlines()
    output = []
    COUNTER = 0
    for row in labels:
        row = row.strip().split(" ")
        if row[7] != "Plant":
            continue
        row = np.array(row[:7], dtype=np.float64)
        nums = get_box_nums(row)
        if len(nums) > 0:
            output.append(box_center_to_corner(nums))
        COUNTER += 1
    return output

def get_boxes(path: str):
    raw_labels = open(path, "r").read().split("\n")
    output = []
    for line in raw_labels:
        raw_line_vals = line.split(" ")
        nums = get_box_nums(raw_line_vals)
        if len(nums) > 0:
            output.append(box_center_to_corner(nums))
    return output

def get_box_nums(box):
    if len(box) < 7:
        print(f"This line only has {len(box)} values.")
        return []
    box = box[:7]
    output = []
    for val in box:
        try:
            output.append(float(val))
        except ValueError:
            print(f"The value {val} cannot be converted to a number.")
            return []
    return output
        
def box_center_to_corner(box):
    translation = box[0:3]
    #h, w, l = box[3], box[4], box[5]
    l, w, h = box[3], box[4], box[5]
    rotation = box[6]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()
    return corner_box.transpose()

SHIFT = 0
ORDER_COLORS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 1]]

def get_lines(boxes, color):
    output = []
    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    # Use the same color for all lines
    colors = [color for _ in range(len(lines))]
    
    i = -SHIFT * len(ORDER_COLORS)
    for b in boxes:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(b)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # colors = [ORDER_COLORS[i] if (i < len(ORDER_COLORS) and i >= 0) else [1, 1, 1] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        output.append(line_set)
        i += 1
    return output

def get_pkl_len(path):
    with open(path, "rb") as fp:
        return len(pickle.load(fp))

def create_geos(i):
    pred_boxes, file_id = get_boxes_from_pkl(result_file, i)
    gt_boxes = get_gt_boxes(file_id)
    gt_lines = get_lines(gt_boxes, [0, 0, 0])
    pred_lines = get_lines(pred_boxes, [1, 0, 0])
    # pred_lines = []
    if len(pred_lines) == 0 and skip_empty:
        return None, file_id
    geos = []
    geos.extend(gt_lines)
    geos.extend(pred_lines)
    pcd = o3d.geometry.PointCloud()
    npy = np.load(os.path.join(points_folder, file_id) + ".npy")
    pcd.points = o3d.utility.Vector3dVector(npy)
    geos.append(pcd)
    return geos, file_id

def edit_screenshot(path, file_id, i):
    with open(result_file, "rb") as fp:
        results = pickle.load(fp)[i]
    count_pred = len(results["pred_labels"])
    with open(os.path.join(labels_folder, f"{file_id}.txt"), "r") as f:
        labels = f.readlines()
    count_gt = len(labels)
    csv = pd.read_csv(ids_csv, sep=",")
    real_name = csv[csv["id"] == int(file_id)]["name"].item()
    
    img = PIL.Image.open(path)
    draw = PIL.ImageDraw.Draw(img)
    txt = f"{file_id} // {real_name}\n" \
        f"Ground truth boxes (black): {count_gt}\n" \
        f"Prediction boxes (red): {count_pred}"
    font = PIL.ImageFont.load_default(24)
    draw.text((0, 0), txt, fill=(0,0,0), font=font)
    img.save(path)

def display(geos, file_id, i):    
    # o3d.visualization.draw_geometries(geos)
    def screenshot(vis):
        path = f"{screenshot_path}/./{file_id}.png"
        vis.capture_screen_image(path)
        vis.destroy_window()
        edit_screenshot(path, file_id, i)
        return False
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geo in geos:
        vis.add_geometry(geo)
    if do_screenshots:
        vis.register_animation_callback(screenshot)
    # vis.get_render_option().background_color = [0.148] * 3
    vis.run()
    vis.destroy_window()

def main():
    if do_screenshots:
        os.makedirs(screenshot_path, exist_ok=True)
    
    num_samples = get_pkl_len(result_file) if limit_samples is None else limit_samples
    for i in range(num_samples):
        i = 36
        geos, file_id = create_geos(i)
        if geos is None:
            continue
        display(geos, file_id, i)
        break
      
if __name__ == "__main__":
    main()
#
#
#

