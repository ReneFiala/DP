import os, argparse, sys
import open3d as o3d
import numpy as np
from pathlib import Path
import random
import PIL.Image, PIL.ImageDraw, PIL.ImageFont

FILTER_CLASSES = [
    
    # "SegmentationPlant", "Plant",
    # "Plant",
    "Leaf", "Embryonic_leaf", "Petiole", "Stem"
    # if a class is here, ignore its boxes
]

BOX_COLORS = {
    "SegmentationPlant": (0, 0, 0), # black
    "Plant": (1, 1, 1), # dark red
    "Stem": (0, 1, 1), # white
    "Leaf": (0, 1, 0),  # lumpy gween
    "Embryonic_leaf": (1, 1, 0), # yellow
    "Petiole": (1, 0, 1) # purple
    # none: (1, 0, 0) red
}
RANDOM_SHIFT = 0
"""
BOX_COLORS = {
    "Plant": (0.0, 0.0, 0.0), # dark red
    "Stem": (0, 0.5, 0.5), # white
    "Leaf": (0, 0.8, 0),  # lumpy gween
    "Embryonic_leaf": (0.5, 0.5, 0), # yellow
    "Petiole": (0.7, 0, 0.7) # purple
    # none: (1, 0, 0) red
}
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs="?", default="./",
                        help="Path to the folder with PCD files. (default: %(default)s)")
    parser.add_argument('--auto', "-a", action='store_true',
                        help="Skip all visualisations and only do automatic checks.")
    parser.add_argument('--manual', "-m", action='store_true',
                        help="Skip all automatic checks and only do visualisation with manual checks. If neither or both --auto and --manual are set, use both.")
    parser.add_argument("--output", "-o", type=Path, metavar='PATH', default=None,
                        help="Output text file for reports.")
    parser.add_argument('--threshold', "-t", type=float, metavar='T', default=3.0,
                        help="Threshold factor of median size to use for automatic scale check. (default: %(default)s)")
    parser.add_argument('--small', "-s", action='store_true',
                        help="During automatic scale check also check for smallest boxes being less than 1/T the scale of the median box.")
    parser.add_argument("--filter", "-f", type=str, metavar='PATH', default=None, nargs="+",
                        help="Check only specified files.")
    return parser.parse_args()

reports = []
def report(message):
    if message in reports:
        return
    reports.append(message)
    if args.output:
        if not args.output.parent.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "a+") as fp:
            fp.write(message + "\n")
    print(message, flush=True)

def get_filenames():
    files = os.listdir(args.input)
    files = [os.path.join(args.input, f)[:-4]
             for f in files
             if f.endswith(".pcd")
             and (args.filter is None or f in args.filter)]
    return files

def get_labels_file(filename):
    path = filename + ".txt"
    if os.path.exists(path):
        return path
    path = filename + ".pcd.txt"
    if os.path.exists(path):
        return path
    report(f"File error:\tNo labels file associated with cloud file.\t{os.path.split(filename)[1]}.pcd")
    return None

def build_geos(cloud_name, label_path):
    boxes, colors = get_corners(label_path)
    geos = get_lines(boxes, colors)
    cloud_path = os.path.join(os.path.split(label_path)[0], cloud_name)
    pcd = o3d.io.read_point_cloud(cloud_path + ".pcd")
    pcd.colors = o3d.utility.Vector3dVector(np.empty([0, 3]))
    geos.append(pcd)
    return geos

def check_duplicate_lines(cloud_name, label_path):
    _, tail = os.path.split(label_path)
    with open(label_path) as f:
        seen = set()
        seen_dict = {}
        for i, line in enumerate(f):
            if line in seen:
                report(f"Dupe check:\tLabel at line {i+1} identical to one at line {seen_dict[line]+1}.\t{tail}")
                add_to_suspects(cloud_name)
            else:
                seen.add(line)
                seen_dict.update({line: i})
    pass

def check_outliers(cloud_name, label_path):
    _, tail = os.path.split(label_path)
    volumes = get_box_volumes(label_path)
    if len(volumes) == 0:
        return
    bot = np.min(volumes)
    med = np.median(volumes)
    top = np.max(volumes)
    ratio = top / med
    if ratio > args.threshold:
        report(f"Scale check:\tLargest box {ratio:.1f} times larger than median.\t{tail}")
        add_to_suspects(cloud_name)
    if not args.small:
        return
    ratio = med / bot
    if ratio > args.threshold:
        report(f"Scale check:\tSmallest box {ratio:.1f} times smaller than median.\t{tail}")
        add_to_suspects(cloud_name)

def get_corners(path):
    raw_labels = open(path, "r").read().split("\n")
    out_boxes, out_colors = [], []
    for line in raw_labels:
        raw_line_vals = line.split(" ")
        nums, color = get_box_nums(raw_line_vals)
        if len(nums) < 7:
            continue
        out_boxes.append(box_center_to_corner(nums))
        out_colors.append(color)
    return out_boxes, out_colors

def get_box_volumes(path: str):
    raw_labels = open(path, "r").read().split("\n")
    output = []
    for line in raw_labels:
        raw_line_vals = line.split(" ")
        nums, _ = get_box_nums(raw_line_vals)
        if len(nums) < 7:
            continue
        try:
            l, w, h = float(nums[3]), float(nums[4]), float(nums[5])
            output.append(l * w * h)
        except Exception:
            continue
    return output

def get_box_nums(box):
    if len(box) < 7:
        if len(box) == 1 and box[0] == "":
            return [], []
        report(f"This line only has {len(box)} values:\n{box}")
        return [], []
    if box[7] in FILTER_CLASSES:
        return [], []
    output = []
    for val in box[:7]:
        try:
            output.append(float(val))
        except ValueError:
            report(f"The value {val} cannot be converted to a number.")
            return [], []
    color = BOX_COLORS.get(box[7], (1, 0, 0))
    return output, color
        
def box_center_to_corner(box):
    translation = box[0:3]
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

def get_lines(boxes, colors):
    output = []
    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    # Use the same color for all lines
    # colors = [[1, 0, 0] for _ in range(len(lines))]
    
    for b, c in zip(boxes, colors):
        rnd = [random.random() * RANDOM_SHIFT for _ in range(3)]
        b = [x + rnd for x in b]
        colors = [c for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(b)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        output.append(line_set)
    return output

def create_vis(filename):
    return MyVis(filename).run()

def add_to_suspects(name):
    global suspects
    name = name + ".pcd"
    if name in suspects:
        return
    suspects.append(name)

suspects = []
def generate_vis_args():
    if len(suspects) == 0:
        return
    output = f'python cloud_checker.py "{args.input}" --manual --filter'
    for s in suspects:
        output += f' "{s}"'
    print(f"Show only the suspects in visualise_all.py with this:\n{output}", flush=True)


BG_DEFAULT = (0.1484, 0.1484, 0.1484)
#BG_DEFAULT = (1, 1, 1)
MIX_EXIT = (0.5, 0.5, 0.5)
MIX_REPORT = (2.0, 0.6, 0.4)
MIX_BACK = (1.2, 2.0, 1.4)

class MyVis:
    def __init__(self, num, filename, direction):
        self.abort = False
        self.report = False
        self.num = num
        self.filename = filename
        self.direction = direction
    
    def get_exit_dict(self):
        return {
            'direction': self.direction,
            'abort': self.abort,
            'report': self.report
        }
    
    def run(self):
        _, cloud_name = os.path.split(self.filename)
        label_path = get_labels_file(self.filename)
        if not label_path:
            return self.get_exit_dict()
        if args.auto:
            check_duplicate_lines(cloud_name, label_path)
            check_outliers(cloud_name, label_path)
        if args.manual:
            geos = build_geos(cloud_name, label_path)
            if geos:
                self.create_vis(geos, cloud_name)
        return self.get_exit_dict()
    
    def update_background(self, vis):
        def multiply_rgb(a, b):
            return (a[0]*b[0], a[1]*b[1], a[2]*b[2])
        color = BG_DEFAULT
        if self.report:
            color = multiply_rgb(color, MIX_REPORT)
        if self.abort:
            color = multiply_rgb(color, MIX_EXIT)
        elif self.direction == -1:
            color = multiply_rgb(color, MIX_BACK)
        vis.get_render_option().background_color = color
    
    def show_warning(self, vis):
        self.report = not self.report
        self.update_background(vis)
        return True
    
    def quit_vis(self, vis):
        self.abort = not self.abort
        self.update_background(vis)
        return True
    
    def backwards(self, vis):
        self.direction *= -1
        self.update_background(vis)
        return True
    
    def take_screenshot(self, vis):
        try:
            path = f"./{self.filename}.png"
            vis.capture_screen_image(path)
            img = PIL.Image.open(path)
            draw = PIL.ImageDraw.Draw(img)
            font = PIL.ImageFont.load_default(24)
            draw.text((0, 0), self.filename, fill=(0,0,0), font=font)
            i = 0
            for label, color in BOX_COLORS.items():
                if label in FILTER_CLASSES:
                    continue
                draw.text((1920-300, i*24), label, font=font, fill=(0,0,0))
                pil_color = tuple([int(256*x) for x in color])
                draw.rectangle([1920-50, i*24, 1920, i*24+24], fill=pil_color)
                i += 1
            img = img.resize((2560, 1440))
            img.save(path)
            print(f"Saved screenshot to {path}", flush=True)
        except Exception as e:
            print(e, flush=True)
        vis.destroy_window()
        return False
        
    def screenshot_all(self, vis):
        ctr = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters("./cam_quarter.json")
        ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
        vis.register_animation_callback(self.take_screenshot)
        return True
    
    def create_vis(self, geos, cloud_name):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=cloud_name)
        for geo in geos:
            vis.add_geometry(geo)
        self.update_background(vis)
        vis.register_key_callback(32,  self.show_warning)     # Space
        vis.register_key_callback(81,  self.quit_vis)         # Q
        vis.register_key_callback(69,  self.backwards)        # E
        vis.register_key_callback(300, self.screenshot_all)  # F11
        vis.register_key_callback(301, self.take_screenshot)  # F12
        
        """
        if self.num != 0:
            ctr = vis.get_view_control()
            parameters = o3d.io.read_pinhole_camera_parameters("./cam_quarter.json")
            ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
            vis.register_animation_callback(self.take_screenshot)
        """
        
        vis.run()
        vis.destroy_window()

def main():
    print("Cloud checker v1.2 2024-09-16", flush=True)
    if not args.auto and not args.manual:
        args.auto = True
        args.manual = True
    if args.auto:
        print("Automatic mode active: Each sample will be checked for duplicate labels and box sizes.", flush=True)
    if args.manual:
        print("Vis mode controls:\n  [Left Mouse] Orbit view\n  [Shift] Roll view\n  [Middle Mouse] Pan view\n  [Space] Queue marking manual suspect\n  [Esc] Close sample\n  [E] Queue loading previous sample\n  [Q] Queue exiting early\n  [F12] Take screenshot to script folder with sample name and label legend\n[Space], [E], and [Q] can be undone by pressing them again. Their actions happen only upon pressing [Esc]. The background colour changes to indicate this.", flush=True)
    filenames = get_filenames()
    
    f, direction = 0, 1
    while f < len(filenames):
        ret = MyVis(f, filenames[f], direction).run()
        if ret['report']:
            tail = os.path.split(filenames[f])[1]
            report(f"User report:\t{tail}")
            add_to_suspects(tail)
        if ret['abort']:
            report("Exited early.")
            break
        direction = ret['direction']
        f += direction
        if f < 0:
            f = 0
            direction = 1
    generate_vis_args()

    
if __name__ == "__main__":
    args = parse_args()
    main()
        
