import numpy as np
import cv2
import tcod
from sklearn.neighbors import KDTree
from skimage.draw import line
import networkx as nx
from graph_utils import nms_points


IMAGE_SIZE = 2048
SAMPLE_MARGIN = 64

def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

# returns (x, y)
def get_points_and_scores_from_mask(mask, threshold):
    rcs = np.column_stack(np.where(mask > threshold)) # n rows, 2 columns
    xys = rcs[:, ::-1]
    scores = mask[mask > threshold]
    return xys, scores

def remove_small_regions(mask: np.ndarray, area_threshold: float, mask_threshold: float) -> np.ndarray:
    """
    remove small regions from mask, these are usually considered as noise
    
    Args:
        mask: input mask, type is ndarray, value is 0-1 float32,
             shape can be (B, H, W) or (H, W)
        area_threshold: area threshold, regions with area less than this value will be removed
        mask_threshold: threshold, regions with value greater than this value will be retained
        
    Returns:
        processed mask, same shape and dtype as input
    """
    # process input dimension
    original_shape = mask.shape
    original_dtype = mask.dtype
    
    if len(original_shape) == 3:  # batch processing case (B, H, W)
        batch_size, height, width = original_shape
        result = np.zeros_like(mask)
        
        for b in range(batch_size):
            # process each batch
            binary_mask = (mask[b] > mask_threshold).astype(np.uint8)
            filtered_mask = filter_small_regions(binary_mask, area_threshold)
            # keep original non-binary values
            result[b] = mask[b] * filtered_mask
    
    else:  # single image (H, W)
        binary_mask = (mask > mask_threshold).astype(np.uint8)
        filtered_mask = filter_small_regions(binary_mask, area_threshold)
        # keep original non-binary values
        result = mask * filtered_mask

    return result.astype(original_dtype)

def filter_small_regions(binary_mask: np.ndarray, area_threshold: float) -> np.ndarray:
    """
    helper function, remove small regions from binary mask
    use OpenCV's connectedComponentsWithStats for efficient processing
    
    Args:
        binary_mask: binary mask (0 or 1), type is ndarray
        area_threshold: area threshold
        
    Returns:
        processed binary mask
    """
    # connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # vectorized method: create mapping from labels to retained status
    areas = stats[:, cv2.CC_STAT_AREA]
    keep_labels = areas >= area_threshold
    keep_labels[0] = False  # background is never retained
    
    filtered_mask = keep_labels[labels].astype(np.uint8)
    
    return filtered_mask

def draw_points_on_image(image, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """
    
    # Iterate through the list of points
    for point in points:
        cv2.circle(image, point, radius, (0, 255, 0), -1)

    return image


def draw_points_on_grayscale_image(image, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """
    
    # Iterate through the list of points
    for point in points:
        cv2.circle(image, point, radius, 255, -1)

    return image


# takes xy
def is_connected_bresenham(cost, start, end):
    c0, r0 = start
    c1, r1 = end
    rr, cc = line(r0, c0, r1, c1)
    kp_block_radius = 4
    cv2.circle(cost, start, kp_block_radius, 0, -1)
    cv2.circle(cost, end, kp_block_radius, 0, -1)
    
    # mean_cost = np.mean(cost[rr, cc])
    max_cost = np.max(cost[rr, cc])

    cv2.circle(cost, start, kp_block_radius, 255, -1)
    cv2.circle(cost, end, kp_block_radius, 255, -1)

    return max_cost < 255


# A* algorithm (A* pathfinding algorithm) to determine if two points in the image are connected.
# it uses A* algorithm to find the path from the start point to the end point, and check if the path exists
# and the length of the path is within the maximum length specified
def is_connected_astar(pathfinder, cost, start, end, max_path_len):
    # we can still modify the cost matrix after creating the pathfinder with it
    # seems pathfinder uses reference
    c0, r0 = start
    c1, r1 = end
    kp_block_radius = 6
    cv2.circle(cost, start, kp_block_radius, 1, -1)
    cv2.circle(cost, end, kp_block_radius, 1, -1)
    
    path = pathfinder.get_path(r0, c0, r1, c1)
    connected = (len(path) != 0) and (len(path) < max_path_len)

    cv2.circle(cost, start, kp_block_radius, 0, -1)
    cv2.circle(cost, end, kp_block_radius, 0, -1)

    return connected


# ensure the cost value of the road region is lower, while the cost value of the non-road region or the region near the keypoint is higher
def create_cost_field(sample_pts, road_mask):
    # road mask shall be uint8 normalized to 0-255
    cost_field = np.zeros(road_mask.shape, dtype=np.uint8)
    kp_block_radius = 4
    for point in sample_pts:
        cv2.circle(cost_field, point, kp_block_radius, 255, -1)
    cost_field = np.maximum(cost_field, 255 - road_mask)
    return cost_field

def create_cost_field_astar(sample_pts, road_mask, block_threshold=200):
    # road mask shall be uint8 normalized to 0-255
    # for tcod, 0 is blocked
    cost_field = np.zeros(road_mask.shape, dtype=np.uint8)
    kp_block_radius = 6
    for point in sample_pts:
        cv2.circle(cost_field, point, kp_block_radius, 255, -1)
    cost_field = np.maximum(cost_field, 255 - road_mask)
    cost_field[cost_field == 0] = 1
    cost_field[cost_field > block_threshold] = 0

    return cost_field


# if road_mask is 0-255, when the third nms is performed, setting all road_score to 0 is unfair,
# because some road_kp have higher scores, if road_mask is binary, then no modification is needed
# def extract_graph_points(keypoint_mask, road_mask, config):
#     kp_candidates, kp_scores = get_points_and_scores_from_mask(keypoint_mask, config.ITSC_THRESHOLD * 255)
#     kps_0 = nms_points(kp_candidates, kp_scores, config.ITSC_NMS_RADIUS)
#     kp_candidates, kp_scores = get_points_and_scores_from_mask(road_mask, config.ROAD_THRESHOLD * 255)
#     kps_1 = nms_points(kp_candidates, kp_scores, config.ROAD_NMS_RADIUS)
#     # prioritize intersection points
#     kp_candidates = np.concatenate([kps_0, kps_1], axis=0)
#     kp_scores = np.concatenate([np.ones((kps_0.shape[0])), np.zeros((kps_1.shape[0]))], axis=0)
#     kps = nms_points(kp_candidates, kp_scores, config.ROAD_NMS_RADIUS)
#     return kps

def extract_graph_points(keypoint_mask, road_mask, config):
    import time
    cost_time = {
        'total': 0,
        'remove_small_regions': 0,
        'get_kp_points_and_scores_from_mask': 0,
        'get_road_points_and_scores_from_mask': 0,
        'merge_kp_and_road_points': 0,
        'nms_concatenate_kp_and_road_points': 0
    }
    
    start_time = time.time()
    keypoint_mask = remove_small_regions(keypoint_mask, config.ITSC_AREA_THRESHOLD, config.ITSC_THRESHOLD)
    road_mask = remove_small_regions(road_mask, config.ROAD_AREA_THRESHOLD, config.ROAD_THRESHOLD)
    end_time = time.time()
    cost_time['remove_small_regions'] = end_time - start_time

    start_time = time.time()
    kp_candidates, kp_scores = get_points_and_scores_from_mask(keypoint_mask, config.ITSC_THRESHOLD)
    kp_scores = kp_scores + 0.1 # assign higher scores to preserve some keypoints
    end_time = time.time()
    print(f"kp_candidates: {kp_candidates.shape}, kp_scores: {kp_scores.shape}")
    cost_time['get_kp_points_and_scores_from_mask'] = end_time - start_time
    
    start_time = time.time()
    road_candidates, road_scores = get_points_and_scores_from_mask(road_mask, config.ROAD_THRESHOLD)
    end_time = time.time()
    print(f"road_candidates: {road_candidates.shape}, road_scores: {road_scores.shape}")
    cost_time['get_road_points_and_scores_from_mask'] = end_time - start_time
    
    start_time = time.time()
    all_candidates = np.concatenate([kp_candidates, road_candidates], axis=0)
    all_scores = np.concatenate([kp_scores, road_scores], axis=0)
    all_points = nms_points(all_candidates, all_scores, config.ROAD_NMS_RADIUS)
    end_time = time.time()
    print(f"after nms, all_points: {all_points.shape}")
    cost_time['nms_concatenate_kp_and_road_points'] = end_time - start_time
    cost_time['total'] = sum(cost_time.values())
    print(f"Cost time in function extract_graph_points: {cost_time}")
    return all_points, keypoint_mask, road_mask


def extract_graph_astar(keypoint_mask, road_mask, config):
    kps = extract_graph_points(keypoint_mask, road_mask, config)

    # cost_field = create_cost_field(kps, road_mask)
    cost_field = create_cost_field_astar(kps, road_mask)
    viz_cost_field = np.array(cost_field)
    viz_cost_field[viz_cost_field == 0] = 255
    # cv2.imwrite('astar_cost_dbg.png', viz_cost_field)
    pathfinder = tcod.path.AStar(cost_field)

    tree = KDTree(kps)
    graph = nx.Graph()
    checked = set()
    for p in kps:
        # TODO: add radius to config
        neighbor_indices = tree.query_radius(p[np.newaxis, :], r=config.NEIGHBOR_RADIUS)[0]
        for n_idx in neighbor_indices:
            n = kps[n_idx]
            start, end = (int(p[0]), int(p[1])), (int(n[0]), int(n[1]))
            if (start, end) in checked:
                continue
            # if is_connected_bresenham(cost_field, p, n):
            if is_connected_astar(pathfinder, cost_field, p, n, max_path_len=config.NEIGHBOR_RADIUS):
                graph.add_edge(start, end)
            checked.add((start, end))
    return graph

# takes xys    
def visualize_image_and_graph(img, graph):
    # Draw nodes as green squares
    for node in graph.nodes():
        x, y = node
        cv2.rectangle(
            img, (int(x) - 2, int(y) - 2), (int(x) + 2, int(y) + 2), (0, 255, 0), -1
        )
    # Draw edges as white lines
    for start_node, end_node in graph.edges():
        cv2.line(
            img,
            (int(start_node[0]), int(start_node[1])),
            (int(end_node[0]), int(end_node[1])),
            (255, 255, 255),
            1,
        )
    return img
    

if __name__ == '__main__':

    # cost = np.array(
    #     [[1, 0, 1],
    #      [0, 1, 0],
    #      [0, 0, 0]],
    #      dtype=np.int32
    # )
    # pathfinder = tcod.path.AStar(cost)
    # print(pathfinder.get_path(0, 2, 0, 0))
    # cost[1, 1] = 0
    # print(pathfinder.get_path(0, 2, 0, 0))
    # cost[1, 1] = 1
    # print(pathfinder.get_path(0, 2, 0, 0))

    rgb_pattern = './cityscale/20cities/region_{}_sat.png'
    keypoint_mask_pattern = './cityscale/processed/keypoint_mask_{}.png'
    road_mask_pattern = './cityscale/processed/road_mask_{}.png'

    index = 0
    rgb = read_rgb_img(rgb_pattern.format(index))
    road_mask = cv2.imread(road_mask_pattern.format(index), cv2.IMREAD_GRAYSCALE)
    keypoint_mask = cv2.imread(keypoint_mask_pattern.format(index), cv2.IMREAD_GRAYSCALE)

    graph = extract_graph_astar(keypoint_mask, road_mask)
    viz = visualize_image_and_graph(rgb, graph)
    cv2.imwrite('test_graph_astar_blk6_r40_m40_inms.png', viz)
