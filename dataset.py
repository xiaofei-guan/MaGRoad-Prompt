import json
import numpy as np
import pickle
import os
import glob
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import graph_utils
import scipy
import rtree


class GraphLabelGenerator():
    def __init__(self, config, full_graph, coord_transform):
        self.config = config
        # full_graph: sat2graph format
        # coord_transform: lambda, [N, 2] array -> [N, 2] array
        # convert to igraph for high performance
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform)
        # find crossover points, we'll avoid predicting these as keypoints
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin) # [(x, y), (),]
        # subdivide version
        self.subdivide_resolution = config.SUBDIVIDE_RESOLUTION
        self.full_graph_subdivide = graph_utils.subdivide_graph(self.full_graph_origin, self.subdivide_resolution)
        # np array, maybe faster
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])
        # pre-build spatial index
        # rtree for box queries
        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            # hack to insert single points
            self.graph_rtee.insert(i, (x, y, x, y))
        # kdtree for spherical query
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # pre-exclude points near crossover points
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # Find intersection points, these will always be kept in nms
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            # if self.full_graph_subdivide.degree(i) != 2:
            if self.full_graph_subdivide.degree(i) > 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num,), dtype=np.float32)
        if len(itsc_indices) > 0:
            self.nms_score_override[np.array(list(itsc_indices))] = 2.0  # itsc points will always be kept

        # Points near crossover and intersections are interesting.
        # they will be more frequently sampled
        interesting_indices = set()
        # interesting_radius = 32
        interesting_radius = config.INTERESTING_RADIUS
        # near itsc
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(np.array(p), interesting_radius)
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num,), 0.1, dtype=np.float32)
        if len(interesting_indices) > 0:
            self.sample_weights[list(interesting_indices)] = config.INTR_SAMPLE_WEIGHT  # itsc and neighbor 也是 interesting points

    def sample_patch(self, patch, rot_index=0):
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices # the point indices in the patch

        # Use NMS to downsample, params shall resemble inference time
        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            # print("==== Patch is empty ====")
            # this shall be rare, but if no points in side the patch, return null stuff
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = ([[0, 0]] * max_nbr_queries, [False] * max_nbr_queries, [False] * max_nbr_queries)
            return fake_points, [fake_sample] * sample_num

        patch_points = self.subdivide_points[patch_indices, :]

        # random scores to emulate different random configurations that all share a
        # similar spacing between sampled points
        # raise scores for intersction points so they are always kept
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override) # scores are between 0.9-1.0, intersection scores are 2.0
        nms_radius = self.config.ROAD_NMS_RADIUS

        # kept_indces are into the patch_points array
        nmsed_points, kept_indices = graph_utils.nms_points(patch_points, nms_scores, radius=nms_radius,
                                                            return_indices=True)
        # now this is into the subdivide graph
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]

        sample_num = self.config.TOPO_SAMPLE_NUM  # has to be greater than 1
        sample_weights = self.sample_weights[nmsed_indices]
        # indices into the nmsed points in the patch
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights))
        # indices into the subdivided graph
        sample_indices = nmsed_indices[sample_indices_in_nmsed]

        radius = self.config.NEIGHBOR_RADIUS
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES  # has to be greater than 1
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        # [n_sample, n_nbr]
        # k+1 because the nearest one is always self
        knn_d, knn_idx = nmsed_kdtree.query(sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius)
        # knn_d inf 填充, knn_idx num_points + 1 填充
        samples = []

        for i in range(sample_num): # 512
            source_node = sample_indices[i]
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]
            valid_nbr_indices = valid_nbr_indices[1:]  # the nearest one is self so remove
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]

            ### BFS to find immediate neighbors on graph
            reached_nodes = graph_utils.bfs_with_conditions(self.full_graph_subdivide, source_node, set(target_nodes),
                                                            radius // self.subdivide_resolution)
            shall_connect = [t in reached_nodes for t in target_nodes]
            ###

            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)

            # zero-pad
            for i in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)

            samples.append((pairs, shall_connect, valid))

        # Transform points
        # [N, 2]
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        # homo for rot
        # [N, 3]
        nmsed_points = np.concatenate([nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1)
        trans = np.array([
            [1, 0, -0.5 * self.config.PATCH_SIZE],
            [0, 1, -0.5 * self.config.PATCH_SIZE],
            [0, 0, 1],
        ], dtype=np.float32)
        # ccw 90 deg in img (x, y)
        rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]

        # Add noise
        noise_scale = self.config.NOISE_SCALE  # pixels
        nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)

        return nmsed_points, samples


class SatMapDataset(Dataset):
    """
    Dataset class for loading satellite/aerial imagery, masks, and graph data.
    Extracts keypoints from the graph, samples them, and applies augmentations.
    """
    def __init__(self, dataset_name="wildroad", is_train=True, max_kp_num=256, 
                 kp_sample_prob=0.8, negative_sample_ratio=1.0, negative_safe_radius=100.0, 
                 graph_config=None,
                 debug=False):
        """
        Args:
            dataset_name (str): Optional name for the dataset.
            is_train (bool): If True, loads train+val data; else loads test data.
            max_kp_num (int): The maximum number of keypoints to sample and return.
            kp_sample_prob (float): Probability of selecting an extracted keypoint during sampling.
            negative_sample_ratio (float): Ratio of negative samples to positive samples (e.g., 1.0 = equal number)
            negative_safe_radius (float): Minimum distance of negative samples from the road network
            graph_config: Configuration object for GraphLabelGenerator
            debug (bool): If True, only use a subset of the data for debugging.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.max_kp_num = max_kp_num
        self.kp_sample_prob = kp_sample_prob
        self.negative_sample_ratio = negative_sample_ratio
        self.negative_safe_radius = negative_safe_radius
        self.graph_config = graph_config

        base_dir = './wildroad'

        # Collect tile IDs from each split folder
        def get_tile_ids_from_folder(folder_path):
            """Extract tile IDs from data_{id}.png files in folder"""
            tile_ids = []
            if not os.path.exists(folder_path):
                print(f'Warning: folder {folder_path} does not exist')
                return tile_ids
            for filename in os.listdir(folder_path):
                if filename.startswith('data_') and filename.endswith('.png'):
                    try:
                        tile_id = filename.replace('data_', '').replace('.png', '')
                        tile_ids.append(tile_id)
                    except:
                        continue
            return sorted(tile_ids)
        
        # Get tile IDs from each split
        train_ids = get_tile_ids_from_folder(os.path.join(base_dir, 'wild_road', 'train_patches', 'train_AB'))
        val_ids = get_tile_ids_from_folder(os.path.join(base_dir, 'wild_road', 'val_patches', 'val_AB'))
        test_ids = get_tile_ids_from_folder(os.path.join(base_dir, 'wild_road', 'test_patches', 'test_AB'))
        
        print(f'Found {len(train_ids)} train tiles, {len(val_ids)} val tiles, {len(test_ids)} test tiles')

        # Prepare tile information list
        tile_info_list = []
        
        # For training: use train + val; For testing: use test
        if self.is_train:
            # Add train tiles
            for tid in train_ids:
                tile_info_list.append({
                    'id': tid,
                    'rgb_pattern': os.path.join(base_dir, 'wild_road', 'train_patches', 'train_AB', 'data_{}.png'),
                    'keypoint_pattern': os.path.join(base_dir, 'wild_road_mask', 'train_patches', 'train_AB', 'keypoint_mask_{}.png'),
                    'road_pattern': os.path.join(base_dir, 'wild_road_mask', 'train_patches', 'train_AB', 'road_mask_{}.png'),
                    'graph_pattern': os.path.join(base_dir, 'wild_road', 'train_patches', 'train_AB', 'gt_graph_{}.pickle')
                })
            # Add val tiles
            for tid in val_ids:
                tile_info_list.append({
                    'id': tid,
                    'rgb_pattern': os.path.join(base_dir, 'wild_road', 'val_patches', 'val_AB', 'data_{}.png'),
                    'keypoint_pattern': os.path.join(base_dir, 'wild_road_mask', 'val_patches', 'val_AB', 'keypoint_mask_{}.png'),
                    'road_pattern': os.path.join(base_dir, 'wild_road_mask', 'val_patches', 'val_AB', 'road_mask_{}.png'),
                    'graph_pattern': os.path.join(base_dir, 'wild_road', 'val_patches', 'val_AB', 'gt_graph_{}.pickle')
                })
        else:
            # Add test tiles
            for tid in test_ids:
                tile_info_list.append({
                    'id': tid,
                    'rgb_pattern': os.path.join(base_dir, 'wild_road', 'test_patches', 'test_AB', 'data_{}.png'),
                    'keypoint_pattern': os.path.join(base_dir, 'wild_road_mask', 'test_patches', 'test_AB', 'keypoint_mask_{}.png'),
                    'road_pattern': os.path.join(base_dir, 'wild_road_mask', 'test_patches', 'test_AB', 'road_mask_{}.png'),
                    'graph_pattern': os.path.join(base_dir, 'wild_road', 'test_patches', 'test_AB', 'gt_graph_{}.pickle')
                })

        # Optional debug shrink
        if debug:
            tile_info_list = tile_info_list[:min(len(tile_info_list), 200)]

        # Initialize storage
        self.rgb_files_map = {}
        self.kp_mask_files = {}
        self.rd_mask_files = {}
        self.graph_files = {}
        self.extracted_keypoints = {}
        self.adjacency_lists = {}
        self.negative_keypoints = {}
        self.graph_label_generators = {}
        
        processed_data = 'train+val' if self.is_train else 'test'
        print(f'-------------------loading {processed_data} data ({len(tile_info_list)} tiles)------------------')

        # Load each tile
        valid_indices = []
        coord_transform = lambda v: v[:, ::-1]  # Convert (row, col) to (x, y)
        
        for i, tile_info in enumerate(tile_info_list):
            tile_id = tile_info['id']
            print(f'Loading tile {tile_id}')

            rgb_path = tile_info['rgb_pattern'].format(tile_id)
            kp_mask_path = tile_info['keypoint_pattern'].format(tile_id)
            rd_mask_path = tile_info['road_pattern'].format(tile_id)
            graph_path = tile_info['graph_pattern'].format(tile_id)

            # Check if all files exist
            if not (os.path.exists(rgb_path) and os.path.exists(kp_mask_path) and 
                    os.path.exists(rd_mask_path) and os.path.exists(graph_path)):
                print(f'===== Skipped tile {tile_id}: missing files =====')
                continue

            # Load graph data
            with open(graph_path, 'rb') as f:
                graph_adj = pickle.load(f)

            # Create GraphLabelGenerator
            graph_label_generator = GraphLabelGenerator(self.graph_config, graph_adj, coord_transform)
            self.graph_label_generators[i] = graph_label_generator
            self.adjacency_lists[i] = graph_adj

            # Extract keypoints
            with Image.open(rgb_path) as img:
                width, height = img.size
            self.extracted_keypoints[i] = self._extract_keypoints(graph_adj, image_size=width)
            
            # Generate negative samples
            estimated_neg_samples = int(max(1, len(self.extracted_keypoints[i])) * self.negative_sample_ratio)
            self.negative_keypoints[i] = self._generate_negative_points(
                graph_adj, width, height, estimated_neg_samples
            )
            
            # Store file paths
            valid_indices.append(i)
            self.rgb_files_map[i] = rgb_path
            self.kp_mask_files[i] = kp_mask_path
            self.rd_mask_files[i] = rd_mask_path
            self.graph_files[i] = graph_path

        if not valid_indices:
            raise RuntimeError("No valid samples found with all corresponding files.")

        self.valid_indices = valid_indices
        print(f"Successfully loaded {len(self.valid_indices)} valid samples.")

        # Define augmentations
        if self.is_train:
            self.rgb_augment = T.Compose([
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ])

    def __len__(self):
        """Return the number of valid samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Loads data for the given index, samples keypoints, applies augmentations,
        and returns a dictionary of tensors.
        """
        # Map dataset index to the actual index in the original file list
        actual_idx = self.valid_indices[idx]

        # Load image and mask data as PIL images
        img, kp_mask, rd_mask = self._load_data(actual_idx)
        
        # Get pre-extracted keypoints for this sample
        keypoints_xy = self.extracted_keypoints[actual_idx]

        # Sample keypoints (returns lists, not tensors)
        sampled_kp, kp_label = self._sample_keypoints(keypoints_xy)
        
        # get pre-generated negative samples
        negative_points = self.negative_keypoints[actual_idx]
        
        # calculate the actual number of negative samples needed
        num_positives = sum(kp_label)  # actual number of positive samples sampled
        num_negatives = max(1, int(num_positives * self.negative_sample_ratio))
        
        # if the number of pre-generated negative samples exceeds the number needed, randomly select a subset
        if len(negative_points) > num_negatives:
            negative_points = random.sample(negative_points, num_negatives)
        
        # combine positive and negative samples
        combined_points = sampled_kp.copy()
        combined_labels = kp_label.copy()
        
        # add negative samples and labels to the sampled points
        for neg_point in negative_points:
            combined_points.append(neg_point)
            combined_labels.append(0)  # negative sample label is 0

        # sample graph pairs
        rot_index = np.random.randint(0, 4)
        patch = ((0, 0), (img.shape[1], img.shape[0]))

        graph_points, topo_samples = self.graph_label_generators[actual_idx].sample_patch(patch, rot_index)
        pairs, connected, valid = zip(*topo_samples)

        # rgb: [H, W, 3] 0-255
        # masks: [H, W] 0-1

        # Apply augmentations (if training) - all inputs/outputs as PIL images or lists
        if self.is_train:
            img, kp_mask, rd_mask, combined_points = self._apply_augmentations(
                img, kp_mask, rd_mask, combined_points, angle=rot_index
            )

        return {
            'rgb': torch.tensor(img, dtype=torch.float32), # [H, W, C] 0-255
            'keypoint_mask': torch.tensor(kp_mask, dtype=torch.float32) / 255.0, # [H, W] 0-1
            'road_mask': torch.tensor(rd_mask, dtype=torch.float32) / 255.0, # [H, W] 0-1
            'sampled_kp': torch.tensor(combined_points, dtype=torch.float32), # [N, 2]
            'kp_label': torch.tensor(combined_labels, dtype=torch.long), # [N]
            # sampled graph pairs, predict if the graph pairs are connected
            'graph_points': torch.tensor(graph_points, dtype=torch.float32),
            'pairs': torch.tensor(pairs, dtype=torch.int32),
            'connected': torch.tensor(connected, dtype=torch.bool),
            'valid': torch.tensor(valid, dtype=torch.bool),
        }

    def _load_data(self, idx):
        """Loads Image and Masks for a given internal index."""
        rgb_path = self.rgb_files_map[idx]
        kp_mask_path = self.kp_mask_files[idx]
        rd_mask_path = self.rd_mask_files[idx]

        # Load image
        img = Image.open(rgb_path).convert('RGB')
        kp_mask = Image.open(kp_mask_path).convert('L')
        rd_mask = Image.open(rd_mask_path).convert('L')

        img = np.array(img)
        kp_mask = np.array(kp_mask)
        rd_mask = np.array(rd_mask)

        return img, kp_mask, rd_mask

    def _extract_keypoints(self, graph_adj, image_size):
        """
        Extracts keypoints (nodes with degree != 2) from the graph adjacency list.
        Converts (row, col) coordinates to (x, y) format.

        Args:
            graph_adj (dict): Adjacency list { (r,c): [(nr, nc), ...], ... }
            image_size (tuple): Image size (width, height)
        Returns:
            list: A list of keypoints, each as [x, y].
        """
        kp_float_rc = [node for node, neighbors in graph_adj.items() if len(neighbors) != 2]
        # kp_float_rc = [node for node, neighbors in graph_adj.items() if len(neighbors) > 2]
        if len(kp_float_rc) == 0:
            return []
        
        keypoints_xy = [list(map(int, node))[::-1] for node in kp_float_rc]
        
        # adjust points close to the edge
        keypoints_xy = self._adjust_edge_points(keypoints_xy, image_size, 10)
        
        return keypoints_xy
    
    def _adjust_edge_points(self, points, image_size, min_distance=10):
        """
        adjust points close to the edge of the image, ensure they are at least a minimum distance from the edge
        
        Args:
            points: keypoints list, each point is [x, y] coordinates
            image_size: image size (assumed to be square)
            min_distance: minimum distance from the edge
            
        Returns:
            adjusted points list
        """
        adjusted_points = []
        
        for point in points:
            x, y = point
            
            # adjust x coordinate, ensure the distance from the left and right edges is at least min_distance
            if x < min_distance:
                x = min_distance
            elif x > image_size - min_distance:
                x = image_size - min_distance
                
            # adjust y coordinate, ensure the distance from the top and bottom edges is at least min_distance
            if y < min_distance:
                y = min_distance
            elif y > image_size - min_distance:
                y = image_size - min_distance
                
            adjusted_points.append([int(x), int(y)])
            
        return adjusted_points

    def _sample_keypoints(self, keypoints_xy):
        """
        Samples keypoints based on probability and ensures at least one keypoint is sampled.

        Args:
            keypoints_xy (list): List of keypoints [[x, y], ...].

        Returns:
            tuple: (sampled_kp_list, kp_label_list)
                   sampled_kp_list: List of [x, y] coordinates
                   kp_label_list: List of labels (1 for valid, 0 for padded)
        """
        num_extracted_kp = len(keypoints_xy)
        if num_extracted_kp == 0:
            return [], []
        sampled_kp = []
        kp_indices = list(range(num_extracted_kp))

        # Sample based on probability
        sampled_indices = [i for i in kp_indices if random.random() < self.kp_sample_prob]

        # ensure at least one positive sample is sampled
        if not sampled_indices and num_extracted_kp > 0:
            sampled_indices = [random.choice(kp_indices)]

        # If more points sampled than needed, randomly select subset
        if len(sampled_indices) > self.max_kp_num:
            sampled_indices = random.sample(sampled_indices, self.max_kp_num)

        # Get the coordinates of sampled keypoints
        for idx in sampled_indices:
            sampled_kp.append(keypoints_xy[idx])

        num_sampled = len(sampled_kp)
        kp_label = [1] * num_sampled

        return sampled_kp, kp_label

    def _generate_negative_points(self, adjacency_list, width, height, num_samples):
        """
        generate negative samples far from the road network

        Args:
            adjacency_list: road network adjacency list (row, col):(Y, X)
            width, height: image size
            num_samples: number of negative samples needed

        Returns:
            negative samples list
        """
        negative_points = []

        # create a list of edges and their line representations, for checking if points are close to the road network
        edges_set = set()
        for node, node_neighbors in adjacency_list.items():
            for neighbor in node_neighbors:
                # create a unique representation for undirected edges
                edge = (min(node, neighbor), max(node, neighbor))
                edges_set.add(edge)
                
        edges = list(edges_set)

        # calculate the center point of the image
        center_x, center_y = width // 2, height // 2

        # calculate the maximum distance from the center to the corners
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

        # set the weights for the center and edges
        center_weight = 1.0
        edge_weight = 0.6  # edge weight should not be too low, ensure the difference is moderate

        # try to generate the specified number of negative samples
        attempts = 0
        max_attempts = num_samples * 10  # prevent infinite loop

        while len(negative_points) < num_samples and attempts < max_attempts:
            attempts += 1

            # improved random point generation strategy, make the probability of the center region higher
            rand_x = random.randint(0, width - 1)
            rand_y = random.randint(0, height - 1)

            # calculate the distance from the point to the center
            distance_to_center = np.sqrt((rand_x - center_x) ** 2 + (rand_y - center_y) ** 2)

            # normalize the distance and calculate the sampling weight
            normalized_distance = distance_to_center / max_distance
            sample_weight = center_weight - (center_weight - edge_weight) * (normalized_distance ** 2)

            # accept-reject sampling
            if random.random() > sample_weight:
                continue  # reject this sample, try the next one

            point = [rand_y, rand_x]

            # check if the point is far from the road network
            is_far_from_network = True

            for node_coors in adjacency_list.keys():
                distance = np.sqrt((point[0] - node_coors[0]) ** 2 + (point[1] - node_coors[1]) ** 2)

                if distance < self.negative_safe_radius:
                    is_far_from_network = False
                    break

            # if the point is close to a keypoint, skip
            if not is_far_from_network:
                continue

            # check the distance from the point to all edges
            for edge in edges:
                p1, p2 = edge

                # calculate the distance from the point to the line segment
                line_length = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

                if line_length == 0:
                    distance = np.sqrt((point[0] - p1[0]) ** 2 + (point[1] - p1[1]) ** 2)
                else:
                    t = max(0, min(1, ((point[0] - p1[0]) * (p2[0] - p1[0]) +
                                       (point[1] - p1[1]) * (p2[1] - p1[1])) / (line_length ** 2)))

                    proj_x = p1[0] + t * (p2[0] - p1[0])
                    proj_y = p1[1] + t * (p2[1] - p1[1])

                    distance = np.sqrt((point[0] - proj_x) ** 2 + (point[1] - proj_y) ** 2)

                if distance < self.negative_safe_radius:
                    is_far_from_network = False
                    break

            # if the point is far from the road network, add it as a negative sample
            if is_far_from_network:
                negative_points.append(point[::-1]) # convert the point coordinates from (y, x) to (x, y)

        return negative_points

    def _apply_augmentations(self, img, kp_mask, rd_mask, keypoints, angle=0):
        """
        Applies geometric and photometric augmentations.
        All operations are done on PIL images and lists, not tensors.

        Args:
            img (PIL.Image): Input image.
            kp_mask (PIL.Image): Keypoint mask.
            rd_mask (PIL.Image): Road mask.
            keypoints (list): List of [x, y] coordinates.
            angle (int): Rotation angle in degrees.

        Returns:
            tuple: Augmented (img, kp_mask, rd_mask, keypoints_list)
        """
        # 1. Geometric Augmentation: Random Rotation
        if angle != 0:
            img = np.rot90(img, angle).copy()
            kp_mask = np.rot90(kp_mask, angle).copy()
            rd_mask = np.rot90(rd_mask, angle).copy()
            # Rotate keypoints
            width, height = img.shape[:2]
            keypoints = self._rotate_keypoints_list(keypoints, angle * 90, width/2, height/2)

        # 2. Apply rgb augmentation to PIL image
        if self.rgb_augment is not None:
            # Convert to tensor temporarily for color jitter
            img_tensor = TF.to_tensor(img)
            img_tensor = self.rgb_augment(img_tensor)
            # Convert back to ndarray
            img = img_tensor.permute(1, 2, 0).numpy() * 255.0

        return img, kp_mask, rd_mask, keypoints

    def _rotate_keypoints_list(self, keypoints_list, angle, center_x, center_y):
        """
        Rotates a list of keypoints around a center point.
        
        Args:
            keypoints_list (list): List of [x, y] coordinates.
            angle (float): Rotation angle in degrees (counter-clockwise).
            center_x, center_y (float): Rotation center coordinates.
            
        Returns:
            list: Rotated keypoints.
        """
        angle_rad = math.radians(-angle)  # Negative for standard rotation
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated_keypoints = []
        for kp in keypoints_list:
            # Skip rotation for padding points [0,0]
            if kp[0] == 0 and kp[1] == 0:
                rotated_keypoints.append([0, 0])
                continue
                
            # Translate to origin
            x, y = kp[0] - center_x, kp[1] - center_y
            
            # Rotate
            x_new = x * cos_a - y * sin_a
            y_new = x * sin_a + y * cos_a
            
            # Translate back
            rotated_keypoints.append([x_new + center_x, y_new + center_y])
            
        return rotated_keypoints
    
def custom_collate_fn(batch):
    """
    custom collate_fn function to handle different lengths of sampled_kp and graph_points
    
    Args:
        batch: a batch of samples from the Dataset
        
    Returns:
        processed batch data
    """
    # separate the fields in the batch
    rgb_batch = [item['rgb'] for item in batch]
    keypoint_mask_batch = [item['keypoint_mask'] for item in batch]
    road_mask_batch = [item['road_mask'] for item in batch]
    sampled_kp_batch = [item['sampled_kp'] for item in batch]
    kp_label_batch = [item['kp_label'] for item in batch]
    graph_points_batch = [item['graph_points'] for item in batch] 
    pairs_batch = [item['pairs'] for item in batch]              
    connected_batch = [item['connected'] for item in batch]     
    valid_batch = [item['valid'] for item in batch]             
    
    # find the maximum number of sampled points in the batch (sampled_kp)
    max_sampled_points = max(kp.shape[0] for kp in sampled_kp_batch)
    
    # find the maximum number of graph nodes in the batch (graph_points)
    max_graph_points = max(gp.shape[0] for gp in graph_points_batch)
    
    # create padded tensors
    batch_size = len(batch)
    padded_sampled_kp = torch.zeros(batch_size, max_sampled_points, 2, dtype=torch.float32)
    padded_kp_labels = torch.full((batch_size, max_sampled_points), -1, dtype=torch.long)  # use -1 to pad the labels
    padded_graph_points = torch.zeros(batch_size, max_graph_points, 2, dtype=torch.float32) 
    
    # pad the points and labels (sampled_kp)
    for i, item in enumerate(batch):
        points = item['sampled_kp']
        labels = item['kp_label']
        num_points = points.shape[0]
        padded_sampled_kp[i, :num_points, :] = points
        padded_kp_labels[i, :num_points] = labels
        
    # pad the graph nodes (graph_points) 
    for i, item in enumerate(batch):
        points = item['graph_points']
        num_points = points.shape[0]
        padded_graph_points[i, :num_points, :] = points
    
    rgb_stacked = torch.stack(rgb_batch)
    keypoint_mask_stacked = torch.stack(keypoint_mask_batch)
    road_mask_stacked = torch.stack(road_mask_batch)
    
    # pairs: [B, S, M, 2], connected: [B, S, M], valid: [B, S, M]
    # S = sample_num, M = max_nbr_queries
    pairs_stacked = torch.stack(pairs_batch)
    connected_stacked = torch.stack(connected_batch)
    valid_stacked = torch.stack(valid_batch)
    
    return {
        'rgb': rgb_stacked,                     # [B, H, W, C]
        'keypoint_mask': keypoint_mask_stacked, # [B, H, W]
        'road_mask': road_mask_stacked,         # [B, H, W]
        'sampled_kp': padded_sampled_kp,        # [B, N_kp_pad, 2]
        'kp_label': padded_kp_labels,           # [B, N_kp_pad]
        'graph_points': padded_graph_points,    # [B, N_gp_pad, 2]
        'pairs': pairs_stacked,                 # [B, S, M, 2]
        'connected': connected_stacked,         # [B, S, M]
        'valid': valid_stacked,                 # [B, S, M]
    }

def visualize_sample(sample, fig_title="Sample Visualization", save_path=None):
    """
    visualize a single sample: RGB+keypoints, keypoint mask and road mask
    
    Args:
        sample: a sample from the dataset
        fig_title: the title of the figure
        save_path: the path to save the image
    """
    # get the data
    rgb = sample['rgb']  # [H, W, C] 0-255
    keypoint_mask = sample['keypoint_mask']  # [H, W] 0-1
    road_mask = sample['road_mask']  # [H, W] 0-1
    sampled_kp = sample['sampled_kp']  # [N, 2]
    kp_label = sample['kp_label']  # [N]
    
    # convert to numpy array for plotting
    rgb_np = rgb.numpy() / 255.0
    keypoint_mask_np = keypoint_mask.numpy()
    road_mask_np = road_mask.numpy()

    # create the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(fig_title)
    
    # plot the RGB image and keypoints
    axes[0].imshow(rgb_np) # 0-1 for float, 0-255 for uint8
    axes[0].set_title("RGB Image with Keypoints")
    
    # separate positive and negative samples
    positive_mask = kp_label == 1
    negative_mask = kp_label == 0
    
    # plot the positive samples (green points)
    pos_kps = sampled_kp[positive_mask].numpy()
    if len(pos_kps) > 0:
        axes[0].scatter(pos_kps[:, 0], pos_kps[:, 1], c='green', s=50, marker='o', label='Positive')
    
    # plot the negative samples (red X)
    neg_kps = sampled_kp[negative_mask].numpy()
    if len(neg_kps) > 0:
        axes[0].scatter(neg_kps[:, 0], neg_kps[:, 1], c='red', s=50, marker='x', label='Negative')
    
    # add the legend
    if len(pos_kps) > 0 or len(neg_kps) > 0:
        axes[0].legend()
    
    # plot the keypoint mask
    axes[1].imshow(keypoint_mask_np, cmap='gray')
    axes[1].set_title("Keypoint Mask")
    
    # plot the road mask
    axes[2].imshow(road_mask_np, cmap='gray')
    axes[2].set_title("Road Mask")
    
    # save the image (if the path is provided)
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            print(f"create directory: {os.path.dirname(save_path)}")
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"image saved to: {save_path}")
        plt.close()
    else:
        plt.show()

def test_dataloader(dataset, batch_size=4):
    """
    test if the DataLoader correctly handles the dataset
    
    Args:
        dataset: the dataset to test
        batch_size: the batch size
    """
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    # these two lines of code will cause the graph_tree to fail to initialize normally
    # create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # get a batch of data
    batch = next(iter(dataloader))
    
    # check the shape of the batch data
    print("\n===== DataLoader batch data validation =====")
    print(f"batch size: {batch_size}")
    print(f"rgb shape: {batch['rgb'].shape}")
    print(f"keypoint_mask shape: {batch['keypoint_mask'].shape}")
    print(f"road_mask shape: {batch['road_mask'].shape}")
    print(f"sampled_kp shape: {batch['sampled_kp'].shape}")
    print(f"kp_label shape: {batch['kp_label'].shape}")
    print(f"graph_points shape: {batch['graph_points'].shape}")
    print(f"pairs shape: {batch['pairs'].shape}")
    print(f"connected shape: {batch['connected'].shape}")
    print(f"valid shape: {batch['valid'].shape}")
    
    # check the value range
    print(f"rgb value range: [{batch['rgb'].min():.2f}, {batch['rgb'].max():.2f}]")
    print(f"keypoint_mask value range: [{batch['keypoint_mask'].min():.2f}, {batch['keypoint_mask'].max():.2f}]")
    print(f"road_mask value range: [{batch['road_mask'].min():.2f}, {batch['road_mask'].max():.2f}]")
    print(f"kp_label value distribution: {torch.bincount(batch['kp_label'][batch['kp_label'] != -1])}")  # count the label distribution (excluding the padding value)
    print(f"sampled_kp value range: [{batch['sampled_kp'].min():.2f}, {batch['sampled_kp'].max():.2f}]")
    print(f"graph_points value range: [{batch['graph_points'].min():.2f}, {batch['graph_points'].max():.2f}]")
    print(f"pairs value range: [{batch['pairs'].min():.2f}, {batch['pairs'].max():.2f}]")
    print(f"connected value range: [{batch['connected'].min():.2f}, {batch['connected'].max():.2f}]")
    print(f"valid value range: [{batch['valid'].min():.2f}, {batch['valid'].max():.2f}]")
    
    # visualize the first sample in the batch
    batch_sample = {
        'rgb': batch['rgb'][0],
        'keypoint_mask': batch['keypoint_mask'][0],
        'road_mask': batch['road_mask'][0],
        'sampled_kp': batch['sampled_kp'][0],
        'kp_label': batch['kp_label'][0]
    }
    
    # visualize_sample(batch_sample, fig_title="Batch Sample Visualization", 
    #                 save_path="./data_vis/batch_sample_visualization.png")
    
    print(f"batch_sampled_kp: {batch['sampled_kp']}")
    print(f"batch_kp_label: {batch['kp_label']}")

# main test function
def run_tests(data_dir, is_train=True, max_kp_num=64):
    """
    run the complete test process
    
    Args:
        data_dir: the dataset directory
        is_train: whether to use the training mode
        max_kp_num: the number of sampled keypoints
    """
    print(f"===== test SatMapDataset =====")
    print(f"dataset directory: {data_dir}")
    print(f"training mode: {is_train}")
    print(f"number of sampled keypoints: {max_kp_num}")

    from addict import Dict
    
    graph_config = Dict()
    
    graph_config.PATCH_SIZE = 1024
    graph_config.TOPO_SAMPLE_NUM = 128
    graph_config.ITSC_NMS_RADIUS = 50
    graph_config.ROAD_NMS_RADIUS = 50 
    graph_config.NEIGHBOR_RADIUS = 300 
    graph_config.MAX_NEIGHBOR_QUERIES = 8 
    graph_config.SUBDIVIDE_RESOLUTION = 25 
    graph_config.INTERESTING_RADIUS = 300
    graph_config.INTR_SAMPLE_WEIGHT = 0.9
    
    try:
        # create the dataset
        dataset = SatMapDataset(
            data_dir=data_dir,
            dataset_name="wildroad",
            is_train=is_train,
            max_kp_num=max_kp_num,
            kp_sample_prob=0.8,
            negative_sample_ratio=1.0,  # the ratio of negative samples to positive samples (1:1)
            negative_safe_radius=100.0,  # the safe distance of negative samples from the road network
            graph_config=graph_config
        )
        
        print(f"dataset size: {len(dataset)}")
        
        # get and visualize a single sample
        # sample_idx = np.random.randint(0, len(dataset))
        sample_indices = list(range(len(dataset)))
        for sample_idx in sample_indices:
            print(f"\nget sample {sample_idx}")
            sample = dataset[sample_idx]
            
            # check the data structure
            print(f"sample data structure:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: type={type(value)}")
            # print(f"sample_kp: {sample['sampled_kp']}")
            # print(f"kp_label: {sample['kp_label']}")
            # visualize the sample
            # visualize_sample(sample, f"sample_{sample_idx} - {'train' if is_train else 'test'}",
            #                 save_path=f"./data_vis2/sample_{sample_idx}_visualization.png")
        
        # test the DataLoader
        test_dataloader(dataset)
        
        print("\n===== test completed =====")
        
    except Exception as e:
        print(f"error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # specify the dataset directory - please modify according to the actual situation
    DATA_DIR = "~/sam_road-main/mydata/map_patches"
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # run the test
    run_tests(DATA_DIR, is_train=False, max_kp_num=10)