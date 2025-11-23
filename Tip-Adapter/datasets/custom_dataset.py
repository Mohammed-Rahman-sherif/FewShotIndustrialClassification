import os
import math
import random
from collections import defaultdict
from .utils import Datum, DatasetBase

# This template puts your semantic name into a sentence for CLIP
template = ['a photo of a {}.']


class CustomDataset(DatasetBase):
    def __init__(self, root, num_shots):
        self.dataset_dir = root  # Path to your 'classification samples/test' folder
        self.template = template

        # 1. Read the data from folders and map IDs to semantic names
        items = self.read_data_from_folder(self.dataset_dir)

        # 2. Split into Train (Few-shot), Val, and Test
        train, val, test = self.split_data(items, num_shots)

        # 3. Initialize the parent class
        super().__init__(train_x=train, val=val, test=test)

    def read_data_from_folder(self, root_path):
        items = []
        # Sort to ensure consistent label ordering
        classes = sorted([d for d in os.listdir(root_path)
                         if os.path.isdir(os.path.join(root_path, d))])

        # ---------------------------------------------------------
        # FULL MAPPING DICTIONARY (Folders 1-15)
        # Maps your Technical Folder IDs -> Semantic Visual Descriptions
        # ---------------------------------------------------------
        folder_to_semantic_name = {
            # Batch 1 (Folders 1-10)
            "new": "light grey vertical industrial gear motor",
            "JB00039888": "red and silver vertical planetary gear motor",
            "JB00040002": "large red right-angle industrial gearbox",
            "JB00047922": "teal vertical electric gear motor",
            "JB00024791": "light blue-grey vertical gear motor",
            "JB00024827": "red vertical planetary gear motor with black ring",
            "JB00028415": "red right-angle bevel gearbox",
            "JB00018206": "grey vertical industrial electric motor",
            "JB00020032": "red vertical planetary gearbox housing without motor",
            "JB00024563": "red vertical gearbox with tall silver electric motor",

            # Batch 2 (Folders 11-15)
            "2T7094VA79A06": "red vertical planetary gearbox with silver aluminum top housing",
            "JB00015001": "red right-angle industrial elbow gearbox with open flange",
            "JB00015906": "short cylindrical planetary gear module with black ring and flat top",
            "2T704T1004A03": "small squat red planetary gear hub with white cap",
            "2T7094VA79A05": "red planetary gearbox with silver right-angle worm drive on top"
        }

        for label_idx, classname in enumerate(classes):
            class_dir = os.path.join(root_path, classname)

            # Find valid images
            image_names = sorted([f for f in os.listdir(
                class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

            # Get the semantic name from dict, or fallback to folder name if missing
            semantic_name = folder_to_semantic_name.get(classname, classname)

            # Clean up formatting (optional safety step)
            semantic_name = semantic_name.replace("_", " ")

            for im_name in image_names:
                item = Datum(
                    impath=os.path.join(class_dir, im_name),
                    label=label_idx,
                    classname=semantic_name  # Tip-Adapter uses this for the text prompt
                )
                items.append(item)

        return items

    def split_data(self, items, num_shots, p_val=0.2):
        """
        Automatically splits your data into:
        - Train: 'num_shots' images per class (for the cache model)
        - Val: 20% of remaining images
        - Test: 80% of remaining images
        """
        tracker = defaultdict(list)
        for item in items:
            tracker[item.label].append(item)

        train, val, test = [], [], []

        for label, class_items in tracker.items():
            random.shuffle(class_items)

            # 1. Select Few-shot Training Set
            if len(class_items) >= num_shots:
                train_items = class_items[:num_shots]
                remaining = class_items[num_shots:]
            else:
                # If a class has fewer images than requested shots, use all of them for train
                # (Note: This leaves none for val/test for this specific class)
                train_items = class_items
                remaining = []

            # 2. Split the rest into Val and Test
            n_remaining = len(remaining)
            if n_remaining > 0:
                # Ensure at least 1 val image if data exists
                n_val = max(1, int(n_remaining * p_val))
                val_items = remaining[:n_val]
                test_items = remaining[n_val:]
            else:
                val_items = []
                test_items = []

            train.extend(train_items)
            val.extend(val_items)
            test.extend(test_items)

        return train, val, test
