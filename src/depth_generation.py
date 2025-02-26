from abc import abstractmethod
from PIL import Image
import numpy as np

import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from transformers import pipeline
class DepthGeneartor:
    def __init__(self, device=None) -> None:
        if device is not None:
            self.device = device
        else:
            self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           

    @abstractmethod
    def run(self, image_path, save_depth=False):
        pass


class DepthGeneratorDepthAnything(DepthGeneartor):
    def __init__(self, device=None):
        super().__init__(device)
        
        checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
        self.pipe = pipeline("depth-estimation", model=checkpoint, device=self.device)
    
    def run(self, image_path, save_depth=False) -> dict:
        image = Image.open(image_path)
        depth_image = self.pipe(image)["depth"]
        if save_depth:
            depth_image.save(image_path[:-4] + "_depth.jpg")
        return {"depth": depth_image,
                "field_of_view": 100,
                "focal_length": 1,
                "default_intrinsics": True
                }

class DepthGeneratorDepthPro(DepthGeneartor):
    def __init__(self, device=None):
        super().__init__(device)
        self.image_pre_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(self.device)

    def run(self, image_path, save_depth=False) -> dict:
        image = Image.open(image_path)
        with torch.no_grad():
            inputs = self.image_pre_processor(image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            post_processed_output = self.image_pre_processor.post_process_depth_estimation(
                                    outputs, target_sizes=[(image.height, image.width)],
                                    )

        field_of_view = post_processed_output[0]["field_of_view"].detach().cpu().numpy()
        focal_length = post_processed_output[0]["focal_length"].detach().cpu().numpy()
        depth = post_processed_output[0]["predicted_depth"]
        depth = (depth - depth.min()) / depth.max()
        depth = depth * 255.
        depth = depth.detach().cpu().numpy()
        depth = Image.fromarray(depth.astype("uint8"))
        if save_depth:
            #TO DO
            #Add custom path to save the depth image
            depth.save(image_path[:-4] + "_depth.jpg")

        return {"depth": np.array(depth),
                "field_of_view": field_of_view,
                "focal_length": focal_length,
                "default_intrinsics": False

                }