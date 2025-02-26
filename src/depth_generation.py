from PIL import Image
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

class DepthGeneartor:
    def __init__(self, device=None) -> None:
        if device is not None:
            self.device = device
        else:
            self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_pre_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

    def run(self, image_path, save_depth=False) -> dict:
        image = Image.open(image_path)
        with torch.no_grad():
            inputs = self.image_pre_processor(image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            post_processed_output = self.image_pre_processor.post_process_depth_estimation(
                                    outputs, target_sizes=[(image.height, image.width)],
                                    )

        field_of_view = post_processed_output[0]["field_of_view"]
        focal_length = post_processed_output[0]["focal_length"]
        depth = post_processed_output[0]["predicted_depth"]
        depth = (depth - depth.min()) / depth.max()
        depth = depth * 255.
        depth = depth.detach().cpu().numpy()
        depth = Image.fromarray(depth.astype("uint8"))
        if save_depth:
            #TO DO
            #Add custom path to save the depth image
            depth.save(image_path[:-4] + "_depth.jpg")

        return {"depth": depth,
                "field_of_view": field_of_view,
                "focal_length": focal_length
                }