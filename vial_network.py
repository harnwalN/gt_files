import torch, torchvision, cv2, os
from PIL import Image
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class Vial_Network:
    def __init__(self, experiment, spec_video, index, video_path, irrel_vials, lamp_var):
        self.experiment = experiment
        self.spec_video = spec_video
        self.index = index
        self.video_path = video_path
        self.irrel_vials = irrel_vials
        self.lamp_var = lamp_var
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self._get_model_instance_segmentation(num_classes=2)
        self.model_path = 'gt_newVial_nn.pth'
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)

    def get_random_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_idx = random.randint(0, total_frames - 1)
        frame_idx = 588
        print(f"Sample Frame: #{frame_idx}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Could not read frame from video.")
        return frame

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        transform = self._get_transform()
        img_tensor = transform(img)
        return img_tensor

    def predict_and_display(self, figsize=(12, 8)):
        random_frame = self.get_random_frame()
        img_tensor = self.preprocess_frame(random_frame)

        self.model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            prediction = self.model([img_tensor])[0]

        pred_boxes = prediction['boxes'].tolist()
        pred_scores = prediction['scores'].tolist()
        combdata = []
        for boxes, scores in zip(pred_boxes,pred_scores):
            row = {
                "x1": boxes[0],
                "y1": boxes[1],
                "x2": boxes[2],
                "y2": boxes[3],
                "scores": scores
            }
            combdata.append(row)
        vial_df = pd.DataFrame(combdata)
        if self.lamp_var == "Yes":
            vial_df = vial_df.drop(vial_df[vial_df['scores']<= 0.80].index)
        # else:
            # vial_df = vial_df.drop(vial_df[vial_df['scores']<= 0.65].index)
        vial_df = vial_df.sort_values(by=['x1', 'x2'])
        vial_df.reset_index(drop=True, inplace=True)
        vial_df.insert(0, 'vials', range(1, len(vial_df) + 1))
        irrel_vials_indices = [x - 1 for x in self.irrel_vials]
        vial_df.drop(irrel_vials_indices, axis=0, inplace=True)

        output_csv_path = f'{self.experiment}/{self.spec_video}/trim_{self.index}_{self.spec_video}_vials_pos.csv'
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        vial_df.to_csv(output_csv_path, index=False)
        print(vial_df)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def _get_transform(self):
        custom_transforms = []
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)

    def _get_model_instance_segmentation(self, num_classes):
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
