import torch
import piq

class QualityAssessment:
    def __init__(self, brisque_threshold=40.0):
        self.brisque_threshold = brisque_threshold

    def evaluate(self, image):
        '''
        Evaluates the image quality using BRISQUE.
        image: torch.Tensor of shape (B, C, H, W) normalized to [0, 1]
        Returns a boolean mask: True if quality is POOR (BRISQUE > threshold)
        BRISQUE lower is better. So if it's > threshold, we need SIGAN.
        '''
        # Ensure image is in [0, 1]
        image_clamped = torch.clamp(image, 0, 1)
        brisque_scores = piq.brisque(image_clamped, data_range=1.0, reduction='none')
        # True means poor quality -> requires SIGAN
        return brisque_scores > self.brisque_threshold
