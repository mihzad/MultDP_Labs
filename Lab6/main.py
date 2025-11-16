from PIL import Image
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2)
upsampler = RealESRGANer(scale=2, model=model, model_path='RealESRGAN_x2plus.pth',device=device)
#model.load_weights('RealESRGAN_x2.pth')

img = cv2.imread('satellite.jpg', cv2.IMREAD_COLOR)  # BGR

out, _ = upsampler.enhance(img)

cv2.imwrite('satellite_x2.png', out)


print("done.")
