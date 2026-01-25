# Aligning-and-Restoring-Imperfect-ssEM-images
pip3 install opencv-python torch scipy pillow scikit-image matplotlib EasyDict

test two images:
python inf.py --reference ./DATA/0000.tif --moving ./DATA/0001.tif --mask ./DATA/0002.tif --model_path ./models/FE_final.pth --output ./test2_aligned.tif

train your own model:
python train.py
