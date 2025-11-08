from model_cv import SmokeDetectionPipeline, test_single_image_3class, SmokeTextureFeatureExtractor
from segmentation_fire import FireSegmnetation, create_intensity_analysis_image, showInRow, analyze_single_image
from segmentation_smoke import SmokeSegmentation, create_smoke_analysis_image, analyze_single_image_smoke
import cv2

model_path = "smoke_fire_3class_model_full_final.pkl"
pipline = SmokeDetectionPipeline()
model = pipline.load_pipeline(model_path)

def find_fire_smoke(test_image_path):
    pr_class, _ = test_single_image_3class(pipline, test_image_path)

    if pr_class == 1:
        img = FireSegmnetation(test_image_path)
        combined_fire_mask = img.combined_threshhold_fire()
        feat, level = analyze_single_image(img.scene, combined_fire_mask)
        int_score = feat['intensity_score']
        output = cv2.cvtColor(create_intensity_analysis_image(img.scene, combined_fire_mask), cv2.COLOR_BGR2RGB)
        orig = cv2.cvtColor(img.scene, cv2.COLOR_BGR2RGB)
        showInRow([orig, output], ['Original', f'Detected fire, intensity {int_score:.3f}'])

    elif pr_class == 0:
        img = SmokeSegmentation(test_image_path)
        smoke_mask = img.combined_threshhold_smoke()
        feat, level = analyze_single_image_smoke(img.scene, smoke_mask)
        int_score = feat['intensity_score']
        output = create_smoke_analysis_image(img.scene, smoke_mask)
        orig = cv2.cvtColor(img.scene, cv2.COLOR_BGR2RGB)
        showInRow([orig, output], ['Original', f'Detected smoke, intensity {int_score:.3f}'])

    else:
        orig = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)
        showInRow([orig], ['Clear Original'])

test_image_path = 'data/dataset/image/WEB11805.jpg'
find_fire_smoke(test_image_path)

test_image_path2 = 'data/dataset/image/WEB11518.jpg'
find_fire_smoke(test_image_path2)