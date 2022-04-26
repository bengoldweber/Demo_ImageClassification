# importing the required libraries
import json
import io
import glob
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

# Pass the parameter "pretrained" as "True" to use the pretrained weights:
model = models.densenet121(pretrained=True)
# switch to model to `eval` mode:
model.eval()


# define the function to pre-process the
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# load the mapping provided by the pytorch
imagenet_class_mapping = json.load(open('imagenet_class_index.json'))


# define the function to get the class predicted of image
# it takes the parameter: image path and provide the output as the predicted class
def get_category(image_path):
    # read the image in binary form
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
    # transform the image
    transformed_image = transform_image(image_bytes=image_bytes)
    # use the model to predict the class
    outputs = model.forward(transformed_image)
    _, category = outputs.max(1)
    # return the value
    predicted_idx = str(category.item())
    return imagenet_class_mapping[predicted_idx][1]


cat = get_category(image_path='static/uploads/sample_1.jpg')
print(cat)