import os
from PIL import Image
for image in os.listdir('../ECS171-Proj/all/'):
    print(image)
    im = Image.open('../ECS171-Proj/all/' + image)
    test = len(image)
    print(len)
    image = image[0:test-4]
    print(image)
    if im.format == 'PNG':
        # and is not RGBA
        if im.mode != 'RGBA':
            im.convert("RGBA").save(f"../ECS171-Proj/images/pokemon2/{image}.png")
        else:
            im.save(f"../ECS171-Proj/images/pokemon2/{image}.png")
    elif im.format == 'JPEG':
        print("enter")
        im.convert("RGBA").save(f"../ECS171-Proj/images/pokemon2/{image}.png")
