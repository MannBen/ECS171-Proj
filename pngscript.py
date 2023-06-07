import os
from PIL import Image
for image in os.listdir('./images/pokemon2/'):
    print(image)
    im = Image.open('./images/pokemon2/' + image)
    # If is png image
    if im.format == 'PNG':
        # and is not RGBA
        if im.mode != 'RGBA':
            im.convert("RGBA").save(f"./images/pokemon3/{image}")
        else:
            im.save(f"./images/pokemon3/{image}")
    elif im.format == 'JPEG':
        print("enter")
        im.convert("RGBA").save(f"./images/pokemon3/{image}")
