import os
import csv
import shutil

# Define the path to the CSV file
csv_file = './ECS171-Proj/pokemon.csv'

# Define the path to the directory containing the PNG images
images_directory = './ECS171-Proj/images/pokemon2'

# Define the column numbers (0-based index) for the attribute columns
attribute_column1 = 1  # Adjust this value to the correct column number
attribute_column2 = 2  # Adjust this value to the correct column number

# Create a directory to store the sorted images
output_directory = './ECS171-Proj/imagesSorted'
os.makedirs(output_directory, exist_ok=True)

# Read the CSV file
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if present

#     # Iterate over each row in the CSV file
    for row in reader:
        # Extract the attributes from the respective columns
        print(row[attribute_column1], row[attribute_column2])
        attribute1 = row[attribute_column1]
        attribute2 = row[attribute_column2]

        # Create a directory for the current combination of attributes
        destination_directory1 = os.path.join(output_directory, attribute1)
        destination_directory2 = os.path.join(output_directory, attribute2)
        os.makedirs(destination_directory1, exist_ok=True)
        os.makedirs(destination_directory2, exist_ok=True)
        print(destination_directory1, destination_directory2)

#         # Find the corresponding PNG image file
        image_file = os.path.join(images_directory, (row[0] + '.png'))  # Assuming the file name is in the first column
        print(image_file)
        # Check if the image file exists and if it's a PNG file
        if os.path.isfile(image_file) and image_file.lower().endswith('.png'):
            shutil.copy(image_file, destination_directory1)
            # shutil.copy(os.path.join(destination_directory1, (row[0] + '.png')), os.path.join(images_directory, (row[0] + '.png')))
            shutil.move(image_file, destination_directory2)

print("Image sorting completed.")
