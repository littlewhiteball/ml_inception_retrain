import math, os, random, shutil, sys
from os import walk

home_dir = sys.argv[1]
training_dir = sys.argv[2]
validation_dir = sys.argv[3]
train_percent = sys.argv[4]

def create_empty_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

for home_path, _, image_names in walk(home_dir):
    # randomly allocate images to training and validation folder
    number_images = len(image_names)
    number_training = math.floor(float(number_images) * float(train_percent))
    index_training = random.sample(range(number_images), number_training)
    print('%s images will be copied to training directory %s' % (number_training, training_dir))
    print('%s images will be copied to validation directory %s' %
          (number_images - number_training, validation_dir))

    # remove content from training and validation folder
    create_empty_dir(training_dir)
    create_empty_dir(validation_dir)

    # copy images in index_training to training folder
    # copy rest to validation folder
    i = 0
    for image_name in image_names:
        shutil.copyfile(os.path.join(home_path, image_name),
                        os.path.join(training_dir, image_name) if i in index_training
                        else os.path.join(validation_dir, image_name))
        i = i + 1

