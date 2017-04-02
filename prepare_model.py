import math, os, random, shutil, sys
from os import walk

def create_empty_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def main(home_dir, training_dir, validation_dir, train_percent):
    # remove content from training and validation folder
    create_empty_dir(training_dir)
    create_empty_dir(validation_dir)

    for home_path, image_dirs, _ in walk(home_dir):
        for image_dir in image_dirs:
            # replicate folder structure of home_dir to
            # training_dir and validation_dir then copy files over
            training_sub_dir = os.path.join(training_dir, image_dir)
            validation_sub_dir = os.path.join(validation_dir, image_dir)
            create_empty_dir(training_sub_dir)
            create_empty_dir(validation_sub_dir)

            image_path = os.path.join(home_path, image_dir)
            for _, _, image_names in walk(image_path):
                # randomly allocate images to training and validation folder
                number_images = len(image_names)
                number_training = math.floor(float(number_images) * float(train_percent))
                index_training = random.sample(range(number_images), number_training)
                print('%s images will be copied to training directory %s' %
                      (number_training, training_sub_dir))
                print('%s images will be copied to validation directory %s' %
                      (number_images - number_training, validation_sub_dir))

                # copy images in index_training to training folder
                # copy rest to validation folder
                i = 0
                for image_name in image_names:
                    shutil.copyfile(os.path.join(image_path, image_name),
                                    os.path.join(training_sub_dir, image_name) if i in index_training
                                    else os.path.join(validation_sub_dir, image_name))
                    i = i + 1

