import tensorflow as tf, sys, os
from os import walk

validation_dir = sys.argv[1]

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with open('results.txt', 'w') as result_file:
    for validation_path, dir_names, _ in walk(validation_dir):

        # Each item represents the accuracy of a category
        accuracy = []

        # Every directory represents one category
        for index, dir_name in enumerate(dir_names):
            if dir_name != label_lines[index]:
                raise ValueError('validation features does not match test features')

            # Iterate images of a category
            for category_path, _, image_names in walk(os.path.join(validation_path, dir_name)):
                # Get category name from category_path
                _, category = os.path.split(category_path)

                correct = 0
                total = 0
                for image_name in image_names:
                    # Skip non jpg files
                    if image_name.split('.')[-1] not in ['jpg', 'jpeg', 'JPG', 'JPEG']:
                        continue;

                    # Read in the image_data
                    image_data = tf.gfile.FastGFile(os.path.join(category_path, image_name), 'rb').read()

                    with tf.Session() as sess:
                        # Feed the image_data as input to the graph and get first
                        # prediction
                        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

                        # Sort to show labels of first prediction in order of confidence
                        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

                        # Increment total. If predicted matches, increment
                        # correct count
                        total = total + 1
                        if label_lines[top_k[0]] == category:
                            correct = correct + 1
                        else:
                            # Write result of every image to a file
                            result_file.write('Category: %s, image: %s\n' % (category, image_name))
                            for node_id in top_k:
                                human_string = label_lines[node_id]
                                score = predictions[0][node_id]
                                result_file.write('%s (score = %.5f)\n' % (human_string, score))

                            print()

                        # Print current status
                        print('processing image %s of category %s, result %s' % (image_name, category, label_lines[top_k[0]]))

                # Calculate accuracy = #correct / #total
                accuracy.append((category, correct/total))

                # Only the first level of image files of category directory
                break

        # Print out accuracy
        print(accuracy)

        # Only the first level of sub folders of validation directory
        break

























































