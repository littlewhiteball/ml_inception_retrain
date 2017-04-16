import os, sys
import tensorflow as tf
from os import walk

image_dir = sys.argv[1]

# Unpersists graph from file
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
            in tf.gfile.GFile("tf_files/retrained_labels.txt")]

for path, _, image_names in walk(image_dir):
    for image_name in image_names:
        image_path = os.path.join(path, image_name);

        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            predictions = sess.run(softmax_tensor, \
                    {'DecodeJpeg/contents:0': image_data})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            print(image_path)
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print('===')
    break;

