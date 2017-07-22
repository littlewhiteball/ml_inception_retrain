import retrain_model
import datetime

start = datetime.datetime.now()

retrain_model.main(bottleneck_dir='tf_files/bottlenecks',
                   how_many_training_steps=1000,
                   model_dir='tf_files/inception',
                   output_graph='tf_files/retrained_graph.pb',
                   output_labels='tf_files/retrained_labels.txt',
                   image_dir='training')

end = datetime.datetime.now()

print('start: %s, end: %s' % (start, end))

