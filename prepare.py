import prepare_model
import datetime, sys

start = datetime.datetime.now()

percentage = .95
if len(sys.argv) > 1:
    percentage = sys.argv[1]

prepare_model.main('../images/', '../training', '../validation/', percentage)

end = datetime.datetime.now()

print('start: %s, end: %s' % (start, end))

