import use_model
import datetime

start = datetime.datetime.now()

use_model.main('../validation')

end = datetime.datetime.now()

print('start: %s, end: %s' % (start, end))

