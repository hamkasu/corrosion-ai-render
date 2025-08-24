from datetime import datetime
import pytz

tz = pytz.timezone('Asia/Kuala_Lumpur')
print(datetime.now(tz))