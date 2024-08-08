import pytz
from timezonefinder import TimezoneFinder
import pandas as pd

# Initialize TimezoneFinder
tf = TimezoneFinder()

# Cache for time zones to avoid redundant lookups
tz_cache = {}

def convert_to_utc_vectorized(lon, lat, local_time):
    tz_str = tz_cache.get((lon, lat))
    if tz_str is None:
        tz_str = tf.timezone_at(lng=lon, lat=lat)
        tz_cache[(lon, lat)] = tz_str

    tz = pytz.timezone(tz_str)
    local_time = tz.localize(local_time)
    utc_time = local_time.astimezone(pytz.utc)
    return utc_time, extract_offset(utc_time), local_time, extract_offset(local_time)


# Function to extract the offset in hours and minutes from time
def extract_offset(x):
    x = str(x)
    if len(x.split("+")) > 1:
        return "+"+x.split("+")[-1]
    else:
        return "-"+x.split("-")[-1]

