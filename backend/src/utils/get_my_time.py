from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def get_my_time():
    # 1. Get current UTC time (Timezone-aware)
    utc_now = datetime.now(timezone.utc)

    # 2. Convert UTC time to Malaysia Time
    malaysia_tz = ZoneInfo("Asia/Kuala_Lumpur")
    malaysia_time = utc_now.astimezone(malaysia_tz)

    # Removing timezone information
    malaysia_time = malaysia_time.replace(tzinfo=None)
    return (malaysia_time)
