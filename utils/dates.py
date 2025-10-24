from datetime import datetime


def set_year_in_date(my_date: datetime, new_year: int) -> datetime:
    return datetime(year=new_year, month=my_date.month, day=my_date.day,
                    hour=my_date.hour, minute=my_date.minute, second=my_date.second)

