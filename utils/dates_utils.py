from datetime import datetime, timedelta


def generate_dates(start_date_str: str, end_date_str: str):
    start_date = datetime.strptime(start_date_str, "%Y_%m_%d")
    end_date = datetime.strptime(end_date_str, "%Y_%m_%d")

    dates = []

    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    return dates
