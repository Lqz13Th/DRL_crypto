from datetime import datetime, timedelta


def generate_dates(start_date_str: str, end_date_str: str):
    # 将字符串解析为日期对象
    start_date = datetime.strptime(start_date_str, "%Y_%m_%d")
    end_date = datetime.strptime(end_date_str, "%Y_%m_%d")

    # 用列表来存储生成的日期
    dates = []

    current_date = start_date
    while current_date <= end_date:
        # 将日期格式化为字符串并添加到列表中
        dates.append(current_date.strftime("%Y-%m-%d"))
        # 获取下一天
        current_date += timedelta(days=1)

    return dates
