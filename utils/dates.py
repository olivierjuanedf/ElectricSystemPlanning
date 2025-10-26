from datetime import datetime


def set_year_in_date(my_date: datetime, new_year: int) -> datetime:
    return datetime(year=new_year, month=my_date.month, day=my_date.day,
                    hour=my_date.hour, minute=my_date.minute, second=my_date.second)


def remove_useless_zero_in_date(date: str, date_sep: str = '/') -> str:
    suppr_first_char = False
    if date.startswith('0'):
        date = date_sep + date
        suppr_first_char = True
    chars_tb_replaced = {f'{date_sep}0{i + 1}': f'{date_sep}{i + 1}' for i in range(9)}
    for old_char, new_char in chars_tb_replaced.items():
        if old_char in date:
            date = date.replace(old_char, new_char)
    if suppr_first_char:
        date = date[1:]
    return date


def set_temporal_period_str(min_date: datetime, max_date: datetime, print_year: bool,
                            min_str_fmt: bool = True, date_sep: str = '/', rm_useless_zeros: bool = True) -> str:
    full_date_fmt = f'%Y{date_sep}%m{date_sep}%d'
    date_wo_year_fmt = f'%m{date_sep}%d'
    date_with_only_day_fmt = '%d'
    sep_str = '-'
    if sep_str == date_sep:
        sep_str = 'to'

    # set min date as str
    min_date_fmt = full_date_fmt if print_year else date_wo_year_fmt
    min_date_str = min_date.strftime(format=min_date_fmt)
    if rm_useless_zeros:
        min_date_str = remove_useless_zero_in_date(date=min_date_str, date_sep=date_sep)

    # idem for max date, with more cases...
    # provide str with full date if
    # (i) not 'min str fmt' requested and print year or
    # (ii) print year and max date year > min date one
    if (not min_str_fmt and print_year) or (print_year and max_date.year > min_date.year):
        max_date_fmt = full_date_fmt if print_year else date_wo_year_fmt
    # with month and day only if
    # (i) not min str fmt and not print year or
    # (ii) not print year or
    # (iii) print year and same year for min and max dates in case of min str fmt requested
    elif not min_str_fmt or max_date.month > min_date.month:
        max_date_fmt = date_wo_year_fmt
    # with day only in other cases i.e., not min str fmt, and same year and month for min and max dates
    else:
        max_date_fmt = date_with_only_day_fmt

    max_date_str = max_date.strftime(format=max_date_fmt)
    if rm_useless_zeros:
        max_date_str = remove_useless_zero_in_date(date=max_date_str, date_sep=date_sep)

    return f'{min_date_str}{sep_str}{max_date_str}'
