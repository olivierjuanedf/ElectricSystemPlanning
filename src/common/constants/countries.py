def set_country_trigram(country: str) -> str:
    return f'{country[:3].lower()}'


def is_country_trigram(my_str) -> bool:
    return isinstance(my_str, str) and len(my_str) == 3 and my_str.islower()
