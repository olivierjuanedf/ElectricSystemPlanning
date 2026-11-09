import re
from typing import Optional

REVERSE_SIGN = "reverse"
LINK_NAME_SEP = "-"


def set_link_name(origin_bus_name: str, dest_bus_name: str, carrier_name: str = "ac", sign: str = None) -> str:
    sign_name = f"-{sign}" if sign is not None else ""
    return f"{origin_bus_name}{LINK_NAME_SEP}{dest_bus_name}{sign_name}_{carrier_name}"


def get_bus_from_link_name(link_name: str) -> Optional[tuple[str]]:
    pattern = re.compile(r"^(?P<origin_bus>[a-z]{3})-(?P<dest_bus>[a-z]{3})_ac$")

    match = pattern.match(link_name)

    if match is None:
        return None

    return match.group("origin_bus"), match.group("dest_bus")


def set_per_origin_bus_links_msg(link_names: list[str]) -> str:
    links_msg = ''
    n_links = len(link_names)
    i_link = 0
    while i_link < n_links:
        common_origin_links = [link_names[i_link]]
        origin = link_names[i_link].split(LINK_NAME_SEP)[0]
        j = 1
        while i_link + j < n_links:
            current_origin = link_names[i_link + j].split(LINK_NAME_SEP)[0]
            if current_origin == origin:
                common_origin_links.append(link_names[i_link + j])
                j += 1
            else:
                break
        links_msg += f'\n- from {origin}: {[tuple(elt_link.split(LINK_NAME_SEP)) for elt_link in common_origin_links]}'
        i_link += j
    return links_msg
