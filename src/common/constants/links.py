import re
from typing import Optional

REVERSE_SIGN = "reverse"


def set_link_name(origin_bus_name: str, dest_bus_name: str, carrier_name: str = "ac", sign: str = None) -> str:
    sign_name = f"-{sign}" if sign is not None else ""
    return f"{origin_bus_name}-{dest_bus_name}{sign_name}_{carrier_name}"


def get_bus_from_link_name(link_name: str) -> Optional[tuple[str]]:
    pattern = re.compile(r"^(?P<origin_bus>[a-z]{3})-(?P<dest_bus>[a-z]{3})_ac$")

    match = pattern.match(link_name)

    if match is None:
        return None

    return match.group("origin_bus"), match.group("dest_bus")
