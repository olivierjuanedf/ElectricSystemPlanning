REVERSE_SIGN = "reverse"


def set_link_name(origin_bus_name: str, dest_bus_name: str, carrier_name: str = "ac", sign: str = None) -> str:
    sign_name = f"-{sign}" if sign is not None else ""
    return f"{origin_bus_name}-{dest_bus_name}{sign_name}_{carrier_name}"
