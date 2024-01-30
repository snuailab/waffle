from nvitop import Device


def cpu_check() -> str:
    """Check CPU Memory using psutil

    Returns:
        str: cpu percent
    """
    import psutil

    return f"{psutil.cpu_percent():.2f}%"


def memory_check() -> str:
    """Check Memory using psutil

    Returns:
        str: memory used, memory total
    """
    import psutil

    mem = psutil.virtual_memory()
    return f"{mem.used / 1024 ** 3:.2f}G / {mem.total / 1024 ** 3:.2f}G"


def gpu_check() -> str:
    """Check GPU Memory using nivdia-smi

    Returns:
        str: gpu_id, memory used, memory total
    """
    width = 8
    s = [["GPU id", "Mem %", "Util %"]]
    devices = Device.all()
    for device in devices:
        s.append(
            [f"{device.index}", f"{device.memory_percent():.2f}", f"{device.gpu_percent():.2f}"]
        )

    result = ""
    for s_ in s:
        result += "".join(list(map(lambda x: x.ljust(width), s_))) + "\n"

    return result
