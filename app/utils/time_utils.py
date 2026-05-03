def parse_timestamp(ts: str) -> float:
    """
    Convert timestamp string to seconds.
    
    Args:
        ts: Timestamp in format "HH:MM:SS.ms" (e.g., "01:02:03.456")
        
    Returns:
        Total seconds as float

    """
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)
