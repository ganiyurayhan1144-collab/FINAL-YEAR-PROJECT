
import re
from urllib.parse import urlparse

def extract_features(url):
    parsed = urlparse(url)

    features = [
        len(url),
        len(parsed.netloc),
        url.count("."),
        url.count("-"),
        url.count("@"),
        url.count("?"),
        url.count("="),
        url.count("/"),
        sum(char.isdigit() for char in url),
        sum(char.isalpha() for char in url),
        int(parsed.scheme == "https"),
        int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        int("@" in url),
        int(bool(parsed.path))
    ]

    return features
