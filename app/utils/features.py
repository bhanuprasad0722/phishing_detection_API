# app/utils/features.py
import re
from urllib.parse import urlparse

def extract_advanced_features(url: str) -> dict:
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query
    
    shortening_services = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd']
    domain_keywords = domain.split('.')

    return {
        'url_length': len(url),
        'special_chars': len(re.findall(r"[/?=&]", url)),
        'suspicious_keywords': sum(kw in url.lower() for kw in ['login', 'signin', 'account', 'verify', 'secure', 'update', 'banking']),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'num_dots': url.count('.'),
        'num_digits': len(re.findall(r"\d", url)),
        'domain_length': len(domain),
        'uses_ip': 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain.split(':')[0]) else 0,
        'has_at_symbol': 1 if '@' in url else 0,
        'has_redirect': 1 if '//' in path else 0,
        'num_subdomains': len(domain.split('.')) - 2 if '.' in domain else 0,
        'suspicious_tld': 1 if any(domain.endswith(tld) for tld in ['.zip', '.tk', '.ml', '.ga', '.cf', '.xyz']) else 0,
        'has_http_in_domain': 1 if 'http' in domain else 0,
        'shortening_service': 1 if any(svc in domain for svc in shortening_services) else 0,
        'domain_in_path': 1 if any(dk in path.lower() for dk in domain_keywords) else 0,
        'num_parameters': len(query.split('&')) if query else 0,
        'prefix_suffix': 1 if '-' in domain else 0,
    }