from collections import Counter
import csv
import json
import numpy as np


def count_unique_sites(urls):
    websites = list()
    
    for url in urls:
        if url.startswith("http"):
            site = url.split('/')[2]
        else:
            site = url.split('/')[0]
        websites.append(site)
    
    counted_sites = dict(Counter(websites))
    sorted_sites = sorted(counted_sites.items(), key=lambda x: counted_sites[x[0]], reverse=True)
    
    return sorted_sites


def get_site(url):
    if url.startswith("http"):
        site = url.split('/')[2]
    else:
        site = url.split('/')[0]
    return(site)