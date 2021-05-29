""" import os
import warnings
import json
import copy
import re

import six
from six import BytesIO
import astropy.units as u
import astropy.coordinates as coord
import astropy.table as tbl
import astropy.utils.data as aud
from collections import OrderedDict
import astropy.io.votable as votable
from astropy.io import ascii, fits

from ..query import BaseQuery
from ..utils import commons
from ..utils import async_to_sync
from ..utils import schema
from . import conf
from ..exceptions import TableParseError """

""" class TrilegalClass(BaseQuery):
    """
    # This class is for querying TRILEGAL.
    # There is an upper limit of how long queries can take. 
    # No error will be given; instead, the page just won't full load. 
    # """
    # TRILEGAL_url = "http://stev.oapd.inaf.it/cgi-bin/trilegal"
    # pass """

""" import requests
url = "http://stev.oapd.inaf.it/cgi-bin/trilegal"
response = requests.get(url,
                        params = {'gc_l': 'gc_l'})

print(f"your request to {url} came back w/ status code {response.status_code}")
html_text = response.json()
print(html_text) """

import xmltojson
import json
import requests


# Sample URL to fetch the html page
url = "http://stev.oapd.inaf.it/cgi-bin/trilegal"

# Headers to mimic the browser
headers = {
	'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 \
	(KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}

# Get the page through get() method
html_response = requests.get(url=url, headers = headers)

# Save the page content as sample.html
with open("trilegal.html", "w") as html_file:
	html_file.write(html_response.text)
	
with open("trilegal.html", "r") as html_file:
	html = html_file.read()
	json_ = xmltojson.parse(html)
	
with open("trilegal_data.json", "w") as file:
	json.dump(json_, file)
	
print(json_)

