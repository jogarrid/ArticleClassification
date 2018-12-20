import time
#import pandas as pd
from xml.etree import ElementTree as _ET
from xml.dom import minidom as _minidom
import re as _re
from suds import client
from base64 import b64encode as _b64encode
from collections import OrderedDict as _OrderedDict


# !pip install suds-jurko



class WosClient():
    """Query the Web of Science.You must provide user and password only to user premium WWS service with WosClient() as wos: results = wos.search(...)"""

    base_url = 'http://search.webofknowledge.com'
    auth_url = base_url + '/esti/wokmws/ws/WOKMWSAuthenticate?wsdl'
    search_url = base_url + '/esti/wokmws/ws/WokSearch?wsdl'
    searchlite_url = base_url + '/esti/wokmws/ws/WokSearchLite?wsdl'
    
    def __init__(self, user=None, password=None, SID=None, close_on_exit=True,lite=False):
        
#"""Create the SOAP clients. user and password for premium access."""
        self._SID = SID
        self._close_on_exit = close_on_exit
        search_wsdl = self.searchlite_url if lite else self.search_url
        self._auth = client.Client(self.auth_url)
        self._search = client.Client(search_wsdl)

        if user and password:
            auth = '%s:%s' % (user, password)
            auth = _b64encode(auth.encode('utf-8')).decode('utf-8')
            headers = {'Authorization': ('Basic %s' % auth).strip()}
            self._auth.set_options(headers=headers)

    def __enter__(self):
        """Automatically connect when used with 'with' statements."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close connection after closing the 'with' statement."""
        if self._close_on_exit:
            self.close()

    def __del__(self):
        """Close connection when deleting the object."""
        if self._close_on_exit:
            self.close()

    def connect(self):
        """Authenticate to WOS and set the SID cookie."""
        if not self._SID:
            self._SID = self._auth.service.authenticate()
            print('Authenticated (SID: %s)' % self._SID)

        self._search.set_options(headers={'Cookie': 'SID="%s"' % self._SID})
        self._auth.options.headers.update({'Cookie': 'SID="%s"' % self._SID})
        return self._SID

    def close(self):
        """Close the session."""
        if self._SID:
            self._auth.service.closeSession()
            self._SID = None

    def search(self, query, count=100, offset=1):
        """Perform a query. Check the WOS documentation for v3 syntax."""
        if not self._SID:
            raise RuntimeError('Session not open. Invoke .connect() before.')

        qparams = _OrderedDict([('databaseId', 'WOS'),
                                ('userQuery', query),
                                ('queryLanguage', 'en')])

        rparams = _OrderedDict([('firstRecord', offset),
                                ('count', count),
                                ('sortField', _OrderedDict([('name', 'RS'),
                                                            ('sort', 'D')]))])
        time.sleep(0.5)
        return self._search.service.search(qparams, rparams)

    def single(wosclient, wos_query, xml_query=None, count=10, offset=1):
        """Perform a single Web of Science query and then XML query the results."""
        result = wosclient.search(wos_query, count, offset)
        print (result)
        xml = _re.sub(' xmlns="[^"]+"', '', result.records, count=1).encode('utf-8')
        if xml_query:
            xml = _ET.fromstring(xml)
            return [el.text for el in xml.findall(xml_query)]
        else:
            return _minidom.parseString(xml).toprettyxml()

    def query(wosclient, wos_query, xml_query=None, count=10, offset=1, limit=100):    
    #"""Query Web of Science and XML query results with multiple requests."""
    
        results = [single(wosclient, wos_query, xml_query, min(limit, count-x+1), x) for x in range(offset, count+1, limit)]
        if xml_query:
            return [el for res in results for el in res]
        else:
            pattern = _re.compile(r'.*?<records>|</records>.*', _re.DOTALL)
            return ('<?xml version="1.0" ?>\n<records>' +
                '\n'.join(pattern.sub('', res) for res in results) +
                '</records>')

    def doi_to_wos(wosclient, doi):
    #"""Convert DOI to WOS identifier."""
        results = query(wosclient, 'DO=%s' % doi, './REC/UID', count=1)
        return results[0].lstrip('WOS:') if results else None
