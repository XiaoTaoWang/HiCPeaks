# Created on Sat May 19 15:53:06 2018

# Author: XiaoTao Wang

import xmlrpclib
from pkg_resources import parse_version as V

__author__ = 'XiaoTao Wang'
__version__ = '0.1.0'
__license__ = 'GPLv3+'

## Check for update
try:
    pypi = xmlrpclib.ServerProxy('http://pypi.python.org/pypi')
    available = pypi.package_releases('hicpeaks')
    if V(__version__) < V(available[0]):
        print '*'*75
        print 'Version %s is out of date, Version %s is available.' % (__version__, available[0])
        print 'Download the latest version from https://pypi.org/project/HiCPeaks/'
        print
        print '*'*75
except:
    pass

Me = __file__

