# Created on Mon Aug 20 21:51:28 2018

# Author: XiaoTao Wang

"""
Setup script for hicpeaks.

This is a free software under GPLv3. Therefore, you can modify, redistribute
or even mix it with other GPL-compatible codes. See the file LICENSE
included with the distribution for more details.

"""
import os, sys, lib, glob
from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if (sys.version_info.major != 2) or (sys.version_info.minor != 7):
    print 'PYTHON 2.7 IS REQUIRED. YOU ARE CURRENTLY USING PYTHON ' + sys.version
    sys.exit(2)

# Guarantee Unix Format
for src in glob.glob('scripts/*'):
    text = open(src, 'rb').read().replace('\r\n', '\n')
    open(src, 'wb').write(text)

setup(
    name = 'hicpeaks',
    version = lib.__version__,
    author = lib.__author__,
    author_email = 'wangxiaotao686@gmail.com',
    url = 'https://github.com/XiaoTaoWang/HiCPeaks/',
    description = 'Identify real loops from Hi-C data.',
    keywords = 'Hi-C interaction contact loop peak',
    package_dir = {'hicpeaks':'lib'},
    packages = ['hicpeaks'],
    scripts = glob.glob('scripts/*'),
    long_description = read('README.rst'),
    classifiers = [
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        ]
    )

