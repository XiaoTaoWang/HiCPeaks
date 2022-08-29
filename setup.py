# Created on Mon Aug 20 21:51:28 2018

# Author: XiaoTao Wang

"""
Setup script for hicpeaks.

This is a free software under GPLv3. Therefore, you can modify, redistribute
or even mix it with other GPL-compatible codes. See the file LICENSE
included with the distribution for more details.

"""
import os, sys, hicpeaks, glob
import setuptools

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if ((sys.version_info.major==2) and (sys.version_info.minor!=7)) or \
   ((sys.version_info.major==3) and (sys.version_info.minor<5)):
    print('PYTHON 2.7/3.5+ IS REQUIRED. YOU ARE CURRENTLY USING PYTHON {}'.format(sys.version.split()[0]))
    sys.exit(2)

# Guarantee Unix Format
for src in glob.glob('scripts/*'):
    text = open(src, 'r').read().replace('\r\n', '\n')
    open(src, 'w').write(text)

setuptools.setup(
    name = 'hicpeaks',
    version = hicpeaks.__version__,
    author = hicpeaks.__author__,
    author_email = 'wangxiaotao686@gmail.com',
    url = 'https://github.com/XiaoTaoWang/HiCPeaks/',
    description = 'A Python implementation for BH-FDR and HiCCUPS',
    keywords = 'Hi-C chromatin interaction contact loops',
    long_description = read('README.rst'),
    long_description_content_type='text/x-rst',
    scripts = glob.glob('scripts/*'),
    packages = setuptools.find_packages(),
    classifiers = [
        'Programming Language :: Python',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        ]
    )

