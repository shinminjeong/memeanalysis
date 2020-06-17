"""
A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='meme analysis',

    version='0.0.1',

    description='COVID19 meme analysis',
    long_description=long_description,

    url='https://github.com/shinminjeong/memeanalysis',

    # Author details
    author='Minjeong Shin',
    author_email='minjeong.shin@anu.edu.au',

    license='N/A',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: N/A',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='sample setuptools development',

    packages=find_packages(exclude=['data']),

    install_requires=['Django', 'numpy', 'requests', 'requests_oauthlib', 
                      'tweepy', 'newspaper3k', 'imagehash', 'wordcloud',
                      'sklearn', 'tensorflow', 'langdetect', 'mysqlclient'],

    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    package_data={
        'sample': ['package_data.dat'],
    },

    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },
)
