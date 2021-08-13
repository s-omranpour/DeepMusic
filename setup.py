from setuptools import setup

url = 'https://github.com/s-omranpour/DeepNote'

setup(
    name='deepnote',
    version='0.2.3',
    description='A python package to manipulate MIDI files, representing them and converting them into a proper format for feeding to machine learning models.',
    url=url,
    author='Soroush Omranpour',
    author_email='soroush.333@gmail.com',
    keywords='midi, deep learning, music',
    packages=['deepnote'],
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy >= 1.7.0', 
        'miditoolkit >= 0.1.14',
        'chorder >= 0.1.2',
        'midi2audio >= 0.1.1'
    ],
    license="MIT license",

    project_urls={  # Optional
        'Bug Reports': f'{url}/issues',
        'Source': url,
    },
)