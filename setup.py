#python -m venv myenv
#myenv\Scripts\activate


from setuptools import find_packages, setup

from typing import List

def get_requirements(file_path:str)->List[str]:
    lst_out = []
    with open(file_path, "r") as f:
	    lines = f.readlines()
    
    lst_out=[i.replace('\n','') for i in lines if '-e .' not in i]

        
    return lst_out



setup(

    name = 'mlproject_strmlt',

    version = '0.0.1',

    author = 'diptarshis',

    author_email="diptarshis@gmail.com",

packages=find_packages(),
install_requires = get_requirements('requirements.txt')
)

