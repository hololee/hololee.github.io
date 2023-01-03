---
title: python package 만들어서 사용하기 (1)
date: 2022-02-18 15:36:00 +0900
categories: [python]
tags: []     # TAG names should always be lowercase
# pin: true
# mermaid: true
# math: true
# toc: false     # Table Of Contents
# comments: false 
img_path: /
---

특정 프로젝트를 페키지로 묶어서 사용해야할 일이 생겼습니다.
setuptools를 이용하면 페키지를 쉽게 만들 수 있는데 나중에 참고 하는 용도로 보기 위해서 정리해둡니다.



패키지로 만들려는 프로젝트의 구조는 다음과 같습니다.
- 패키지로 만드려는 디렉터리에는 꼭 __init__.py 를 추가하여야 합니다.  


~~~
.
├── README.md
├── requirements.txt
├── setup.py
└── src
    └── {package_name}
        ├── __init__.py
        ├── {sub_package_name}
        │   ├── __init__.py
        │   ├── pack1.py
        │   ├── pack2.py
        └── other_data
            ├── info.db
       
~~~

<br>

이때 setup.py를 통해서 패키지를 생성할 수 있습니다.   
setup.py 작성 항목은 많지만 여기서는 간단하게 몇가지만 작성하였습니다.

~~~python
from setuptools import find_packages, setup

install_requires = [
    'numpy==1.19.4',
    'opencv-python==4.5.2.52',
]

setup(
    name='{package_name}',
    version='1.0',
    description='pack to kage',
    author='hololee',
    author_email='lccandol@naver.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'{package_name}': ['other_data/*.db']},
    python_requires='>=3.8',
    install_requires=install_requires,
)
~~~

- `{package_name}`은 실제 이용할때 호출할 package name 입니다.
- find_packages()를 이용해서 빠르게 패키지를 등록할 수 있습니다. 여기서는 `['{package_name}', {package_name}.{sub_package_name}]`이 됩니다.
- 기본적으로 패키지화를 하는 경우 py 파일 이외에는 포함이 되지 않습니다. 따라서 다른 형식의 파일은 package_data 를 통해서 추기해줄 수 있습니다. `{'{package_name}': ['other_data/*.db']}`의 의미는 `{package_name}` 위치에서 other_data의 모든 .db 확장자 파일을 추가하겠다는 의미입니다.


<br>

제가 작성한 코드의 경우에는 pack1.py에서 info.db를 사용해야 했습니다. 이러한 경우나 또는 해당 패키지를 이용하면서 내부의 파일을 이용하여야 한다면 다음과 같은 방법으로 불러올 수 있습니다.

~~~python
import pkg_resources

INFORMATION_DB = pkg_resources.resource_filename('{package_name}', 'other_data/info.db')
~~~


이제 작성한 setup.py를 이용해서 패키지를 만듭니다.

~~~bash
$ python3 setup.py install

// 또는 
$ pip3 install .
~~~

