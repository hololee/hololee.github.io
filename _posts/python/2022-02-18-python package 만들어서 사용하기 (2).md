---
title: python package 만들어서 사용하기 (2)
date: 2022-05-16 11:46:00 +0900
categories: [python]
tags: []     # TAG names should always be lowercase
# pin: true
# mermaid: true
# math: true
# toc: false     # Table Of Contents
# comments: false 
img_path: /
---

[python package 만들어서 사용하기 (1)](https://blog.naver.com/lccandol/222651163325)에서 이어서 진행됩니다.

python package를 만들어서 사용하면 dirtory level에 따른 import에 크게 신경쓰지 않아도 되고 버전별로 관리하기 편해집니다.
여기에 추가적으로 CLI 에서 사용 가능한 메서드를 작성해보겠습니다.


## script를 등록하는 경우

커맨드 라인에서 메서드를 이용하는 방법중 하나는 script를 등록하여 사용한는 것 입니다.
이 방식은 shell script 를 사용하기 때문에, 꼭 python 스크립트가 아니여도 적용이 가능하다는 장점이 있습니다.

setup.py 와 같은 위치에 bin 디렉터리를 생성하고 스크립트를 추가합니다.

~~~
.
├── README.md
├── requirements.txt
├── setup.py
├── bin
│   └── test_script
└── src
    └── {package_name}
        ├── __init__.py
~~~

test_script 안에는 간단한 path 출력 메서드를 작성하겠습니다.

~~~python
#!/usr/bin/env python

import os
print os.getcwd()
~~~

여기서 `#! (shebang)`은 스크립트의 시작을 알리고 보통 shell script 에서는 `#! /bin/sh`로 시작합니다.  
위처럼 env + python 으로 지정하는 경우 env 에서 python 바이너리의 위치를 찾아서 실행해줍니다. 

다시 **setup.py** 로 돌아와서 아래의 파라미터를 추가해줍니다.

~~~python
setup(
    ...
    scripts=['bin/test_script'],
    ...
)
~~~

이제 페키지를 설치하고 CLI 에서 아래와 같이 입력해주면 setup에 script로 지정해준 스크립트를 실행하게 됩니다.

~~~
$ test_script
~~~

## Entry Point를 이용하는 경우
entry point는 스크립트가 아닌 python 함수를 CLI에서 바로 사용할 수 있습니다.
스크립트가 아니다 보니 python에서 테스트가 쉬운 장점을 가지고 있습니다.

우선 패키지 안에 command line을 처리하는 파이썬 스크립트 cli_python.py 를 추가해줍니다.

~~~
.
├── README.md
├── requirements.txt
├── setup.py
└── src
    └── {package_name}
        ├── __init__.py
        └── cli_python.py
~~~

해당 코드에서 파라미터로 들어오는 입력을 출력해주는 함수를 추가해보겠습니다.

~~~python
import argparse

def print_txt(txt):
    print(txt)

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--text', action="store", requires=True)

        args = parser.parse_args()

        print_txt(args.text)
~~~


다시 setup.py로 돌아와서 아래의 파라미터를 추가해줍니다.
cli_test는 {package_name}.cli_python 파일의 main 함수를 의미하게 됩니다.

~~~python
setup(
    ...
    entry_points = {
        'console_scripts': ['cli_test={package_name}.cli_python:main'],
    }
    ...
)
~~~

이제 페키지를 설치하고 CLI 에서 아래와 같이 입력해주면 함수가 실행됩니다.

~~~bash
$ cli_test -t test_test_test
~~~
