---
title: Docker container에서 cron을 이용한 스케줄링
date: 2022-07-01 22:30:00 +0900
categories: [docker]
tags: [mlops, linux]     # TAG names should always be lowercase
# pin: true
# mermaid: true
# math: true
# toc: false     # Table Of Contents
# comments: false 
img_path: /
---

cron은 유닉스 계열에서 사용가능한 잡 스케쥴러이다.
정기적으로 또는 특정 주기를 따라서 동작을 하고 싶을때 이용할 수 있다.   


docker-compose를 이용하여 컨테이너 실행시 바로 cron을 등록하고 스케줄링을 하고 싶었는데 대부분 아래와 같은 방법으로 알려주고 있었다.

~~~
$ crontab -e  # 등록 (에디터를 이용해서 작성)
$ crontab -l  # 등록된 잡 확인.
~~~

물론 이렇게 해도 상관은 없지만 에디터로 직접 등록을 해줘야 하기 때문에 docker-compose를 통한 자동화를 할수 없었다.
이걸 해결하기 위해서 cron 수식을 직접 등록해주는 방법으로 진행하였다.

~~~
# Dockerfile

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install cron

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

RUN ln -s $(which python3) /usr/local/bin/python

COPY ./main.py /test/main.py
COPY ./root /etc/cron.d/root

RUN chmod 0644 /etc/cron.d/root
RUN chmod +x /test/main.py

RUN touch /var/log/test.log
CMD cron && tail -f /var/log/test.log
~~~


주기적으로 실행할 파일 main.py와 cron 명령어가 들어있는 파일 root를 이동시켜준다. 
그다음 각각 퍼미션을 할당한다.


root 파일은 `/etc/cron.d/`에 위치시키는데 cron 명령어 실행시 여기에 위치한 파일들을 job으로 등록하는것 같았다. (자세한 정보는 조금 여유로울때 다시 조사..)

~~~python
from datetime import datetime

if __name__ == '__main__':
    print(datetime.now(), 'test')
~~~
~~~bash
* * * * * root /usr/local/bin/python /test/main.py >> /var/log/test.log 2>&1
~~~

간단하게 1분 간격으로 현재 시간을 출력하고 이를 로그로 기록해 보았다.
~~~
test_1  | 2022-07-01 06:52:01.822615 test
test_1  | 2022-07-01 06:53:01.857943 test
test_1  | 2022-07-01 06:54:01.892707 test
test_1  | 2022-07-01 06:55:01.927534 test
test_1  | 2022-07-01 06:56:01.962774 test
test_1  | 2022-07-01 06:57:02.003161 test
test_1  | 2022-07-01 06:58:01.042311 test
test_1  | 2022-07-01 06:59:01.079271 test
test_1  | 2022-07-01 07:00:01.115631 test
test_1  | 2022-07-01 07:01:01.149939 test
test_1  | 2022-07-01 07:02:01.186468 test
test_1  | 2022-07-01 07:03:01.226175 test
test_1  | 2022-07-01 07:04:01.263987 test
test_1  | 2022-07-01 07:05:01.299888 test
test_1  | 2022-07-01 07:06:01.338136 test
test_1  | 2022-07-01 07:08:01.414428 test
~~~

정확히 1분은 아니고 조금의 오차는 발생하나 크게 영향을 줄 정도는 아닌것 같다.

