---
title: 원격서버의 container로 ssh 접속하기
date: 2021-08-03 17:32:00 +0900
categories: [tips]
tags: [ubuntu, docker, ssh, os]     # TAG names should always be lowercase
img_path: /
---



일반적으로 ssh를 통해서 원격으로 서버에 접속해서 CLI 환경에서 작업을 많이 하게 된다.

문제는 원격 서버에서 container를 열어서 새로운 서비스를 실행한다면 외부에서 접속하기 위해서 `서버` - `컨테이너`를 통과하여야 한다.



우선 container 실행시 옵션을 통해서 포트를 맵핑시켜준다.

아래처럼 옵션을 주면  7777번 포트를 통해서 컨테이너로 접속하면 컨테이너의 8888번 포트에서 실행중인 프로세스에 접근할 수 있다. ([참고](https://docs.docker.com/config/containers/container-networking/))

~~~bash
$ docker run ... -p 7777:8888 ...
~~~



다음으로 컨테이너에서 ssh 설치 및 `sshd_config` 수정 (root 계정일 경우 root 접속 허용 할것. )

~~~bash
$ apt-get update
$ apt-get install ssh

---------- install finished -----------

$ vi /etc/ssh/sshd_config

---------------------------------------
PermitRootLogin yes # root 계정 이용시
~~~



다음으로 필요에 따라서 접속 비밀번호를 변경한다.

~~~bash
$ passwd {계정}
~~~



마지막으로 재부팅하면 끝.

~~~bash
$ service ssh restart
~~~



아래와 같이 접속하면 해당 서비스로 바로 접속 가능하다.

~~~bash
$ ssh {container 계정}@{container가 돌아가는 서버  IP} -p 7777
~~~





> 네이버에서 이전한 포스트로 오래전 작성되어서 내용이 잘못되었을 수 있습니다. 오류나 잘못된 정보 전달시 댓글로 알려주세요!
{: .prompt-info }