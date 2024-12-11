---
title: docker container의 pycharm x11 forwarding
date: 2020-12-21 00:40:00 +0900
categories: [docker]
tags: [ubuntu, pycharm, ssh, x11_forwarding, os]     # TAG names should always be lowercase
img_path: /
---

매번 Anaconda를 사용해서 환경을 구성하다가 [Docker를 공부하고 세팅](https://github.com/hololee/How-to-set-up-deeplearning-server)을 해보았다.



목표는 `mobaXterm` - `ubuntu` - `docker` 로  원격 윈도우 PC 에서 ssh 를 통해서 ubuntu 위에서 돌아가는 docker container 의 x11 forwarding을 하는것이다.



바로 pycharm을 공유 volume 에 넣고 실행하려니 아래와 같은 오류가 발생했다..

~~~bash
Startup Error 
Unable to detect graphics environment
~~~



<br/>

찾아본 결과 ssh를 이용해서 접속할때 docker container 내부에서  x11forwarding 을 이용하려면 Xserver 를 호스트랑 연동 해주어야 했다.



1. xhost 연결.

   ~~~bash
   $ xhost +local:docker
   // $ xhost
   ~~~

2. 다음 아래와 같이 image 를 run 할때  다음과 같은 옵션을 준다.

   ~~~bash
   nvidia-docker run -it --rm --net=host -v ${HOME}/Desktop/pycharm:${HOME}/data -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix nvidia/cuda:10.0-cudnn7-runtime /bin/bash
   ~~~

3. 컨테이너 안에서 export 로 display 값 확인.

   ~~~bash
   $ export
   ...
   declare -x DISPLAY=":0"
   ~~~

4. ssh 로 접속하려는 PC 에서 export 로 display 값 확인(보통 localhost:xx.0 으로 되어있다.).

   ~~~bash
   $ export
   ...
   declare -x DISPLAY="localhost:0.0"
   ~~~

   

ssh로 접속하려는 pc 도 같이 맞춰주니(":0"으로 변경) 동작을 하였다..

사실 계속 수정해 가며 하다보니 갑자기 동작해서 원인을 못찾고 있다..



## 원인 파악 및 해결



export DISPLAY=":0" 으로 지정을 하면 원본 OS 본체에 해당하는 곳에서 디스플레이가 표시된다. 그렇다고 docker 안에서 ssh 로 연결된 localhost:xx.0 으로 연결을 하면  아래와 같은 오류를 보냈다.(물론  docker 밖에서 는 localhost:xx.0 로 x11forwarding 이 잘 되었다.)

~~~bash
Unable to init server: Could not connect: Connection refused 
(gedit:27585): Gtk-WARNING **: 01:30:12.406: cannot open display: localhost:11.0
~~~



Connection refused가 이상해서 좀 찾아보니 xauth 관련된 권한 문제로 보였고 아래와 같은 run 옵션을 통해서 해결하였다.

~~~bash
// DISPLAY 환경변수 할당. 
-e DISPLAY=$DISPLAY

//x11관련폴더volume연결.
−v /tmp/.X11−unix:/tmp/.X11−unix

//Xauthority관련volume연결.
--volume="$HOME/.Xauthority:/root/.Xauthority:rw"
~~~

<br/>

최종적으로 사용한 run 옵션은 아래와 같다.

~~~bash
nvidia-docker run -it --rm --net=host -v ${HOME}/Desktop/pycharm:${HOME}/data -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --volume="$HOME/.Xauthority:/root/.Xauthority:rw" nvidia/cuda:10.0-cudnn7-runtime /bin/bash
~~~





p.s. tensorflow 에서 제공하는 image 를 사용해보았는데 pycharm 이 GUI 관련 오류로 실행이 안되었다.

아래의 커맨드로 `gedit` 을 설치하니 관련 파일들이 설치되어 정상 이용이 가능했다. (다음에 다시 정리하는걸로..)

~~~bash
apt−get update 
apt-get install gedit
~~~






> 네이버에서 이전한 포스트로 오래전 작성되어서 내용이 잘못되었을 수 있습니다. 오류나 잘못된 정보 전달시 댓글로 알려주세요!
{: .prompt-info }