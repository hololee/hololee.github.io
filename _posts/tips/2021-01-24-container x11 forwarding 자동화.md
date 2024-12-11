---
title: container x11 forwarding 자동화
date: 2021-01-24 01:45:00 +0900
categories: [tips]
tags: [ubuntu, pycharm, ssh, x11_forwarding, os]     # TAG names should always be lowercase
img_path: /
---

지난번에 ssh 를 통해 접속한  bash  session의 container 안에서 x11forwading 을 통해서 `사용자PC` - `server` - `docker container`구조로 x11 forwarding을 하는 방법을 작성해두었다.



문제는 docker run 옵션으로 많은 부분을 타이핑 해두어야 하는점인데 좀더 편하게 이용하고자 아래처럼 bash shell script 를 작성하였다.

~~~bash
#!/bin/bash

# Put path info =========================================================

dockerTargetPath="/shared_path"
pycharmPath=${dockerTargetPath}"/_PYCHARM/pycharm-2020.3.1/bin"

# =======================================================================

imageList=$(docker images |tail -n +2 |awk '{print $1":"$2}')
dockerImages=(${imageList})

# Read image name from user.
echo "===== Docker image list ====="
echo ""
count=1
for i in ${dockerImages[@]}; do
    echo "${count}. [ ${i} ]"
    ((count=${count}+1))
done
echo ""
echo "============================="
echo ""
echo -e "Plase select docker-image number (1-${#dockerImages[@]}) :"
read sn
((sn=${sn}-1))
echo "Image name : ${dockerImages[${sn}]}"
imageName=${dockerImages[${sn}]}
echo ""
echo ""

# Allow display.
echo `xhost +`

# Run docker and set options.
echo "Now start pycharm..."
echo `nvidia-docker run -it --rm --net=host -v ${dockerTargetPath}:${dockerTargetPath} -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --volume="$HOME/.Xauthority:/root/.Xauthority:rw" ${imageName} /bin/bash -c "cd ${pycharmPath}; sh pycharm.sh"`
~~~



동작하면 아래처럼 결과가 나온다.

~~~bash
$ ./show.sh
===== Docker image list =====

1. [ hololee/tensorflow:1.15.5 ]
2. [ hololee/tensorflow:2.4.0 ]
3. [ hololee/pytorch:1.7.0 ]
4. [ pycharm_helpers:PY-203.6682.86 ]
5. [ hololee/pytorch:<none> ]
6. [ pycharm_helpers:PY-202.8194.22 ]
7. [ hololee/tensorflow:<none> ]
8. [ hololee/tensorflow:<none> ]
9. [ busybox:latest ]
10. [ hololee/pytorch:<none> ]

=============================

Plase select docker-image number (1-10) :
3
Image name : hololee/pytorch:1.7.0


Now start pycharm...

~~~



p.s.1 스크립트 내부에서 docker run을 하였더니 쉘이 멈췄다. 현재는 copy 하는 정도에서 하고 추후 방법을 찾아봐야 할듯.

p.s.2 단순히 커멘드 실행 으로 변경했더니 실행이 잘 되었다.. 코드 참고..




> 네이버에서 이전한 포스트로 오래전 작성되어서 내용이 잘못되었을 수 있습니다. 오류나 잘못된 정보 전달시 댓글로 알려주세요!
{: .prompt-info }