---
title: WOL(Wake on lan) 우분투, 윈도우 자유롭게 부팅하기
date: 2021-02-14 15:10:00 +0900
categories: [tips]
tags: [ubuntu, wol, os]     # TAG names should always be lowercase
img_path: /
---

연구실에서 사용하는 서버는 항시 가동이기에 크게 상관 없지만 집에서 별도로 구성한 서버는 전력 소모등을 이유로 항상 켜두지 않는다.

이번에 서버 구성을 바꾸고 os 를 재설치 하면서 원격 부팅을 위한 wol 구성을 다시 하였다.



#### 우분투 wol 설정

우선 필요한 페키지를 설치한다.

~~~bash
$ sudo apt-get install net-tools wakeonlan ethtool
~~~



#####  네트워크 인터페이스 확인

 ifconfig를 이용하여 사용자 네트워크 인터페이스를 확인한다.

~~~
$ ifconfig
~~~

![img](assets/img/posts/Screenshot_from_2021-02-14_14-45-08.png)

앞 부분의 eno1 이 인터페이스 이름이다.

간혹 docker 등에 의해서 다른 카드이름이 섞여 있을 수 있으니 실제 사용하는 인터페이스 이름을 적어둔다.



##### wol 설정 확인

ethtool을 통해 wol 설정을 확인해 본다.

![img](assets/img/posts/Screenshot_from_2021-02-14_14-46-21.png)



아래와 같이 wol 을 g 로 잡아주면 동작한다.

~~~bash
$ ethtool -s {card name} wol g
~~~

![img](assets/img/posts/Screenshot_from_2021-02-14_14-47-22.png)



##### 재부팅 적용

설정 파일을 변경하여 재부팅 시에도 wol이 적용될 수 있도록 한다.

~~~bash
$ sudo vi /etc/network/interfaces
~~~



우분투는 잘 동작한다!

윈도우는 설정 법이 간단하고 많이 설명 되어있다. 

iptime 공유기에서 제공하는 wol 기능을 이용해서 키고 끈다. 



#### 우분투, 윈도우 바꿔가며 부팅하기

기본적으로 윈도우 설치 후 리눅스를 설치하면 GRUB이 부트로더로서 설치되고 이를 통해서 윈도우나 리눅스로 선택해서 부팅하게 된다.

문제는 WOL을 사용할 경우에 <u>외부에서 OS를 선택할 방법이 없기 때문에</u> 기본 지정된 os가 아닌 다른 os를 사용하고자 할때 매우 곤란하다.



이를 재부팅할때 기본 선택을 변경하는 방법을 통해 해결 할 수 있다.



##### grub default 변경

grub 설정 파일을 아래와 같이 열어본다.

~~~bash
$ sudo vi /etc/default/grub
~~~



파일을 보면 `GRUB_DEFAULT=0` 으로 되어있다.

 `GRUB_DEFAULT=saved` 를 지정하면 아래처럼 기본 부팅 순서를 선택할 수 있다.

> If you set this to ‘saved’, then the default menu entry will be that saved by ‘GRUB_SAVEDEFAULT’ or grub-set-default. This relies on the environment block, which may not be available in all situations (see Environment block). [Link](https://www.gnu.org/software/grub/manual/grub/grub.html#Simple-configuration)



수정한 후 파일에 나와있는것처럼 grub 설정을 업데이트 한다.

~~~bash
$ sudo update-grub
~~~



##### 기본 실행 순서 설정

다음으로 기본으로 선택되는 순서를 정한다. (여기서는 우분투 0번을 선택하였다.)

~~~bash
$ sudo grub-set-default 0
~~~



##### 선택 순서를 지정하고 재부팅하기

여기 까지 하면 준비과정은 끝났다.

사용을 하려면 아래처럼 입력해서 재부팅할 선택 순서를 정해주고 재부팅을 한다.

~~~bash
$ sudo grub-reboot 2
$ sudo reboot now
~~~

>grub에서 보여지는 부팅리스트의 순서를 입력해주면 된다. 나는 윈도우가 3번째라 2를 선택했다. 
{: .prompt-tip }



윈도우로 재부팅 성공. 

다음번부터 윈도우로 부팅하고 싶으면 우분투로 켜졌을때 위와 같이 작성하면 된다.



##### 부팅 자동화 하기

`.bashrc`에 함수를 추가해둬서 위의 과정을 단순화 할 수 있다. 

(`.bashrc` 는 bash shell이 처음 동작할때 실행되는 쉘 스크립트다.  `/home/{$user_name}/.bashrc` 참고)

나는 아래의 함수를 작성하였다.

~~~bash
# reboot window.
function bootWin(){
	echo `sudo grub-reboot 2`
	echo `sudo reboot now`
}
~~~

~~~bash
# update .bashrc
$ source ~/.bashrc
~~~





> 네이버에서 이전한 포스트로 오래전 작성되어서 내용이 잘못되었을 수 있습니다. 오류나 잘못된 정보 전달시 댓글로 알려주세요!
{: .prompt-info }
