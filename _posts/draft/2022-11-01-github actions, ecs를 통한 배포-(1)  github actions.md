---
title: github actions, ecs를 통한 배포 (1)  github actions
date: 2022-11-01 18:29:00 +0900
categories: [AWS, ECS]
tags: [aws, mlops, devops]     # TAG names should always be lowercase
# pin: true
# mermaid: true
# math: true
# toc: false     # Table Of Contents
# comments: false 
img_path: /
---

# github actions, ecs를 통한 배포 (1)  github actions



## github actions 란?



## github action workflow script 구조

github action은 .github/workflows/ 아래에 yml 파일로 저장할 수 있다. [github doc](https://docs.github.com/en/actions/using-workflows/about-workflows)에 설명되어있으며 기본 구조는 아래 템플릿과 같다.

~~~yaml
name: learn-github-actions
on: [push]
jobs:
  job1:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: ls -al
~~~

### name:

워크플로우의 이름을 지정한다. github action탭에 나타나는 이름이다.



### on:

해당 워크플로우에 대한 트리거 이벤트를 지정할 수 있다.

#### workflow_dispatch

post request를 이용해서 이벤트 발생가능 

#### workflow_call



### jobs:

워크플로우에서 돌아가는 일련의 job 묶음이다.



### job1:

job 이름을 직접 지정해서 설정 할 수 있다. 여기서는 `job1`으로 설정하였다.



###   runs-on:

github에서 호스팅하는 가상환경을 세팅한다.

위에서는 `ubuntu-latest`로 지정하여 최신버전의 우분투를 이용한다


###  steps:

job1 아래에서 돌아가는 step들을 정의한다.


### uses:

여기서는 `uses: actions/checkout@v3` 으로 되어있는데 `actions/checkout@v3 action`을 사용하겠다는 의미이다.

[actions](https://github.com/actions) 에 가보면 다양한 action들을 확인해볼 수 있다.

예를들어서 python을 해당 환경에서 이용하고 싶으면 다음과 같은 항목을 추가해주면 된다. ([참고](https://github.com/actions/setup-python))

~~~yaml
- uses: actions/setup-python@v4
  with:
    python-version: '3.10' 
~~~



### run:

run 키워드는 커맨드라인에서 실행시킬 커맨드를 지정할 수 있다.



## github CLI를 이용한 수동 트리거

~~~shell
gh workflow run .github/workflows/deploy.yml --ref integration -F environment=alpha
~~~

~~~
name: deploy_alpha
on:
  workflow_dispatch:
  push:
    branches: [integration]

jobs:
  build:
    runs-on: ubuntu-latest
    environment: alpha
    steps:
      - name: Checkout action
        uses: actions/checkout@v3

      - name: Configure aws credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.ACCESS_KEY }}
          aws-secret-access-key: ${{ secrets.SECRET_ACCESS_KEY }}
          aws-region: ap-northeast-2

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build, tag, and push
        uses: docker/build-push-action@v3
        env:
          ECR_REPOSITORY: ${{ steps.login-ecr.outputs.registry }}/name
        with:
          context: .
          push: true
          file: Dockerfile.main
          cache-from: ${{ secrets.ECR_REPOSITORY }}:latest
          secrets: |
            "USER_ID=${{ secrets.USER_ID }}"
            "USER_PWD=${{ secrets.USER_PWD }}"
          tags: ${{ secrets.ECR_REPOSITORY }}:latest

      - name: Notify slack on finish
        uses: rtCamp/action-slack-notify@v2
        if: always()
        env:
          SLACK_WEBHOOK: ${{ secrets.DEPLOY_SLACK_WEBHOOK }}
          SLACK_USERNAME: Github Action
          SLACK_COLOR: ${{ job.status }}
          SLACK_ICON: https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png
          SLACK_MESSAGE: ${{ job.status }}


~~~


# ECS

ecs 이용시 용량 공급자를 설정해줘야 한다. 클러스터 생성시 만들어진 autoscaling group을 용량 공급자로 지정해주고 클러스터 업데이트를 눌러서 추가 용량 공급자를 설정해준다.

서비스 개수를 원하는 개수로 맞추면 알아서 ec2 인스턴스를 늘려준다.

이때 task의 cpu, 메모리 제한을걸어줘야 인스턴스에서 용량이 부족해지면 scaling을 자동적으로 해준다.



## 작업 크기

하나의 작업 (task)가 돌아가기 위한 리밋. 예를들어서 4vcpu 8GB 인스턴스라면 1vcpu 2GB 제한의 task를 4개 돌릴 수 있다. 위에서 용량 공급자 설정이 되어있다면 서비스에서 작업 개수가 4를 넘어가면 새로운 인스턴스가 실행된다.



## 작업이 변경된 경우

클러스터 - 서비스에서 서비스 업데이트 버튼을 눌러서 작업 정의를 변경해서 업데이트를 진행한다.



## 실수로 ecs agent를 종료한 겨우 
[https://docs.aws.amazon.com/ko_kr/AmazonECS/latest/developerguide/ecs-agent-install.html](https://docs.aws.amazon.com/ko_kr/AmazonECS/latest/developerguide/ecs-agent-install.html)


## 서비스 타겟 개수에 따라서 실행이 안되는 문제가 발생

용량 공급자로 설정된 ASG가 예상대로 scaling 동작을 해주지 않음.

-> 클러스터와 서비스에 용량공급자가 제대로 설정 되어있나 확인할것,

- 추가적으로 용량 공급자를 여러개 설정하여 인스턴스별로 다른 타입의 인스턴스를 사용 가능하다


## ECS가 중단되는 현상

관리형 조정과 관리형 종료방지 기능을 켜주는것이 좋음. (scale in시 발새할 수 있는 충돌을 방지할 수 있음.)

task 에서 로그 구성 추가
![image-20221111110746372](assets/img/posts/image-20221111110746372.png)

## reference

### official

- [aws-actions](https://github.com/aws-actions)
- [GitHub Docs - workflows](https://docs.github.com/en/actions/using-workflows/about-workflows)
- [What is Amazon Elastic Container Service?](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html)
- [Amazon ECS Workshop](https://ecsworkshop.com)
- [CodeDeploy란 무엇인가요?](https://docs.aws.amazon.com/ko_kr/codedeploy/latest/userguide/welcome.html)

- [ECS 개발자 안내서](https://docs.aws.amazon.com/ko_kr/AmazonECS/latest/developerguide/Welcome.html)

- [ecs-refarch-batch-processing](https://github.com/aws-samples/ecs-refarch-batch-processing)

- [Docker- github cache](https://docs.docker.com/build/ci/github-actions/examples/#cache)



### blog

- [github-action 으로 ec2 에 배포하기](https://blog.bespinglobal.com/post/github-action-으로-ec2-에-배포하기/)

- [workflow_dispatch를 이용한 github action 수동 트리거](https://www.hahwul.com/2020/10/18/how-to-trigger-github-action-manually/)

- [GitHub Actions의 workflow_call로 워크플로우 재사용하기](https://blog.outsider.ne.kr/1591)

- [AWS로 배포하기 시리즈 - 1. Code Deploy 사용하기](https://jojoldu.tistory.com/281)

- [ECS로 서버 배포 및 자동화하기](https://blog.naver.com/sssang97/222626113440)

- [AWS ECS 살펴보기](https://boostbrothers.github.io/technology/2020/01/29/AWS-ECS-살펴보기/)

- [[소개] amazon-ecs란](https://tech.cloud.nongshim.co.kr/2021/08/30/소개-amazon-ecs란/)

- [How to add github secrets to env variable](https://github.com/docker/build-push-action/issues/390)

- [Sharing environment variables using Github Action secrets](https://andrei-calazans.com/posts/2021-06-23/passing-secrets-github-actions-docker)

- [GitHub Actions workflow를 수동으로 trigger하기(feat. inquirer.js)](https://fe-developers.kakaoent.com/2022/220929-workflow-dispatch-with-inquirer-js/)

- [GitHub Actions에서 도커 캐시를 적용해 이미지 빌드하기](https://fe-developers.kakaoent.com/2022/220414-docker-cache/)
- [AWS 배포자동화 실제로 구축하기④ - ECS, ELB, EC2 인스턴스 생성하기](https://blog.naver.com/developer501/222692679078)
