---
title: data pipeline (workflow scheduler)
date: 2022-12-21 22:46:00 +0900
categories: [mlops]
tags: [data-management]     # TAG names should always be lowercase
# pin: true
# mermaid: true
# math: true
# toc: false     # Table Of Contents
# comments: false 
img_path: /
---

# **data pipeline (workflow scheduler)**

ML 관련 프로세스 작업을 하다보니 주기적으로 데이터를 처리해아할 일이 발생하게 되었다. 예를 들어 한없이 쌓이는 데이터를 미리 학습용 데이터 구조로 변경시켜둔다면 나중에 학습할때 손쉽게 이용할 수 있다.

데이터 사이언스에서는 보통 이를 **ETL** (`Extract`, `Transform`, `Load`) (요즘은 순서가 바뀐 ELT도 많이 사용되고 있다.)라고 하는데 일반적으로 비정형 데이터 (로그 등)을 데이터 웨어하우스(`Redshift`, `BigQuery`, `Snowflake` 등)으로 쌓는 작업을 말한다.

하지만 ML/DL에서는 데이터의 구조가 비정형적이고 관계형 DB 보다는 문서형 DB에 맞다.(물론 경우에 따라 다르다.) 그리고 실제 학습에 사용할때는 db에 들어있는 것보다 파일로 저장되어있는것이 처리에 편한다. 



## scheduling jobs

그럼 기존의 ETL에 사용되던 툴들을 사용해볼까 하여 조사를 시작하였다. 기존에 사내에서 사용하는 airflow와 redshift의 조합은 맞지 않아서 여러 방법을 검토해보았다.

이를 위해서 데이터 파이프라인을 반복적으로나, 특정 조건에 맞춰서 동작시키는 것이 중요한데 가장먼저 생각난건 crontab을 이용한 방법이었다.

하지만 모니터링이 어렵고 수동으로 관리해야 하다보니 불편한점이 많다.

일단 규모가 크지 않아서 구성 리소스가 적게 들어가는 cron으로 관리해보고 추후에 파이프라인이 많아지고 관리가 힘들어지면 다른 툴을 도입하기로 하였다.   

그때를 대비하여 미리 조사한 내용들을 남겨본다.


- cron 무조건 주기적으로 반복, 많은 메모리를 사용하게 되는 경우 문제가 발생할 수 있음.
  - python-airflow (que가 있기는한데 사용법이 익숙하지 않아서 좀더 조사가 필요함)
  - crontab

- que - scheduling 큐에 작업들을 넣어서 순서대로 처리하기 때문에 비교적 메모리 이슈에서 자유로울 수 있음,
	- ruby-sideqic
	- python-rqscheduler



이외에도 다양한 툴들이 있는것을 확인하였다.

- python-luigi
- Hadoop jobs based(**Azkaban**, **Oozie**)
- aws step-functions
- aws lambda
  - 각 process당 최대 메모리 10G 제한 (cpu는 메모리 크기에 따라 자동 지정)
  - 최대 런타임이 15분으로 제한적
- aws sagemaker & glue job
- github action



이 외에도 많은 툴들이 있지만 나중에 MLOps관련 포스팅을 할때 좀더 추가해볼까 한다.

리서치 AI를 하다가 production AI를 하니 확실이 부족했던 부분들을 많이 느끼고 있다. (리서치도 부족하지만..)
