---
title: Production에서의 ML과 MLOps tools 🚀
date: 2023-01-28 12:27:00 +0900
categories: [mlops]
tags: [tools]     # TAG names should always be lowercase
# pin: true
# mermaid: true
# math: true
# toc: false     # Table Of Contents
# comments: false 
img_path: /
---

> 이 포스트는 주관적인 견해로 작성하였습니다. 더 좋은 정보가 있다면 댓글로 알려주세요! 
{: .prompt-warning }

# Production의 ML과 Research ML의 차이

연구실에서 연구 목적을 위한 모델의 학습과 회사에서 프로덕트 적용을 위한 모델 학습을 하면서 많은 차이점을 느끼게 되어서 정리겸 작성을 해본다.

일반적으로 연구실에서의 학습은 정해진 데이터셋을 가지고 baseline 모델들보다 좋은 성능을 보이고 이를 근거를 들어서 SOTA(State-Of-The-Art)에 준하는 성능을 입증하면 좋은 성과로 이어지고는 하는것 같다.

반대로 프로덕션의 경우에는 학습 데이터셋은 정해져있지 않고 계속 변한다.
개인정보와 관련된 데이터는 무려 삭제되기도 한다! 이로인해 구축해둔 데이터 셋이 망가지는 경우도 발생하기 때문에 고려할 사항이 많아진다. 또한 이러한 방대한 데이터를 처리하기 위한 파이프라인의 필요성도 대두된다.

모델에 있어서 작은 차이의 성능 향상보다는 빠른 개발과 빠른 속도, 성능대비 적은 추론 비용을 가진 모델을 선호하게 되는것 같다.

<u>개인적으로 느낀점</u>을 정리를 해보면,

|| **Research ML** |
|-:|-|
|**데이터**|주로 다양하고 검증된 정해진 데이터셋에 대해 학습 및 테스트 |
|**파이프라인**|그때 그때 코드를 정리하거나 즉석해서 생성|
|**학습**|정해진 데이터셋에 대한 성능 검증을 위한 단발성 학습|
|**성능**|정해진 테스트셋에 대한 나은 성능|
|**모델**|SOTA 모델을 개선하거나 다양한 새로운 방식들을 개선하여 성능 향상 |
|**규모**|주제에 따라 다르지만 성능이 목적이라면 규모에 영향을 크게 받지 않음 |
|**배포**|모델과 학습 코드만 개발하거나 배포에 크게 신경쓰지 않음 (요즘은 많이 바뀌는듯)|

|| **Production ML** |
|-:|-|
|**데이터**| 목적 데이터가 정해져있지만 데이터가 방대하고 변동성이 있음 |
|**파이프라인**|동일한 환경에서 발생하는 데이터가 많아서 데이터 및 기타 파이프라인 구축 필요|
|**학습**| 데이터 변화에 따른 지속적 학습|
|**성능**|정량적 성능뿐만 아니라 정성적 성능도 중요함|
|**모델**| 기존에 구축된 모델을 빠르게 개선 및 적용|
|**규모**|비용적 이유로 실제 규모가 제한적|
|**배포**|배포가 매우 중요!|

<br/>

# 도움을 줄 수 있는 도구들
앞서 말한것처럼 Production에서의 AI 개발은 쉽지 않았고 이를 도와줄 툴들을 찾아보게 되었다.
아래의 리스트들은 주관적인 측면에서 분류해두었고 검색으로도 잘 안나오는 경우가 있어서(다른 주제가 검색된다거나) 링크도 같이 넣어두었다.

각 툴들에 대한 설명은 따로 추가하지 않았다. (직접 접해보니 샘플이라도 직접 사용해보는게 더 좋은 경험이었다.)


## Infra

전체적인 인프라 구축에 도움을 줄 수 있는 툴들이다. 하지만 규모가 커서 리소스나 비용적인 이유로 바로 적용이 어려울 수 있다.

[k8s](https://kubernetes.io/ko/)-[kubeflow](https://www.kubeflow.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: ML workflow를 관리하기 위한 통합 플랫폼.

[AWS sagemaker](https://aws.amazon.com/ko/sagemaker/) <img alt="Html" src ="https://img.shields.io/badge/-pricing-red"/>
: AWS에서 제공하는 ML workflow 통합 플랫폼.


## Parameter, Configurations

[hydra](https://hydra.cc) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: configuration 관리 툴.

[optuna](https://optuna.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 하이퍼파라미터를 효과적으로 튜닝할 수 있는 툴.


## CI/CD

[github action](https://docs.github.com/ko/actions) <img alt="Html" src ="https://img.shields.io/badge/-pricing-red"/>
: 정말 많이 사용하는 툴.

[Jenkins](https://www.jenkins.io) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 유명한 CI 툴.

[CML](https://cml.dev) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 머신러닝을 위한 CI/CD 툴.


## Pipeline

crontab
: 리눅스 OS 자체의 crontab도 휼륭한 툴로서 사용할 수 있음.

[Apache airflow](https://airflow.apache.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: python 기반의 스케줄링 툴.

[Apache kafka](https://kafka.apache.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 이벤트 스트리밍 관리 툴.

[Celery](https://docs.celeryq.dev/en/stable/index.html) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 메시지 비동기 잡 큐.

[luigi](https://github.com/spotify/luigi) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: python 기반의 배치 잡 큐.

[argo](https://argoproj.github.io) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Open source tools for Kubernetes to run workflows, manage clusters, and do GitOps right.

[Apache spark](https://spark.apache.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Engine for executing data engineering, data science, and machine learning on single-node machines or clusters.

[Apache hadoop](https://hadoop.apache.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Distributed processing of large data sets across clusters of computers using simple programming models.

[flyte](https://flyte.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: ML 워크플로우 툴.

[python-rqscheduler](https://github.com/rq/rq-scheduler) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: RQ Scheduler is a small package that adds job scheduling capabilities to RQ, a Redis based Python queuing library.


## Model Management & Monitoring

[tensorboard](https://www.tensorflow.org/tensorboard?hl=ko) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 텐서플로우 시각화 툴킷.

[mlflow](https://mlflow.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 머신러닝 라이프사이클 관리 툴.

[metaflow](https://metaflow.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Netflix의 ML 워크플로우 툴.

[Weight and Bias](https://wandb.ai/site) <img alt="Html" src ="https://img.shields.io/badge/-pricing-red"/>
: Build better models faster with experiment tracking, dataset versioning, and model management.

[comet ml](https://www.comet.com/site/) <img alt="Html" src ="https://img.shields.io/badge/-pricing-red"/>
: Manage, visualize, and optimize models.

[Neptune ai](https://neptune.ai) <img alt="Html" src ="https://img.shields.io/badge/-pricing-red"/>
: Log, organize, compare, register, and share all your ML model metadata in a single place.

[dvclive](https://dvc.org/doc/dvclive) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: dvc 기반의 트레킹 툴.

[ZenML](https://zenml.io/home) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: The Open Source MLOps Framework for Unifying Your ML Stack.


## Training

[skypilot](https://github.com/skypilot-org/skypilot) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Framework for easily and cost effectively running ML workloads on any cloud.

[petals](https://petals.ml) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 토렌트 스타일의 NLP 학습 툴.


## Data Management

[dvc](https://dvc.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 데이터 버전 컨트롤.

[Pachyderm](https://www.pachyderm.com) <img alt="Html" src ="https://img.shields.io/badge/-pricing-red"/>
: 데이터 파이프라인 툴.


## Data Visialize

[streamlit](https://streamlit.io) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Streamlit turns data scripts into shareable web apps in minutes.

[gradio](https://gradio.app) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Gradio is the fastest way to demo your machine learning model with a friendly web interface.

[dash](https://github.com/plotly/dash) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Dash is the most downloaded, trusted Python framework for building ML & data science web apps.

[metabase](https://github.com/metabase/metabase) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Metabase is the easy, open-source way for everyone in your company to ask questions and learn from data.

[pynecone](https://pynecone.io) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Build web apps in minutes, Deploy with a single command.


## Model Serving

[flask](https://palletsprojects.com/p/flask/) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Flask is a lightweight WSGI web application framework.

[fastapi](https://fastapi.tiangolo.com/ko/) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: FastAPI는 현대적이고, 빠르며(고성능), 파이썬 표준 타입 힌트에 기초한 Python3.6+의 API를 빌드하기 위한 웹 프레임워크입니다.

[heroku](https://www.heroku.com) <img alt="Html" src ="https://img.shields.io/badge/-pricing-red"/>
: 헤로쿠는 웹 애플리케이션 배치 모델로 사용되는 여러 프로그래밍 언어를 지원하는 클라우드 PaaS.

[bentoml](https://docs.bentoml.org/en/latest/) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: BentoML makes it easy to create ML-powered prediction services that are ready to deploy and scale.

[seldon-core](https://www.seldon.io/solutions/open-source-projects/core) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Seldon Core is the open-source framework for easily and quickly deploying models and experiments at scale.


## Distributed Computing

[Ray](https://www.ray.io) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Makes it easy to scale AI and Python workloads

## ETC

[locust](https://locust.io) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: 로드 테스트

[fabric](https://www.fabfile.org) <img alt="Html" src ="https://img.shields.io/badge/-open--source-blue"/>
: Execute shell commands remotely

[infisical](https://infisical.com) <img alt="Html" src ="https://img.shields.io/badge/-pricing-red"/>
: SecretOps


# 툴의 활용

앞서 나열한 툴들을 무조건 좋고 거대한 툴을 사용하는것이 아닌 진행중인 프로젝트에 맞춰서 적절한 도구를 선택하는것이 매우 중요한것 같다.

데이터 파이프라인을 예로들면 처음부터 하둡, 스파크 등의 도입을 고려할것 없이 작은 DB 하나와 cronjob만으로도 매우 좋은 파이프라인을 만들 수 있을것 같다.
