---
title: MLOps-Basics
date: 2021-12-07 21:00:00 +0900
categories: [ mlops]
tags: [ai, mlops, devops]     # TAG names should always be lowercase
img_path: /
---

>이 포스트는 작성자의 허락을 받아 번역하고 게시하였습니다. 내용과 관련한 질문은 [작성자의 repo](https://github.com/graviraja/MLOps-Basics)를 이용해주세요.
{: .prompt-info}

# MLOps-베이직

 > There is nothing magic about magic. The magician merely understands something simple which doesn’t appear to be simple or natural to the untrained audience. Once you learn how to hold a card while making your hand look empty, you only need practice before you, too, can “do magic.” – Jeffrey Friedl, 서적 Mastering Regular Expressions

**Note: 제안, 수정 또는 피드백이 있는 경우 Issue를 올려주세요.**   

- *원 작성자의 허락을 받아서 리포를 번역했습니다. 오역이 있을 수도 있으니 `main` branch나 [원본 리포](https://github.com/graviraja/MLOps-Basics)를 참고해주세요.*

MLOps-Basics 시리즈의 목표는 모델의 `구축(building)`, `모니터링(monitoring)`, `구성(configurations)`, `테스트(testing)`, `패키징(packaging)`, `배포(deployment)`, `CI/CD`와 같은 MLOps의 기본을 이해하는 것입니다.

[![pl](assets/img/posts/summary.png)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/summary.png)

## 0주차: Project 준비

<img src="assets/img/posts/v1.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-project-setup-part1)를 참고해주세요.

이 프로젝트에서는 간단한 classification 문제를 다루고 있습니다. 이번 주차에서는 아래의 질문에 답할 수 있는 범위를 다루게 됩니다:

- `데이터는 어떻게 구할까?`
- `데이터를 어떻게 처리해야 할끼?`
- `데이터 로더(dataloader)를 어떻게 정의 해야 할까?`
- `모델은 어떻게 정의할까?`
- `모델을 어떻게 학습할까?`
- `추론은 어떻게 해야하나?`

[![pl](assets/img/posts/pl.jpeg)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/pl.jpeg)

이 프로젝트를 위해서 아래의 내용을(tech stack)숙지하고 있어야 합니다:

- [Huggingface Datasets](https://github.com/huggingface/datasets)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/)


## 1주차: 모델 모니터링 - 가중치(Weights)와 바이어스(Biases)

<img src="assets/img/posts/v1.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-wandb-integration)를 참고해주세요.   

하이퍼 파라미터(hyper-parameters)를 수정하고 성능 테스트를 위해서 다른 모델을 사용하는 것 그리고 모델과 입력 데이터의 관계를 살펴보는 것과 같이 모든 상황을 추적하는 것은 더 나은 모델을 설계할 수 있도록 합니다.

이번 주차에서는 아래의 질문에 답할 수 있는 범위를 다루게 됩니다:

- `가중치(W)와 바이어스(B)로 기본적인 로깅(logging)을 어떻게 구성할까?`
- `어떻게 매트릭스를 연산하고 W와 B로서 기록할 수 있을까?`
- `W와 B를 어떻게 그래프로 나타낼 수 있을까?`
- `어떻게 데이터를 W와 B에 녹여낼 수 있을까?`

[![wannb](assets/img/posts/wandb.png)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/wandb.png)

이 프로젝트를 위해서 아래의 내용을(tech stack)숙지하고 있어야 합니다:

- [Weights and Biases](https://wandb.ai/site)
- [torchmetrics](https://torchmetrics.readthedocs.io/)

References:

- [Tutorial on Pytorch Lightning + Weights & Bias](https://www.youtube.com/watch?v=hUXQm46TAKc)
- [WandB Documentation](https://docs.wandb.ai/)

## 2주차: 구성(Configurations) - Hydra

<img src="assets/img/posts/v1.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-hydra-config)를 참고해주세요.   

구성 관리(Configuration management)는 복잡한 소프트웨어 시스템을 관리하는 데 필요합니다. Configuration management가 부족하면 안정성, 가동 시간, 시스템 확장 기능에 심각한 문제가 발생할 수 있습니다.

이번 주차에서는 아래와 같은 범위를 다루게 됩니다:

- `Hydra의 기본`
- `구성의 재정의(Overridding configurations)`
- `다양한 파일에 configuration을 분할하는 방법`
- `변수 인터폴레이션(Variable Interpolation)`
- `다른 파라미터 조합으로 어떻게 모델을 학슬할까?`

[![hydra](assets/img/posts/hydra.png)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/hydra.png)

이 프로젝트를 위해서 아래의 내용을(tech stack)숙지하고 있어야 합니다:

- [Hydra](https://hydra.cc/)

References

- [Hydra Documentation](https://hydra.cc/docs/intro)
- [Simone Tutorial on Hydra](https://www.sscardapane.it/tutorials/hydra-tutorial/#executing-multiple-runs)

## Week 3: Data Version Control - DVC

<img src="assets/img/posts/v1.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-dvc)를 참고해주세요.

전형적인 버전 컨트롤 시스템은 큰 파일들을 다룰 수 있도록 설계되어 있지 않습니다. 따라서 이러한 시스템은 기록을 복제하고 저장하는 것을 실용적이지 못하게 만듭니다. 머신러닝에서는 이러한 일이 다반사 입니다.

이번 주차에서는 아래와 같은 범위를 다루게 됩니다:

- `DVC의 기본`
- `DVC 초기화`
- `리모트 저장소를 구성하는 방법`
- `리모트 저장소에 모델을 저장하는 방법`
- `모델의 버전 관리`

[![dvc](assets/img/posts/dvc.png)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/dvc.png)

이 프로젝트를 위해서 아래의 내용을(tech stack)숙지하고 있어야 합니다:

- [DVC](https://dvc.org/)

References

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorial on Versioning data](https://www.youtube.com/watch?v=kLKBcPonMYw)

## 4주차: 모델 패킹(packing) - ONNX

<img src="assets/img/posts/v1-20220914083524022.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-onnx)를 참고해주세요.

왜 모델 패킹이 필요할까요? 모델은 다양한 머신러닝 프레임워크(sklearn, tensorflow, pytorch, 기타 등등)를 통해서 만들어 질 수 있습니다. 이러한 모델들을 모바일, 웹, 라즈베리파이와 같은 다양한 환경에 배포하고 싶고 파이토치로 학습하고 텐서플로우로 추론하는 것과 같이 다양한 프레임워크를 이용하고 싶을 수도 있습니다.   
이처럼 AI 개발자가 다양한 프레임워크, 도구, 런타임 및 컴파일러와 함께 모델을 사용할 수 있도록 하는 공통 파일 포멧은 많은 도움이 될 수 있습니다.

커뮤니티 프로젝트인 `ONNX`를 이용하면 앞서 언급한 목적들을 달성할 수 있습니다.

이번 주차에서는 아래와 같은 범위를 다루게 됩니다:

- `ONNX란?`

- `학습된 모델을 ONNX 포멧으로 어떻게 변환할까?`

- `ONNX Runtime이란?`

- `변환된 ONNX 모델을 ONNX Runtime에서 구동하는 방법은?`

- `비교`

[![ONNX](assets/img/posts/onnx.jpeg)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/onnx.jpeg)

이 프로젝트를 위해서 아래의 내용을(tech stack)숙지하고 있어야 합니다:

- [ONNX](https://onnx.ai/)
- [ONNXRuntime](https://www.onnxruntime.ai/)

References

- [Abhishek Thakur tutorial on onnx model conversion](https://www.youtube.com/watch?v=7nutT3Aacyw)
- [Pytorch Lightning documentation on onnx conversion](https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html)
- [Huggingface Blog on ONNXRuntime](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)
- [Piotr Blog on onnx conversion](https://tugot17.github.io/data-science-blog/onnx/tutorial/2020/09/21/Exporting-lightning-model-to-onnx.html)


## 5주차: 모델 패킹(packaging) - 도커(docker)

<img src="assets/img/posts/v1.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-docker)를 참고해주세요.

모델 패킹이 왜 필요할까요? 어플리케이션을 다른 누군가에게 공유해줘야 할 수도 있고, 이러한 경우 많은 상황에서 어플리케이션은 의존성 문제나 OS관련 문제로 돌아가지 않습니다. 그래서 많은 경우 다음과 같은 말을 남겨둡니다. `이 프로젝트는 내 OO랩탑, OO시스템에서 테스트 되었습니다.`

따라서 어플리케이션을 실행하기 위해서는 실제 동작했던 환경과 동일한 환경을 구성해야 합니다. 결국 동일한 환경을 구성하기 위해서는 수동으로 많은 것들을 설정 해야하고 많은 컴포넌트를 설치해야 합니다(가끔은 이러한 환경 문제가 더 안풀리기도 하죠ㅠ).

이러한 한계를 극복할 수 있는 방법을 컨테이너(Containers)기술 이라고 합니다.

어플리케이션을 컨테이너화/패키징 함으로써 어떠한 클라우드 플랫폼에서도 어플리케이션을 실행할 수 있고 관리형 서비스(managed services), 오토스케일링(autoscaling), 안정성과 같은 다양한 이점을 얻을 수 있습니다.
이러한 작업을 위해서 가장 많이 찾는 툴이 바로 Docker🛳 입니다. 

이번 주차에서는 아래와 같은 범위를 다루게 됩니다:

- `FastAPI wrapper`
- `Docker 기본`
- `Docker Container 빌드하기`
- `Docker Compose`

[![Docker](assets/img/posts/docker_flow.png)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/docker_flow.png)

References

- [Analytics vidhya blog](https://www.analyticsvidhya.com/blog/2021/06/a-hands-on-guide-to-containerized-your-machine-learning-workflow-with-docker/)


## 6주차: CI/CD - GitHub Actions

<img src="assets/img/posts/v1-20220914083524022.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-github-actions)를 참고해주세요.

CI(Continuous Integration)/CD(Continuous Delivery)는 반복적인 코드의 변화를 지속적으로 빌드, 테스트, 배포할 수 있는 일련의 과정을 말합니다.

이러한 반복적인 프로세스는 버그로 가득찬 코드나 잘못된 이전 버전으로부터 새로운 코드를 개발하는 것을 방지하는데 도움을 줄 수 있습니다. 또한 이 방법은 개발에서 배포에 이르기까지 사람이 직접 작업하는 과정을 줄이거나 없도록 노력합니다.

이번 주차에서는 아래와 같은 범위를 다루게 됩니다:

  `GitHub Actions 기본`
- `첫번째 GitHub Action`
- `Google Service Account 생성`
- `Service account 엑세스 권한 부여`
- `Google Service account를 사용하기 위한 DVC 설정`
- `Github Action 설정`

[![Docker](assets/img/posts/basic_flow.png)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/basic_flow.png)

References

- [Configuring service account](https://dvc.org/doc/user-guide/setup-google-drive-remote)
- [Github actions](https://docs.github.com/en/actions/quickstart)


## 7주차: Container Registry - AWS ECR

<img src="assets/img/posts/v1-20220914083524022.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-container-registry)를 참고해주세요.

컨테이너 레지스트리(container registry)는 컨테이너 이미지를 보관하기 위한 저장소 입니다. 컨테이너 이미지는 어플리케이션을 하나의 인스턴스에서 구동하기 위해서 여러 레이어로 이루어진 파일입니다. 동일한 저장소 위치에서 모든 이미지를 관리하면 사용자가 필요할때 커밋(commit)이나 풀(pull)할 수 있으며 이미지를 식별할 수도 있습니다.

Amazon Simple Storage Service (S3)는 인터넷을 이용하는 클라우드 스토리지 서비스 입니다. 이 서비스는 여러 지역에 걸쳐서 대용량의 저비용 스토리지 프로비저닝을 위해서 설계되어 있습니다.

이번 주차에서는 아래와 같은 범위를 다루게 됩니다:

- `S3 기본`
- `프로그래밍 방식으로 S3에 접근하는 방법`
- `AWS S3를 DVC에서 원격 저장소로 구성하는 방법`
- `ECR 기본`
- `GitHub Actions에서 S3, ECR을 사용하도록 설정하는 방법`

[![Docker](assets/img/posts/ecr_flow.png)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/ecr_flow.png)


## 8주차: 서버리스(Serverless) 배포 - AWS Lambda

<img src="assets/img/posts/v1-20220914083524022.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-serverless)를 참고해주세요.

서버리스 아키텍처는 인프라를 별도로 관리하지 않아도 어플리케이션과 서비스를 구축하고 실행할 수 있습니다. 어플리케이션은 서버에서 돌아가지만 모든 서버의 관리는 AWS와 같은 서드 파티 서버 관리자에 의해서 이루어집니다. 어플리케이션을 유지보수 하기 위해서 프로비저닝, 스케일링 및 서버관리가 필요하지 않습니다(serverless). 서버리스 아키텍처를 사용하므로써 개발자는 핵심 프로덕트에 더 집중할 수 있고 클라우드(cloud)나 온프레미스(on-premises)의 관리, 동작에 신경쓰지 않아도 됩니다.   

이번 주차에서는 아래와 같은 범위를 다루게 됩니다:

- `Serverless 기본`
- `AWS Lambda 기본`
- `Triggering Lambda with API Gateway`
- `Lambda를 이용한 Container 배포`
- `Github Actions을 이용한 Lambda 자동 배포`

[![Docker](assets/img/posts/lambda_flow.png)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/lambda_flow.png)


## 9주차: 예측 모니터링 - Kibana

<img src="assets/img/posts/v1-20220914083524022.svg"/>

자세한 내용은 [블로그 포스트](https://www.ravirajag.dev/blog/mlops-monitoring)를 참고해주세요.


시스템을 모니터링하면 시스템이 잘 돌아가고 있다는 것을 확신할 수 있고, 시스템에 오류가 발생하는 경우 근본 원인을 진단할때 적절한 컨텍스트(context)를 빠르게 제공할 수 있습니다.

모델의 학습과 추론시에 우리가 모니터링 하고자 하는 것은 다릅니다. 학습 과정에서는 loss가 줄어드는지 확인해야 하고 과대적합(overfitting)과 같이 학습을 방해하는 요소들이 발생하지 않는지 고려해야 합니다. 

하지만 모델의 추론시에는 모델의 출력이 정확한 출력을 하고 있는것인지 확실할 수 있어야 합니다.

모델이 유용한 추론 결과를 도출하지 못하는 것은 다음과 같은 여러 이유가 있습니다:

- 시간이 흐르면서 데이터의 분포가 변화하고 사용한 모델이 너무 오래된 경우.
- 학습에는 사용되지 않았던 데이터(edge cases)가 추론시에 이용되는 경우(이러한 경우 모델의 성능이 저하되거나 오류가 발생할 수 있습니다.).
- 모델이 배포시에 설정이 잘못되는 경우(매우 빈번하게 일어납니다.).

위와 같은 상황에서도 서비스 관점에서는 `성공적인` 예측을 한다고 볼 수 있습니다. 그러나 그 예측 결과는 의미가 없습니다. 머신러닝을 모니터링하면 이와 같은 상황들을 감지하고 관리하는데 도움이 될 수 있습니다(예를 들면 모니터링 결과에 따라서 모델을 재학습하고 배포하는 파이프라인을 가동할 수 있습니다.).

이번 주차에서는 아래와 같은 범위를 다루게 됩니다:

- `Cloudwatch Logs 기본`
- `Elastic Search Cluster 생성`
- `Elastic Search로 Cloudwatch Logs를 설정`
- `Kibana에서 Index Patterns 생성`
- `Kibana Visualisations 생성`
- `Kibana Dashboard 생성`

[![Docker](assets/img/posts/kibana_flow.png)](https://github.com/hololee/MLOps-Basics/blob/translate/korean/images/kibana_flow.png)