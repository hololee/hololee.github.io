---
title: MLOps-Basics week0 프로젝트 셋업
date: 2021-12-19 21:00:00 +0900
categories: [AI, MLOps]
tags: [ai, mlops, devops]     # TAG names should always be lowercase
img_path: /
---

**Note: 이 프로젝트의 목적은 모델의 SOTA 달성이 아닌 라이브러리들을 알아보고 배우는데 목적을 가지고 있습니다. **

# Requirements:

이 프로젝트에서는 Python 3.8 을 사용합니다.

다음 커멘드를 이용해서 가상 환경을 생성합니다:

```bash
conda create --name project-setup python=3.8
conda activate project-setup
```

requirements를 설치합니다:

```bash
pip install -r requirements.txt
```

# 실행

### 학습

requirements를 설치한 후에 모델을 학습을 간단하게 실행할 수 있습니다:

```bash
python train.py
```

### 추론

학습이 끝난 후 모델 checkpoint path를 업데이트하고 아래처럼 실행합니다.

```bash
python inference.py
```

### notebooks 실행하기

작가는 notebook을 실행하기 위해서  [Jupyter lab](https://jupyter.org/install)을 이용합니다. 

virtualenv를 사용하기때문에,  `jupyter lab` 커멘드를 이용할때 virtualenv를 사용할 수도 안할 수도 있습니다.

virutalenv를 사용하기 위해서 `jupyter lab`실행전 아래의 커맨드를 입력합니다.

```bash
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```
<br/>
<br/>
<br/>

# MLOps 베이직 [Week 0]: Project 설정
<br/>

## 🎬 강의 시작    
이 강의의 목표는 MLOps의 기본적인 요소들(eg. 모델 빌드, 모니터링, 설정, 테스트, 페키징, 배포, CI/CD)을 이해하는 것 입니다. 첫번째로 프로젝트를 설정해봅시다. 저자는 NLP에 관심이 많아서 NLP모델 위주로 설명이 진행되지만 기본적인 절차는 모두 비슷하게 진행됩니다. 여기서는 간단한 classification task를 가지고 진행합니다.

이 글에서는 아래와 같은 질문들을 다뤄봅니다.

- <b>`데이터를 얻는 방법은 무엇인가?`</b>
- <b>`데이터를 어떻게 처리할 것인가?`</b>
- <b>`dataloader를 정의하는 방법은 무엇인가?`</b>
- <b>`모델은 어떻게 선언하는 것인가?`</b>
- <b>`모델을 어떻게 학습하는가?`</b>
- <b>`모델 추론을 어떻게 하는 것인가?`</b>

<i>노트: 기본적인 머신러닝에 대한 이해가 필요합니다.</i>   
<br/>
<br/>

## 🛠 딥러닝 라이브러리   
딥러닝 프로젝트를 개발하기 위해서 아래와 같은 다양한 라이브러리를 활용할 수 있습니다.
- <b>[Tensorflow](https://www.tensorflow.org/)</b>
- <b>[Tensorflow Lite](https://www.tensorflow.org/lite)</b>
- <b>[Pytorch](https://pytorch.org/)</b>
- <b>[Pytorch Lightning](https://www.pytorchlightning.ai/)</b>
- etc..

저자는 여러 특성과 자동화된 코드를 활용하기 위해서 `Pytorch Lightning`을 이용합니다.
<br/>
<br/>

## 📚 데이터 셋   
여기서는 `CoLA`(Corpus of Linguistic Acceptability) 데이터 셋을 이용합니다. 주어지는 문장이 문법적으로 맞는지, 아닌지 2개의 class로 분류하는 작업을 진행해봅시다.

- ❌ `Unacceptable`: 문법적으로 맞지 않음
- ✅ `Acceptable`: 문법적으로 맞음

데이터를 다운로드 하고 로드하기 위해서 [Huggingface datasets](https://huggingface.co/docs/datasets/quicktour.html)을 사용합니다. 이 라이브러리는 800+의 데이터셋을 지원하고 커스텀 데이터도 이용할 수 있습니다.   

아래와 같이 쉽게 다운로드 할 수 있습니다.
~~~python
cola_dataset = load_dataset("glue", "cola")
print(cola_dataset)
~~~
~~~python
DatasetDict({
    train: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 8551
    })
    validation: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1043
    })
    test: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1063
    })
})
~~~

데이터를 한번 살펴봅니다.
~~~python
train_dataset = cola_dataset['train']
print(train_dataset[0])
~~~
~~~json
{
    'idx': 0,
    'label': 1,
    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."
}
~~~
<br/>
<br/>

## 🛒 데이터 불러오기   
데이터 파이프라인은 아래의 도구를 이용해서 생성할 수 있습니다.   

- 🍦 Vanilla Pytorch [DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- ⚡ Pytorch Lightning [DataModules](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html)   
  

`DataModules`가 CPU & GPU에 더 최적화 되어있고 구조화가 잘 되어 있습니다. 가능하다면 `DataModules`을 이용하는 것을 추천합니다.   

`DataModules`은 다음과 같은 interface에 의해 정의됩니다.   
- `prepare_data` (optional) 는 1개의 GPU에서 한번만 호출됩니다. -- 일반적으로 아래에서 설명할 데이터 다운로드하는 것과 같은 작업이 포함될 수 있습니다.   
- `setup`은 각각 GPU에서 호출되며 **fit**또는 **test**단계에 있는지 정의하기 위해서 **stage**를 이용합니다.
- 각 데이터셋을 불러오기 위해서 `train_dataloader`, `val_dataloader` and `test_dataloader`를 이용합니다.

`DataModule`은 PyTorch에서 데이터 처리에 관여하는 5가지 스텝으로 캡슐화 되어 있습니다.
- 전처리 단계(다운로드/ 토큰화/ process)
- 정리하거나 디스크에 저장하기
- Dataset으로 불러오기
- transforms 적용(회전, 토큰화, 기타…)
- DataLoader로 감싸기

프로젝트에서 사용하는 `DataModule`코드는 다음과 같습니다.
~~~python
class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        # processing the data
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )
~~~
<br/>
<br/>

## 🏗️ Lightning을 이용한 모델 구성    
PyTorch Lightning에서 모델은 [LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html)로 구성됩니다. 기본적으로 `torch.nn.Module`(🍦)위에 다양한 function들(🍨)이 올라가 있는 구조 입니다(바닐라 아이스크림 위에 체리!).
이 체리들은 반복되는 코드를 줄일 수 있고 엔지니어링 코드를 머신러닝 코드와 분리할 수 있도록 도와줍니다.

예를들면, 한 epoch에 반복되는 batch에서 어떠한 동작이 이루어지는지를 `training_step`을 이용해서 정의할 수 있습니다.   

모델을 `LightningModule`로 동작하게 하려면 새로운 `class`를 정의하고 몇가지 메서드를 추가해야 합니다.   

`LightningModule`은 다음과 같은 interface에 의해 정의됩니다.
- `init` 모듈의 초기 설정을 정의합니다.
- `forward` 주어지는 입력에 대해서 어떠한 동작을 할지 정의합니다(loss 계산, weight 업데이트는 제외합니다.).
- `training_step` training step으로 loss 계산이나 metric 계산을 수행합니다. weight업데이트는 필요하지 않습니다.
- `validation_step` validation step
- `test_step` test step (optional)
- `configure_optimizers` 어떤 optimizer를 이용할지 설정합니다.   

이외에도 다양한 기능들을 이용할 수 있습니다. 자세한 정보는 [여기](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html)를 확인하세요.   

이 프로젝트에서는 다음과 같이 `LightningModule`을 이용합니다.
~~~python
class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
        val_acc = torch.tensor(val_acc)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
~~~
<br/>
<br/>

## 👟 학습   
`DataLoader`, `LightningModule`는 `Trainer`의해 사용됩니다. `Trainer`는 데이터 로드, gradient 계산, optimizer 로직, 로깅등을 조율해서 처리합니다.   

`Trainer`는 여러가지 옵션들로 로깅, 그라디언트 축적, half precision training, 분산 컴퓨팅 등과 같이 커스텀 할 수 있습니다.   

여기서는 기본적인 예제를 이용하겠습니다.
~~~python
cola_data = DataModule()
cola_model = ColaModel()

trainer = pl.Trainer(
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=1,
    fast_dev_run=False,
)
trainer.fit(cola_model, cola_data)
~~~

`fast_dev_run=True`로 설정하면 training 1스텝,validation 1step으로 진행합니다(True로 설정하는 것이 대부분 좋습니다. validation 스텝에서 잘못된 부분이 바로 발생하기 떄문에 학습이 완료될때까지 기다릴 필요가 없습니다.).   
<br/>
<br/>

## 📝 로깅   
모델 학습을 로깅하는 것은 다음과 같이 간단합니다.
~~~python
cola_data = DataModule()
cola_model = ColaModel()

trainer = pl.Trainer(
    default_root_dir="logs",
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=1,
    fast_dev_run=False,
    logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
)
trainer.fit(cola_model, cola_data)
~~~

`logs/cola`디렉터리가 생성되고 아래의 커맨드를 이용해서 tensorboard로 시각화할 수 있습니다.
~~~bash
tensorboard --logdir logs/cola
~~~
텐서보드는 `http://localhost:6006/`로 접근할 수 있습니다.
<br/>
<br/>

## 🔁 Callback   
`Callback`은 프로젝트에 전반적으로 재활용될 수 있는 내장된 프로그램입니다.

예를들어, **ModelCheckpoint** callback을구현한다고 합니다. 이 callback은 학습된 모델을 저장합니다.   
metric을 모니터링해서 어떤 모델을 저장할지 선택할 수 있습니다(여기서는 `val_loss`를 이용합니다.). 가장 좋은 모델은 `dirpath`에 저장됩니다.   

Callback에 대한 자세한 정보는 [여기](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html)를 참고해주세요.   
~~~python
cola_data = DataModule()
cola_model = ColaModel()

checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="val_loss", mode="min"
)

trainer = pl.Trainer(
    default_root_dir="logs",
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=1,
    fast_dev_run=False,
    logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
    callbacks=[checkpoint_callback],
)
trainer.fit(cola_model, cola_data)
~~~

또한 callback을 여러개 엮을 수 있습니다. `EarlyStopping`callback은 특정 파라미터(여기서는 `val_loss`를 이용합니다.)를 모니터링하면서 모델의 overfit을 방지합니다.   
~~~python
early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
)

trainer = pl.Trainer(
    default_root_dir="logs",
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=1,
    fast_dev_run=False,
    logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
    callbacks=[checkpoint_callback, early_stopping_callback],
)
trainer.fit(cola_model, cola_data)
~~~
<br/>
<br/>

## 🔍 추론   
모델이 학습되고나면 예측을위해서 학습된 모델을 이용할 수 있습니다.   
일반적으로 `Inference`는 다음과 같은 과정을 포함합니다.
- 학습된 모델 로드
- 런타임 입력 얻기
- 입력을 알맞은 포멧으로 변경하기
- 예측값 얻기
~~~python
class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        # loading the trained model
        self.model = ColaModel.load_from_checkpoint(model_path)
        # keep the model in eval mode
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        # text => run time input
        inference_sample = {"sentence": text}
        # tokenizing the input
        processed = self.processor.tokenize_data(inference_sample)
        # predictions
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions
~~~

이 포스트에 나와있는 전체 코드는 다음 링크에서 확인 가능합니다: [Github](https://github.com/graviraja/MLOps-Basics)