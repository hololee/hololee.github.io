---
title: 포스트 작성시 Typora 이미지 업로드 기능 이용하기
date: 2022-09-13 22:32:00 +0900
categories: []
tags: []     # TAG names should always be lowercase
# pin: true
# mermaid: true
# math: true
# toc: false     # Table Of Contents
# comments: false 
img_path: /
---



~~~mermaid
graph LR
A --> B
subgraph VPC
style VPC fill:#ffdcb4,stroke:#ff7a11,stroke-width:1px
B[ELB] --> C[nginx]
B[ELB] --> D[nginx]
B[ELB] --> E[nginx]
subgraph "Server (EC2)"
C --> F["WAS (Fastapi)"]
end
subgraph "Server (EC2)"
D --> G["WAS (Fastapi)"]
end
subgraph "Server (EC2)"
E --> H["WAS (Fastapi)"]
end
end
~~~

