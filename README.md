# COVID-19

## 文献

1. Review
   - [Review of Artificial Intelligence Techniques in Imaging Data Acquisition, Segmentation and Diagnosis for COVID-19](https://arxiv.org/abs/2004.02731)
2. Radiomics
   - [Machine learning-based CT radiomics model for predicting hospital stay in patients with
     pneumonia associated with SARS-CoV-2 infection: A multicenter study](https://www.medrxiv.org/content/medrxiv/early/2020/03/03/2020.02.29.20029603.full.pdf)

3. Diagnose
   - [Deep Learning-based Detection for COVID-19 from Chest CT using Weak Label](https://www.medrxiv.org/content/medrxiv/early/2020/03/26/2020.03.12.20027185.full.pdf)
   - [Rapid ai development cycle for the coronavirus (covid-19) pandemic: Initial results for automated detection & patient monitoring using deep learning ct image analysis](https://arxiv.org/abs/2003.05037)
   - [Artificial intelligence distinguishes covid-19 from community acquired pneumonia on chest ct](https://pubs.rsna.org/doi/abs/10.1148/radiol.2020200905)
   - [Deep learning-based model for detecting 2019 novel coronavirus pneumonia on high-resolution computed tomography: a prospective study](https://www.medrxiv.org/content/10.1101/2020.02.25.20021568v2.abstract)
   - [AI-assisted CT imaging analysis for COVID-19 screening: Building and deploying a medical AI system in four weeks](https://www.medrxiv.org/content/10.1101/2020.03.19.20039354v1.abstract)
   - [Lung Infection Quantification of COVID-19 in CT Images with Deep Learning](https://arxiv.org/abs/2003.04655)
   - [Serial Quantitative Chest CT Assessment of COVID-19: Deep-Learning Approach](https://pubs.rsna.org/doi/abs/10.1148/ryct.2020200075)

## 环境配置

```bash
cd script
pip install -r requirements.txt
```
对于显卡有tensor core(图灵架构2080ti)的用户建议安装[apex](https://github.com/apex/apex)进行加速.
只要在train.py中设置‘apexType’参数为‘O1’,默认为‘O0’. 

## 训练和测试

```bash
python ./train.py
python ./evaluate.py
```

## 性能
- 测试集--整个病例　F1:0.8745, ACC:0.8632, Sen:0.9018, Spe:0.8200
- 测试集--单肺　F1:0.8591, ACC:0.8465, Sen:0.8873, Spe:0.8011