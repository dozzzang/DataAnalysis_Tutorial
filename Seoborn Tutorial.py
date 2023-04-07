# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# -*- coding: utf-8 -*-

# %matplotlib inline


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
# -

# 한글처리
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 13

# # Plotting with numerical data

#tips는 seaborn이 제공하는 데이터
tips = sns.load_dataset("tips")
tips.head(5)

#Scatter Plots
sns.scatterplot(x="total_bill",y="tip",data=tips)

#hue
sns.scatterplot(x="total_bill",y="tip",hue="smoker",data=tips)

#hue
sns.scatterplot(x="total_bill",y="tip",hue="day",data=tips)

#hue를 수로 했을 때?
sns.scatterplot(x="total_bill",y="tip",hue="total_bill",data=tips)

#hue의 style argument (marker style을 의미)
sns.scatterplot(x="total_bill",y="tip",hue="smoker",style="smoker",data=tips)

#marker size control
sns.scatterplot(x="total_bill",y="tip",size="size",data=tips)

#marker의 상한과 하한
#relplot의 도입 seaborn.relplot()이나 seaborn.scatterplot()은 뭔가를 return함.
#replot의 default는 lineplot이며 lineplot과 relplot 혼합 지원 하지만 단일도 지원
#하기 때문에 relplot으로 사용하자!!
sns.relplot(x="total_bill",y="tip",hue="smoker",size="size",
           sizes=(15,200),data=tips,kind='scatter')

# <seaborn.axisgrid.FacetGrid at 0x241e8e83dc0>
# relplot() 함수가 return하는 변수의 설명, at 뒤는 메모리 주소!

#무언갈 return한다고 했기 때문에 변수에 넣어주자. 
#근데 안나옴. 이미 그려진 g위에 제목을 추가하는 코드이기 때문에
g = sns.relplot(x="total_bill",y="tip",hue="smoker",size="size",
                sizes=(15,200),data=tips,kind='scatter')
g = g.set_titles('scatter plot example')

# # Utils

g = sns.relplot(x="total_bill",y="tip",hue="smoker",size="size",
               sizes=(15,200), data=tips, kind='scatter')
plt.title('scatter plot example')

# +
#scatterplot보다 relplot이 savefig기능을 구현할 때 코드 절약!
#근데 g1은 default인데 왜 scatter임?
g0 = sns.relplot(x="total_bill",y="tip",hue="smoker",
                size="size",data=tips,kind='scatter')
g1 = sns.relplot(x="total_bill",y="tip",size="size",
                sizes=(15,200),data=tips)

g0.savefig('total_bill_tip_various_color_by_size.png')
g1.savefig('total_bill_tip_various_color_by_size.png')
# -

#pandas df plot savefig를 이용하기 위해 Figure을 만들어줘야해!
#return을 변수로 받지 않았음에 주목! matplotlib은 항상 마지막에 그린 그림을..
#df plot함수의 return type : Not figure but AxesSubplot
g = tips.plot(x = 'total_bill',y='tip',kind='scatter',title='pandas plot example')
g = g.get_figure()
g.savefig('pandas_plot_example.png')

#바로 저장 근데 쓰지말자 혼동 o
ax = tips.plot(x='total_bill',y='tip',kind='scatter',title='pandas plot example')
plt.savefig('pandas_plot_example_2.png')

#relplot() 두 번 이용 -> 각각의 그림 scatterplot() -> 그림 겹치기
#random noise data create
data = {
    'x': np.random.random_sample(100) * 50,
    'y': np.random.random_sample(100) * 10
}
random_noise_df = pd.DataFrame(data, columns=['x','y'])
random_noise_df.head(5)

#메모리 주소도 당연히 같다
g0 = sns.scatterplot(x = "total_bill",y="tip",hue='smoker',
                    alpha=0.8,size="size",sizes=(15,200),data=tips)
g1 = sns.scatterplot(x="x",y="y",alpha=0.2,color='g',data=random_noise_df)

#Matplotlib은 현재 Figure가 닫히지 않으면 계속 그 Figure위에 덧그리는 형식!!
g0 = sns.scatterplot(x="total_bill", y="tip", hue='smoker',
    alpha=0.8, size="size", sizes=(15, 200), data=tips)
plt.close() #요걸로 닫아주기
g1 = sns.scatterplot(x="x", y="y", alpha=0.2, color='g', data=random_noise_df)

#제목을 추가하여 figure로 다시 만들기?
g0.set_title('total bill ~ tip scatter plot')
g0.get_figure()

#g0,g1 메모리 주소 다르다.
g1.set_title('random noise')
g1.get_figure()

#relplot() 함수 호출하니 이전에 그리던 Figure 모두 닫힘을 알 수 있음.
#교훈 : 새 그림을 그릴 때에는 습관적으로 close()함수를 호출해주자.
g0 = sns.relplot(x="total_bill",y="tip",hue='smoker',
                alpha=0.8, size="size",sizes=(15,200), data=tips)
g1 = sns.scatterplot(x="x",y="y",alpha=0.2,color='g',data=random_noise_df)

# # Plotting with numerical data 2

# # line Plots

#시계열 데이터는 line plot이쥐 *cumsum()함수 : 값 누적 함수
data = {
    'time': np.arange(500),
    'value':np.random.randn(500).cumsum()
}
df = pd.DataFrame(data)
df.head(5)

g = sns.lineplot(x="time",y="value",data=df)

#lineplot()과 replot()에서 kind를 'line'으로 정의하는 것의 결과는 같지만 
#return type이 다르다.
#x 중심 
g = sns.relplot(x="time",y="value",kind="line",data=df)

#2차원 데이터 500개 생성 (데이터 정렬 시각화를 위해)
data = np.random.randn(500,2).cumsum(axis=0)
df = pd.DataFrame(data,columns=["x","y"])
df.head(5)

g = sns.relplot(x="x",y="y",sort=False,kind="line",data=df)

g = sns.relplot(x="x", y="y", sort=True, kind="line", data=df)

#lineplot은 신뢰구간과 추정회귀선을 손쉽게 그려준다.
#fmri 데이터는 대상자(subject)의 활동(event)에 따라 시점(timepoint)별로 fmri의
#측정값 중 하나의 센서값을 정리한 시계열 데이터
fmri = sns.load_dataset("fmri")
fmri.head(5)

#lineplot의 default는 신뢰 구간과 추정 회귀선
g = sns.relplot(x="timepoint",y="signal",kind="line",data=fmri)

#confidence interval의 약자 ci
g = sns.relplot(x="timepoint",y="signal",kind="line",data=fmri,ci=None)

#표준편차를 이용하여 confidence interval을.. (standard deviation)
g = sns.relplot(x = "timepoint",y="signal",kind="line",data=fmri,ci="sd")

#추정 회귀선이 없으니 마치 주파수
g = sns.relplot(x="timepoint",y="signal",kind="line",data=fmri,estimator=None)

g = sns.relplot(x="timepoint",y="signal",hue="event",kind="line",data=fmri)

g = sns.relplot(x="timepoint",y="signal",hue="event",
               style="event",kind="line",data=fmri)

g = sns.relplot(x="timepoint",y="signal",hue="region",style="event",
               markers=True,kind="line",data=fmri)

#선의 색은 region하지만 각 선은 subject에 대해 중복으로 그릴 경우
#estimator=None으로 설정하지 않으면 syntax error
#가독성 매우 낮음
g = sns.relplot(x="timepoint", y="signal", hue="region",
    units="subject", estimator=None,kind="line",
    data=fmri.query("event == 'stim'"))

# # Date data

data = {
    'time' : pd.date_range("2023-04-07",periods=500),
    'value':np.random.randn(500).cumsum()
}
df = pd.DataFrame(data)
df.head(5)

g = sns.relplot(x="time",y="value",kind="line",data=df)

g = sns.relplot(x="time",y="value",kind="line",data=df)
g.fig.autofmt_xdate()

#scatterplot의 return type은 AxesSubplot relplot의 return type은 FacetGrid
#aspect:비 height : 각각그래프의 높이
g = sns.relplot(x="timepoint",y="signal",hue="event",style="event",
               col="subject",col_wrap=5,height=3,aspect=.75,linewidth=2.5,
               kind="line",data=fmri.query("region == 'frontal'"))

#col_oreder과 row_order
col_order = [f's{i}' for i in range(14)]
g = sns.relplot(x="timepoint",y="signal",hue="event",style="event",
               col="subject",col_wrap=5,height=3,aspect=.75,linewidth=2.5,
               kind="line",data=fmri.query("region == 'frontal'"),
                col_order=col_order
               )

g = sns.relplot(x="total_bill",y="tip",hue="smoker",data=tips,col="time")

g = sns.relplot(x="total_bill", y="tip", hue="smoker",
    data=tips, col="time", row="sex")

# # Plotting with categorical data


