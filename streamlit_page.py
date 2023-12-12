"Разведочный анализь данных с помощью streamlit"
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def preload_content():
    """ preload content used in web app """

    main_img = Image.open('images/main.png')

    return main_img


def histogram(df, column):
    fig = go.Figure()

    # Добавляем данные
    feature = df[column]
    fig.add_trace(go.Histogram(x=feature, nbinsx=30, name='Original Data'))

    # Добавляем вертикальные линии для среднего и 3ех стандартных отклонений
    # влево и вправо от среднего
    feature_mean = feature.mean()
    feature_min = feature.min()
    feature_max = feature.max()
    feature_std = feature.std()
    lower = feature_mean - 3*feature_std
    upper = feature_mean + 3*feature_std
    lower_2 = feature_mean - 4*feature_std
    upper_2 = feature_mean + 4*feature_std
    y_max = max(np.histogram(feature, bins=30)[0])
    fig.add_shape(go.layout.Shape(
        type="line", x0=feature_mean, x1=feature_mean, y0=0, y1=y_max,
        line=dict(color="black", width=3)))
    if lower > feature_min:
        fig.add_shape(go.layout.Shape(
            type="line",
            x0=lower,
            x1=lower,
            y0=0,
            y1=y_max,
            line=dict(color="red", width=3, dash='dash')
        ))
    if upper < feature_max:
        fig.add_shape(go.layout.Shape(
            type="line",
            x0=upper,
            x1=upper,
            y0=0,
            y1=y_max,
            line=dict(color="red", width=3, dash='dash')
        ))
    if lower_2 > feature_min:
        fig.add_shape(go.layout.Shape(
            type="line",
            x0=lower_2,
            x1=lower_2,
            y0=0,
            y1=y_max,
            line=dict(color="green", width=3, dash='dash')
        ))
    if upper_2 < feature_max:
        fig.add_shape(go.layout.Shape(
            type="line",
            x0=upper_2,
            x1=upper_2,
            y0=0,
            y1=y_max,
            line=dict(color="green", width=3, dash='dash')
        ))

    # Обновляем layout
    rows_cnt = df[(feature < lower) | (feature > upper)].shape[0]
    fig.update_layout(
        title=f"""
            Распределение признака {column} \n
            3 std: {lower:0.2f} - {upper:0.2f}, {rows_cnt} outliers
        """,
        xaxis_title=f"{column}",
        yaxis_title="Частота",
        bargap=0.05
    )

    # Отображаем фигуру
    st.plotly_chart(fig)


def render_page(main_img):
    """ creates app page with tabs """

    st.title('Отклик клиентов банка')
    st.subheader('Проводим разведочный анализ данных о клиентах банка')
    st.image(main_img, width=1000)

    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'datasets', 'df_result.csv')
    df = pd.read_csv(file_path)
    df.drop('Unnamed: 0', axis=1, inplace=True)

    st.subheader('**1. Посмотрим на характеристики признаков**')
    num_col = df.select_dtypes(include='number')
    binary_col = num_col.columns[num_col.nunique() == 2]
    only_numeric = list(set(num_col) - set(binary_col) - {'AGREEMENT_RK'})

    st.write('**Числовые признаки**')
    st.table(df[only_numeric].describe())
    st.write((
        '- `AGE` - в выборке представлены клиенты в возрасте 21 - 67 лет. '
        '50% клиентов имеют возраст в интервале от 30 до 50;\n'
        '- `CHILD_TOTAL` - в выборке представлены в т.ч. многодетные семьи '
        '(до 10 детей). Но минимум у 75% клиентов менее 3х детей;\n'
        '- `CREDIT` - выглядит, как будто суммы последних кредитов '
        'соразмерны заработным платам клиентов. Максимальная сумма кредита '
        '- 119 тыс.руб. В 75% случаев суммы не превышают 12 тыс.руб;\n'
        '- `DEPENDANTS` - клиенты имеют максимум 7 иждивенцев. Это меньше, '
        'чем количество детей. Вероятно, некоторые дети уже '
        'совершеннолетние;\n'
        '- `FST_PAYMENT` - странно, что максимальный первый платеж по '
        'кредиту в выборке превышает максимальную сумму кредита. '
        'На такие случаи нужно смотреть дополнительно;\n'
        '- `LOAN_NUM_TOTAL`, `LOAN_NUM_CLOSED` - видимо, выборка была взята '
        'из списка клиентов, бравших в банке кредит, т.к. все клиенты брали '
        'ранее кредиты. Максимальное количество редитов - 11;\n'
        '- `OWN_AUTO` - у подавляющего большинства клиентов нет автомобилей. '
        'Максимальное число автомобилей - 2;\n'
        '- `PERSONAL_INCOME` - 75% клиентов зарабатывают менее 17 тыс.руб. '
        'Максимальная сумма дохода - 250 тыс.руб;\n'
        '- `TERM` - в 75% случаев кредиты краткосрочные (3-10 месяцев). '
        'Максимальный срок кредита - 3 года;'
    ))
    st.write('**Бинарные признаки**')
    st.table(df[binary_col].describe())
    st.write("""
        - `FL_PRESENCE_FL` - только у 31% клиентов есть в собственности жилье;
        - `GENDER` - в выборке чаще встречаются мужчины (65%);
        - `SOCSTATUS_WORK_FL` - большинство клиентов (91%) работают;
        - `SOCSTATUS_PENS_FL` - около 13% клиентов уже вышли на пенсию;
        - `TARGET` - процент откликов на рекламную компанию не высок - 12%;
    """)
    st.write('**Категориальные признаки**')
    st.dataframe(df.describe(include=object))
    st.write((
        '- `ADDRESS_PROVINCE` - есть 3 признака с регионом (фактический, '
        'регистрации и почтовый). Есть строки, где они отличаются, поэтому '
        'пока не удаляем;\n'
        '- `FAMILY_INCOME` - семейный доход измеряется 5-ю категориями. '
        'У большинства семей (46%) доход находится в интервале от 10 до 20 '
        'тыс.руб;\n'
        '- `EDUCATION` - в данных 7 уровней образования. Около 43% клиентов '
        'имеют среднее специальное образование;\n'
        '- `MARITAL_STATUS` - есть 5 видов семейного положения. '
        'Около 62% клиентов состоят в законном браке;'
    ))

    st.divider()

    st.subheader('**2. Посмотрим на корреляцию признаков**')

    num_cols = df.columns[df.dtypes != 'object'].tolist()
    num_cols.remove('AGREEMENT_RK')
    df_corr = np.round(df[num_cols].corr(), 2)
    fig = px.imshow(df_corr, text_auto=True, aspect='auto', width=1000)
    st.plotly_chart(fig, theme=None)

    st.write((
        'Самые скоррелированные признаки:\n'
        '- `SOCSTATUS_WORK_FL` и `SOCSTATUS_PENS_FL`: обратная зависимость. '
        'Только некоторые пенсионеры работают после выхода на пенсию;\n'
        '- `LOAN_NUM_TOTAL` и `LOAN_NUM_CLOSED`: прямая зависимость. '
        'Чем больше кредитов берут пользователи, тем больше из них закрытых. '
        'Корреляция 0.86, при построении модели можно подумать о том, '
        'чтобы удалить один из признаков;\n\n'
        'С целевой переменной все признаки плохо коррелируют.'
    ))
    st.divider()

    st.subheader('**3. Посмотрим на распределение признаков**')

    histogram(df, 'AGE')
    st.write((
        'В выборке представлены клиенты от 20 до 67 лет. '
        'Средний возраст в выборке - 40 лет.\n\n'
        'В данных нет каких-то аномалий по возрасту'
    ))
    histogram(df, 'PERSONAL_INCOME')
    st.write((
        'Большинство значений лежит в диапазоне до 30 тыс.\n\n'
        'В данных есть аномалии. Доход свыще 50 тыс.находится на '
        'расстоянии более 4-х стандартных отклонений'
    ))
    histogram(df, 'CREDIT')
    st.write((
        'Большинство значений лежит в диапазоне до 30 тыс. '
        'Средний доход около 13 тыс\n\n'
        'В данных есть аномалии. Доход свыше 50 тыс.находится на '
        'расстоянии более 4-х стандартных отклонений'
    ))

    st.divider()

    st.subheader('**4. Посмотрим отдельно на некоторые признаки**')

    st.write('**EDUCATION**')
    df_group = df.groupby(
        ['EDUCATION', 'GENDER'],
        as_index=False
    ).TARGET.count()
    male = df_group[df_group['GENDER'] == 1].sort_values('EDUCATION')
    female = df_group[df_group['GENDER'] == 0].sort_values('EDUCATION')
    labels = male.EDUCATION
    fig = make_subplots(
        1,
        2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=['Male', 'Female']
    )
    fig.add_trace(go.Pie(
        labels=labels,
        values=male['TARGET'],
        scalegroup='one',
        name="Education Male"
    ), 1, 1)
    fig.add_trace(go.Pie(
        labels=labels,
        values=female['TARGET'],
        scalegroup='one',
        name="Education Female"
    ), 1, 2)
    st.plotly_chart(fig, theme=None)
    st.write((
        'Распределение уровней образования среди клиентов женского и '
        'мужского пола примерно одинаково.\n\n'
        'Мы можем заметить, что среди мужчин немного чаще встречаются '
        'лица с высшим образованием, а среди женщин со средним.'
    ))

    st.divider()

    st.write('**MARITAL_STATUS**')

    df_group = df.groupby(
        ['MARITAL_STATUS', 'GENDER'],
        as_index=False
    ).TARGET.count()
    male = df_group[df_group['GENDER'] == 1].sort_values('MARITAL_STATUS')
    female = df_group[df_group['GENDER'] == 0].sort_values('MARITAL_STATUS')
    labels = male.MARITAL_STATUS
    fig = make_subplots(
        1,
        2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=['Male', 'Female']
    )
    fig.add_trace(go.Pie(
        labels=labels,
        values=male['TARGET'],
        scalegroup='one',
        name="Marital status Male"
    ), 1, 1)
    fig.add_trace(go.Pie(
        labels=labels,
        values=female['TARGET'],
        scalegroup='one',
        name="Marital status Female"
    ), 1, 2)
    st.plotly_chart(fig, theme=None)
    st.write((
        'Среди женщин значительно больше тех, кто состоит в браке.\n\n'
        'Среди мужчин, в свою очередь, больше вдовцов, '
        'разведенных и никогда не состоявших в браке.'
    ))

    df_group = df.groupby(
        ['MARITAL_STATUS', 'TARGET'],
        as_index=False
    ).agg(COUNT=('AGREEMENT_RK', 'count'))
    df_group['TARGET'] = df_group['TARGET'].apply(
        lambda x: 'Отклик' if x == 1 else 'Нет отклика'
    )
    fig = px.histogram(
        df_group,
        x="MARITAL_STATUS",
        y="COUNT",
        color="MARITAL_STATUS",
        pattern_shape="TARGET",
        width=1000
    )
    st.plotly_chart(fig, theme=None)

    st.divider()

    st.write('**FAMILY_INCOME**')
    df_group = df.groupby(
        ['FAMILY_INCOME', 'TARGET'],
        as_index=False
    ).agg(COUNT=('AGREEMENT_RK', 'count'))
    df_group['TARGET'] = df_group['TARGET'].apply(
        lambda x: 'Отклик' if x == 1 else 'Нет отклика'
    )
    fig = px.histogram(
        df_group,
        x="FAMILY_INCOME",
        y="COUNT",
        color="FAMILY_INCOME",
        pattern_shape="TARGET",
        width=1000,
        category_orders={'FAMILY_INCOME': [
            'до 5000 руб.',
            'от 5000 до 10000 руб.',
            'от 10000 до 20000 руб.',
            'от 20000 до 50000 руб.',
            'свыше 50000 руб.'
        ]}
    )
    st.plotly_chart(fig, theme=None)

    st.divider()

    st.write('**AGE**')
    fig = go.Figure(layout=dict(width=1000))
    fig.add_trace(go.Box(x=df[df['TARGET'] == 1].AGE, name='Отклик'))
    fig.add_trace(go.Box(x=df[df['TARGET'] == 0].AGE, name='Нет отклика'))
    fig.update_layout(xaxis_title='Возраст клиента')
    st.plotly_chart(fig, theme=None)
    st.write((
        'На графике видно, что откликаются на маркетинговые '
        'предложения чаще более молодые клиенты. \n\n'
        'Медианный возраст откликнувшихся - 35, а неоткликнувшихся - 40.'
    ))

    st.divider()

    st.write('**PERSONAL_INCOME**')
    fig = go.Figure(layout=dict(width=1000))
    fig.add_trace(go.Box(
        x=df[df['TARGET'] == 1].PERSONAL_INCOME,
        name='Отклик'
    ))
    fig.add_trace(go.Box(
        x=df[df['TARGET'] == 0].PERSONAL_INCOME,
        name='Нет отклика'
    ))
    fig.update_layout(xaxis_title='Доход клиента')
    st.plotly_chart(fig, theme=None)
    st.write((
        'На графике видно, что средний доход откликнувшихся несколько '
        'выше среднего дохода тех, кто проигнорировал предложение банка.'
    ))


def load_page():
    """ loads main page """

    main_img = preload_content()

    st.set_page_config(layout="wide",
                       page_title="Отклик клиентов",
                       page_icon=':bank:')

    render_page(main_img)


if __name__ == "__main__":
    load_page()
