import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def input_section():
    product_price_input = st.text_input("Ціна товару за одиницю", placeholder=148, value=148)
    product_price_output = st.text_input("Ціна збуту товару за одиницю", placeholder=280, value=280)
    lost_profit = st.text_input("Втрачена вигода від відсутності товару", placeholder=170, value=170)
    product_quantity = st.text_input("Розмір закупівель (в шт. через пробіл)", placeholder='5 6 7 8 9 10 11', value='5 6 7 8 9 10 11').split()
    probabilities = st.text_input("Ймовірності попиту (від 0 до 1 через пробіл)", placeholder='0.1 0.1 0.2 0.2 0.2 0.1 0.1', value='0.1 0.1 0.2 0.2 0.2 0.1 0.1').split()

    return product_price_input, product_price_output, lost_profit, product_quantity, probabilities


def input_lost_profit_range():
    start_lost_profit = st.number_input("Початкове значення втраченої вигоди", value=0)
    end_lost_profit = st.number_input("Кінцеве значення втраченої вигоди", value=200)
    step_lost_profit = st.number_input("Крок втраченої вигоди", value=5)

    return start_lost_profit, end_lost_profit, step_lost_profit


def all_needed_values_were_provided(list_of_values):
    if all([value for value in list_of_values]):
        return True
    return False


def cast_input_values(input_values):
    product_price_input, product_price_output, lost_profit, product_quantity, probabilities = input_values

    product_price_input = int(product_price_input)
    product_price_output = int(product_price_output)
    lost_profit = int(lost_profit)

    product_quantity = list(map(int, product_quantity))
    probabilities = list(map(float, probabilities))

    return product_price_input, product_price_output, lost_profit, product_quantity, probabilities


def calculate_and_fill_matrix(product_price_input, product_price_output, lost_profit, product_quantity, probabilities):
    matrix_df = pd.DataFrame(index=product_quantity, columns=product_quantity)

    for i in product_quantity:
        for j in product_quantity:
            if j <= i:
                matrix_df.loc[i, j] = (product_price_output * j) - (product_price_input * i)
            else:
                matrix_df.loc[i, j] = (product_price_output * i) - (product_price_input * i) - ((j - i) * lost_profit)

    matrix_df['mean'] = matrix_df.iloc[:, :len(product_quantity)].apply(
        lambda x: np.round(np.dot(x.astype(float), probabilities), 2), axis=1
    )

    matrix_df['deviation'] = matrix_df.iloc[:, :len(product_quantity)].apply(
        lambda x: np.round(np.dot((x - matrix_df['mean']) ** 2, probabilities), 2), axis=1
    )
    matrix_df['std'] = np.round(np.sqrt(matrix_df['deviation']), 2)

    return matrix_df


def calculate_optimal_strategy(matrix_df):
    matrix_df['mean_to_std'] = matrix_df['mean'] / matrix_df['std']
    best_optimized_index = matrix_df['mean_to_std'].idxmax()
    matrix_df.drop('mean_to_std', axis=1, inplace=True)

    return best_optimized_index


def build_progresion_plot(product_price_input, product_price_output, product_quantity, probabilities, lost_profit_range):
    mean_values_df = pd.DataFrame()

    for lp in lost_profit_range:
        mean_values = calculate_and_fill_matrix(product_price_input, product_price_output, lp, product_quantity, probabilities)['mean']
        mean_values_df = pd.concat([mean_values_df, pd.DataFrame(mean_values).T], ignore_index=True)

    mean_values_df.columns = list(map(lambda x: 'Закупівля ' + str(x) + ' шт.', product_quantity))
    mean_values_df.index = lost_profit_range
    st.write(mean_values_df.T)
    st.line_chart(mean_values_df)


def main():
    st.title('Оптимізація закупівель')

    st.subheader('Введіть дані для розрахунку')
    input_values = input_section()

    if not all_needed_values_were_provided(input_values):
        st.warning("Будь ласка, заповніть всі поля")
        return

    st.subheader('Результати розрахунку')
    product_price_input, product_price_output, lost_profit, product_quantity, probabilities = cast_input_values(input_values)


    probabilities_df = pd.DataFrame(probabilities, index=product_quantity, columns=['Ймовірність'])
    matrix_df = calculate_and_fill_matrix(product_price_input, product_price_output, lost_profit, product_quantity, probabilities)
    matrix_df.columns = list(map(lambda x: 'Попит ' + str(x), product_quantity)) + ['mean', 'deviation', 'std']
    matrix_df.index = list(map(lambda x: 'Закупівля ' + str(x), product_quantity))
    best_optimized_index = calculate_optimal_strategy(matrix_df)

    st.dataframe(probabilities_df.T)
    st.dataframe(matrix_df)
    st.success(f"Найкращий оптимізований варіант: **{best_optimized_index }** із значенням **{matrix_df.loc[best_optimized_index, 'mean']}**")

    st.subheader('Графік залежності середнього значення від втраченої вигоди')
    start_lost_profit, end_lost_profit, step_lost_profit = input_lost_profit_range()
    lost_profit_range = np.arange(start_lost_profit, end_lost_profit, step_lost_profit)
    build_progresion_plot(product_price_input, product_price_output, product_quantity, probabilities, lost_profit_range)


if __name__ == '__main__':
    main()