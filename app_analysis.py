import pickle

complexity_dictionary_binomial = {0: ['n', '1', 'logn'], 1: ['n_square', 'nlogn']}

complexity_dictionary_multinomial = {0: 'n', 1: 'n_square', 2: '1', 3: 'nlogn', 4: 'logn'}


class AppAnalysis:
    def __init__(self):
        pass

    def calculate_complexity_binomial(self, values_from_file):
        file_logistic_regression_binomial = open('models_regression_store/logistic_regression_binomial.pkl', 'rb')
        model_logistic_binomial = pickle.load(file_logistic_regression_binomial)
        predicted = model_logistic_binomial.predict(values_from_file)
        return complexity_dictionary_binomial[predicted.tolist()[0]]

    def calculate_complexity_multinomial(self, values_from_file):
        file_logistic_regression_multinomial = open('models_regression_store/logistic_regression_multinomial.pkl', 'rb')
        model_logistic_multinomial = pickle.load(file_logistic_regression_multinomial)
        predicted = model_logistic_multinomial.predict(values_from_file)
        return complexity_dictionary_multinomial[predicted.tolist()[0]]
